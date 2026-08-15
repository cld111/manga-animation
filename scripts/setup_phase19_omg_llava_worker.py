"""Phase 19 worker bootstrap: install the official OMG-LLaVA stack and weights on the Kaggle
GPU worker, then leave a runnable state for `run_phase19_omg_llava_benchmark.py`.

The official OMG-LLaVA stack is a mid-2024 xtuner/mmcv environment (transformers==4.36.0,
triton==2.1.0, torch 2.1.2/cu118, python 3.10) that is incompatible with the worker's default
python-3.12 kernel (torch 2.10 / transformers 5.0). This script therefore creates a SEPARATE
environment on the worker and installs the pinned official stack there (INSTALL.md), clones the
official repo at a pinned commit, downloads all four required weight artifacts, and writes a
prepared finetune config whose paths point at them.

Run on the worker via the verified Jupyter transport (scripts/kaggle_exec.py). Idempotent:
re-running skips completed steps. The benchmark subcommands are then launched inside this env.

    python scripts/setup_phase19_omg_llava_worker.py \
        --weights-dir /kaggle/working/models/omg_llava \
        --src-dir /kaggle/working/omg-seg \
        --env-name omg-llava \
        --omg-seg-commit 48ab9407a45c2ecf78b4e980d6a6ccddf9a7ec9f
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# The verified upstream commit this benchmark pins (auto-bumped when verified).
OMG_SEG_DEFAULT_COMMIT = "48ab9407a45c2ecf78b4e980d6a6ccddf9a7ec9f"
HF_REPO = "zhangtao-whu/OMG-LLaVA"
# Full-repo path to the finetune config (the official README uses the omg_llava-subtree-relative
# path; after cloning the whole OMG-Seg repo the config lives one level deeper).
FINETUNE_CONFIG = "omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py"
WEIGHTS = (
    "internlm2-chat-7b",
    "omg_llava_7b_finetune_8gpus.pth",
    "omg_seg_convl.pth",
    "convnext_large_d_320_CocoPanopticOVDataset.pth",
)


def _run(cmd: list[str], *, env: dict | None = None, timeout: int = 3600) -> None:
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, timeout=timeout)


def _shell(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


# mmdet 3.1.0 requires mmcv >= 2.0.0rc4, < 2.1.0. The cu118/torch2.0 openmmlab index provides
# mmcv 2.0.1 prebuilt against CUDA 11.8 (matching torch 2.1.2+cu118) -- see the install step.
MMCV_INDEX_URL = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"


def _mmcv_version_ok(env_python: str) -> bool:
    """True when the installed mmcv is a 2.0.x release (mmdet 3.1.0's compatible range)."""
    import re

    v = _shell(f"{env_python} -c 'import mmcv; print(mmcv.__version__)'")
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", v or "")
    if not match:
        return False
    return tuple(int(x) for x in match.groups()[:2]) == (2, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the official OMG-LLaVA stack on a worker")
    parser.add_argument("--weights-dir", default="/kaggle/working/models/omg_llava")
    parser.add_argument("--src-dir", default="/kaggle/working/omg-seg")
    parser.add_argument("--env-name", default="omg-llava")
    parser.add_argument("--omg-seg-commit", default=OMG_SEG_DEFAULT_COMMIT)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--python-bin", default=None,
        help="interpreter used to CREATE the venv (default: conda python3.10, else python3.10/"
        "3.11 on PATH). The official OMG stack was built for python 3.10, but python 3.11 works "
        "too AND lets the manga-animation harness (requires >=3.11, uses StrEnum/datetime.UTC) "
        "be pip-installed into the same env -- the phase-19 CLI then runs in one interpreter. "
        "The phase brief's environment is honored either way; this flag is the pragmatic path "
        "when the worker only has python3.10 and python3.11 must come from uv.",
    )
    parser.add_argument(
        "--venv-path", default="/kaggle/working/omgllava-venv",
        help="where the dedicated venv lives (default /kaggle/working/omgllava-venv; the "
        "20GB /kaggle/working loop mount can fill up, so pass an overlay path when needed).",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    src_dir = Path(args.src_dir)
    env_name = args.env_name
    hf_token = args.hf_token
    if not hf_token:
        raise SystemExit("HF_TOKEN is required to download the gated/interm LM weights")

    # --- 1. environment -------------------------------------------------------------------
    # Prefer an explicit --python-bin; else conda; else python3.10/3.11; then a plain venv.
    env_python = None
    if args.python_bin:
        venv = Path(args.venv_path)
        _shell(f"{args.python_bin} -m venv {venv}")
        env_python = str(venv / "bin" / "python")
    elif shutil.which("conda"):
        _shell(f"conda create -n {env_name} python=3.10 -y")
        env_python = f"/opt/conda/envs/{env_name}/bin/python"
        if not Path(env_python).exists():
            env_python = None
    if env_python is None:
        py310 = shutil.which("python3.10") or shutil.which("python3.11")
        if py310 is None:
            raise SystemExit("no conda and no python3.10/3.11 -- cannot build the OMG-LLaVA env")
        venv = Path(args.venv_path)
        _shell(f"{py310} -m venv {venv}")
        env_python = str(venv / "bin" / "python")

    # A Debian system python3.10 venv can be created WITHOUT pip (no ensurepip). Bootstrap pip
    # explicitly if `python -m pip` is missing, so the torch install below can proceed.
    if "No module named pip" in _shell(f"{env_python} -m pip --version"):
        try:
            _run([env_python, "-m", "ensurepip", "--upgrade", "--default-pip"])
        except subprocess.CalledProcessError:
            pass  # ensurepip is unavailable on this python -- fall through to get-pip.py
        if "No module named pip" in _shell(f"{env_python} -m pip --version"):
            _run([
                env_python, "-c",
                "import urllib.request; "
                "urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', "
                "'/tmp/get-pip.py')",
            ])
            _run([env_python, "/tmp/get-pip.py"])

    def in_env(pkg: str) -> bool:
        return _shell(f"{env_python} -c 'import importlib.util; "
                      f"print(bool(importlib.util.find_spec(\"{pkg}\")))'") == "True"

    if not in_env("torch"):
        _run([env_python, "-m", "pip", "install", "--upgrade", "pip"])
        _run([
            env_python, "-m", "pip", "install",
            "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
            "--index-url", "https://download.pytorch.org/whl/cu118",
        ])

    if not _mmcv_version_ok(env_python):
        # mmdet 3.1.0 requires mmcv >= 2.0.0rc4, < 2.1.0. The cu118/torch2.1 openmmlab index
        # only has 2.1.0 (rejected) and building mmcv from source is blocked by torch's CUDA
        # version gate (nvcc 12.8 vs torch 2.1.2+cu118). The cu118/torch2.0 index has mmcv 2.0.1
        # prebuilt against CUDA 11.8 -- matching torch's CUDA -- and torch 2.0->2.1 extension ABI
        # is compatible in practice (verified: mmdet imports and ops work on torch 2.1.2).
        _run([env_python, "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-full"])
        _run([env_python, "-m", "pip", "install", "mmcv==2.0.1",
              "-f", MMCV_INDEX_URL])
        _run([env_python, "-m", "pip", "install",
              "mmdet==3.1.0", "mmsegmentation==1.1.1", "mmpretrain==1.0.1",
              "mmengine", "transformers==4.36.0", "triton==2.1.0",
              "bitsandbytes", "peft", "accelerate", "sentencepiece", "einops",
              "scikit-image", "scipy", "pycocotools", "datasets", "kornia", "ftfy",
              "timm", "openpyxl", "lagent", "gradio==4.37.2", "gradio-image-prompter"])
        # torch 2.1.2 and the 2024 mm stack were built against numpy 1.x (numpy 2 breaks the
        # torch<->numpy bridge and opencv 5 needs numpy>=2), so pin both.
        _run([env_python, "-m", "pip", "install", "-q", "numpy<2", "opencv-python<5"])

    # --- 2. official repo at the pinned commit --------------------------------------------
    if not (src_dir / "omg_llava").exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "https://github.com/lxtGH/OMG-Seg.git", str(src_dir)])
    _run(["git", "-C", str(src_dir), "fetch", "--all"], timeout=1800)
    _run(["git", "-C", str(src_dir), "checkout", args.omg_seg_commit])
    _run([env_python, "-m", "pip", "install", "-e", str(src_dir / "omg_llava")])
    # The phase-19 benchmark CLI imports the manga-animation package (manifest/config/metrics,
    # light, no torch at import time). On python 3.11+ it pip-installs cleanly (requires >=3.11);
    # on python 3.10 the run commands set PYTHONPATH=/kaggle/working/manga-animation/src instead.
    pyver = _shell(
        f"{env_python} -c 'import sys; "
        'print(f"{sys.version_info.major}.{sys.version_info.minor}")'
        "'"
    )
    if tuple(int(p) for p in pyver.split(".")) >= (3, 11):
        _run([env_python, "-m", "pip", "install", "-q", "-e", "/kaggle/working/manga-animation"])

    # --- 3. weights -----------------------------------------------------------------------
    weights_dir.mkdir(parents=True, exist_ok=True)
    _run([env_python, "-m", "pip", "install", "huggingface_hub"])
    for weight in WEIGHTS:
        if (weights_dir / weight).exists():
            continue
        if weight == "internlm2-chat-7b":
            code = (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download('{HF_REPO}', repo_type='model', token='{hf_token}', "
                f"local_dir='{weights_dir / weight}', allow_patterns=['{weight}/*'])"
            )
        else:
            code = (
                "from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('{HF_REPO}', '{weight}', repo_type='model', token='{hf_token}', "
                f"local_dir='{weights_dir}')"
            )
        _run([env_python, "-c", code])

    # --- 4. prepared config ----------------------------------------------------------------
    # A copy of the official finetune config with the weight paths pointed at weights_dir.
    cfg_src = src_dir / FINETUNE_CONFIG
    cfg_dst = weights_dir / "omg_llava_7b_inference.py"
    text = cfg_src.read_text(encoding="utf-8")
    text = text.replace("./pretrained/omg_llava/internlm2-chat-7b",
                        str(weights_dir / "internlm2-chat-7b"))
    text = text.replace("./pretrained/omg_llava/omg_llava_7b_finetune_8gpus.pth",
                        str(weights_dir / "omg_llava_7b_finetune_8gpus.pth"))
    text = text.replace("./pretrained/omg_llava/omg_seg_convl.pth",
                        str(weights_dir / "omg_seg_convl.pth"))
    text = text.replace("./pretrained/omg_llava/convnext_large_d_320_CocoPanopticOVDataset.pth",
                        str(weights_dir / "convnext_large_d_320_CocoPanopticOVDataset.pth"))
    cfg_dst.write_text(text, encoding="utf-8")
    print(f"prepared config: {cfg_dst}")

    print("SANITY CHECK")
    _run([env_python, "-c",
          "import torch, transformers, mmcv, mmdet, mmengine; "
          "print('torch', torch.__version__, 'transformers', transformers.__version__, "
          "'cuda', torch.cuda.is_available(), torch.cuda.device_count())"])
    print("DONE -- OMG-LLaVA stack ready")
    print(f"  env python:   {env_python}")
    print(f"  weights:      {weights_dir}")
    print(f"  config:       {cfg_dst}")
    print("  next:  python scripts/run_phase19_omg_llava_benchmark.py smoke ...")


if __name__ == "__main__":
    main()
