"""AnimateAnything end-to-end GPU run: original panel + SAM masks + Qwen descriptions -> video.

Runs the production panel pipeline (`run_page_panels`) with the GENERATIVE animation engine
(ADR 0024): grounding (DINO) -> object description (Qwen) -> segmentation (SAM) ->
AnimateAnything (merged SAM motion mask + prompt built from the accepted Qwen descriptions ->
frame sequence) -> render (H.264). No LaMa reconstruction and no deterministic CV animation
are used on this path -- AnimateAnything is the ONLY animation engine here.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

The AnimateAnything model needs its OWN Python environment on the worker because its pinned
stack (diffusers==0.24.0, transformers==4.36.2) conflicts with the project's `ml` extra
(transformers>=5.0). One-time worker setup:

    # 1. create the isolated interpreter (reuses the worker's torch; shadows the pinned stack)
    python -m venv --system-site-packages /kaggle/working/aa-venv
    /kaggle/working/aa-venv/bin/pip install \
        diffusers==0.24.0 transformers==4.36.2 \
        einops imageio opencv-python-headless safetensors pydantic accelerate==0.20.3

    # 2. download + extract the checkpoint (from the upstream project's release)
    cd /kaggle/working && aria2c -x 8 -s 8 -k 1M \
        https://cloudbook-public-production.oss-cn-shanghai.aliyuncs.com/ \
        animation/animate_anything_512_v1.02.tar \
        -d models/ && tar -xf models/animate_anything_512_v1.02.tar -C models/

    # 3. verify the isolated env before the run (client.load() also does this)
    #    (selfcheck needs a checkpoint; run it via the pipeline instead)

Usage:

    python scripts/run_animate_anything_e2e.py \
        --pages examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --aa-checkpoint /kaggle/working/models/animate_anything_512_v1.02 \
        --aa-python /kaggle/working/aa-venv/bin/python \
        --out outputs/experiments/animate_anything_<ts>.json

Writes one git-ignored experiment JSON per invocation: per-panel statuses, the animation
engine, the prompt(s) used, frame counts and render loop metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen3VLClient
from manga_animation.animation_anything.client import AnimateAnythingClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.orchestrator import DEFAULT_ANIMATION_LABELS
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.segmentation import Sam21Client

# Path to the isolated worker entrypoint, relative to this script (repo layout is canonical:
# scripts/ sits next to src/).
_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "manga_animation" / \
    "animation_anything" / "worker.py"


class CountingVLM:
    """Counts generate() calls of the real Qwen client (one call per panel it processes)."""

    def __init__(self, inner):
        self._inner = inner
        self.call_count = 0

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        return self._inner.generate(image, prompt)

    def unload(self) -> None:
        self._inner.unload()


def _panel_report(page_result) -> list[dict]:
    return [
        {
            "panel_id": p.panel_id,
            "status": p.status,
            "failure_stage": p.failure_stage,
            "failure_reason": p.failure_reason,
            "runtime_s": p.metrics.get("runtime_s"),
            "frame_count": p.metrics.get("frame_count"),
            "output_video": str(p.output_video) if p.output_video else None,
        }
        for p in page_result.panels
    ]


def _prompt_and_mask_report(page_result, page_dir: Path) -> list[dict]:
    """Per-panel (prompt, motion mask) provenance from the AnimateAnything work dirs."""
    report = []
    for panel in page_result.panels:
        workdir = page_dir / "animate_anything" / panel.panel_id
        spec_path = workdir / "spec.json"
        if not spec_path.exists():
            report.append({"panel_id": panel.panel_id, "spec": None})
            continue
        spec = json.loads(spec_path.read_text())
        report.append(
            {
                "panel_id": panel.panel_id,
                "prompt": spec.get("prompt"),
                "num_frames": spec.get("num_frames"),
                "fps": spec.get("fps"),
                "num_inference_steps": spec.get("num_inference_steps"),
                "guidance_scale": spec.get("guidance_scale"),
                "motion_strength": spec.get("motion_strength"),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True, help="Qwen3-VL-4B model dir on the worker")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--aa-checkpoint", required=True, help="extracted AnimateAnything dir")
    parser.add_argument("--aa-python", required=True, help="isolated worker interpreter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate semantic labels (default: the pipeline default list)",
    )
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": 1536})
    config.model_variants.update(
        {
            "animation": "animate-anything-512-v1.02",
            "grounding": "grounding-dino-swin-l",
            "segmentation": "sam2.1-hiera-base",
        }
    )
    setup_logging(False)
    import transformers  # noqa: F401 -- pre-load before worker threads spawn (Phase 22 lesson)
    from transformers import Sam2Model, Sam2Processor  # noqa: F401

    labels = list(args.labels or DEFAULT_ANIMATION_LABELS)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0
    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0
    # Device split (two cards): Qwen3-VL-4B fp16 (~8.5 GiB) runs as ONE instance on GPU 0,
    # and the AnimateAnything worker gets GPU 1 DEDICATED (its fp16 model is another ~2 GiB
    # with the CLIP text encoder). DINO/SAM are stage-owned and small; they share GPU 0.
    # Running Qwen on both cards would put a resident 8.5 GiB model on the AA card and OOM
    # the generative worker (real run) -- Qwen stays single-card here.
    qwen_device = "cuda:0" if n_gpus else "cpu"
    qwen: CountingVLM = CountingVLM(
        Qwen3VLClient(source=args.qwen, dtype="float16", max_new_tokens=4096,
                      device=qwen_device)
    )
    aux_device = "cuda:0" if n_gpus else "cpu"
    aa_device = "cuda:1" if n_gpus > 1 else ("cuda:0" if n_gpus == 1 else "cpu")
    dino = GroundingDinoClient(source=args.dino, device=aux_device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=aux_device, dtype="float32")
    aa = AnimateAnythingClient(
        source=args.aa_checkpoint,
        python_bin=args.aa_python,
        worker_script=_WORKER_SCRIPT,
        device=aa_device,
        num_frames=config.animation_num_frames,
        fps=config.animation_fps,
        num_inference_steps=config.animation_num_inference_steps,
        guidance_scale=config.animation_guidance_scale,
        motion_strength=config.animation_motion_strength,
        seed=config.seed,
    )

    report: dict = {
        "phase": "animate-anything-e2e",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
        "animation_engine": "animate-anything-512-v1.02",
        "vlm_instances": 1,
        "aa_device": aa_device,
        "vlm_calls": 0,
    }
    started = time.perf_counter()
    try:
        for page in args.pages:
            page_path = Path(page)
            page_result = run_page_panels(
                page_path,
                config,
                vlm_client=qwen,
                grounding_client=dino,
                segmentation_client=sam,
                reconstruction_client=None,
                animation_client=aa,
                out_dir=out_dir / "videos",
                labels=labels,
            )
            page_dir = out_dir / "videos" / page_path.stem
            page_entry = {
                "page": page,
                "panels": _panel_report(page_result),
                "animate_anything": _prompt_and_mask_report(page_result, page_dir),
            }
            report["pages"].append(page_entry)
            print(json.dumps(page_entry, indent=1), flush=True)
    finally:
        for client in (qwen, dino, sam, aa):
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown at the very end
                    pass

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    report["vlm_calls"] = qwen.call_count
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
