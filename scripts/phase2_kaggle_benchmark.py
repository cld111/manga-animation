"""Phase 2 candidate benchmark runner for the remote Kaggle/Jupyter GPU worker.

Covers the four model-benchmarking stages from `configs/benchmark_candidates.yaml`:
`vlm`, `grounding`, `segmentation`, `inpainting` (= hidden-region reconstruction, owned by
`cv-agent` — see `.claude/agents/cv-agent.md`). Uses the existing model-agnostic harness
(`manga_animation.benchmarking`) from Phase 2's first commit; this script is the missing
piece that first grounding-stage benchmark run did NOT leave behind: a committed,
reproducible adapter implementation, rather than ad hoc notebook code that only the numeric
results (docs/phase2-benchmark-results.md) survived from.

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** — see ADR 0003 and
ADR 0004's "standing project policy" (ask the user for the current server URL; never
guess/reuse a stale one, per CLAUDE.md). Kaggle GPU images typically already have
`torch`/`transformers`/`diffusers` preinstalled; if not: `pip install torch transformers
accelerate diffusers simple-lama-inpainting`. Deliberately NOT added to `pyproject.toml`'s
`ml` extra — see ADR 0004's "Consequences": model-loading dependencies are only added once a
candidate is actually *selected* per stage, not while still comparing candidates.

Every adapter below implements the `ModelAdapter` protocol from
`manga_animation.benchmarking.runner` (load/infer/unload) and is written against the
documented/expected `transformers`/`diffusers` API for each candidate in
`configs/benchmark_candidates.yaml`. Candidates released very recently relative to this
script's authoring (Qwen3-VL, SAM 3) are marked `# VERIFY:` — their exact class names should
be confirmed against the installed library version on first real run, per this project's
existing convention of treating desk research as provisional (see ADR 0004).

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/phase2_kaggle_benchmark.py --stage grounding
    uv run python scripts/phase2_kaggle_benchmark.py --stage vlm --candidates qwen2.5-vl-7b-instruct
    uv run python scripts/phase2_kaggle_benchmark.py --stage segmentation --dtype float16
    uv run python scripts/phase2_kaggle_benchmark.py --stage inpainting

Writes `outputs/experiments/phase2_<stage>_<timestamp>.json` (raw `BenchmarkResult`s +
environment metadata) and prints a markdown comparison table — paste that into
`docs/phase2-benchmark-results.md` alongside the run's environment section, matching the
existing grounding-stage write-up's format.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from manga_animation.benchmarking.registry import load_candidates
from manga_animation.benchmarking.report import render_markdown
from manga_animation.benchmarking.runner import ModelAdapter, run_sweep
from manga_animation.benchmarking.schemas import BenchmarkResult

# Manga-relevant prompt classes, matching the first grounding-stage run
# (docs/phase2-benchmark-results.md) so results stay comparable across runs.
GROUNDING_PROMPT = "hair. face. hand. speech bubble. eye."
VLM_PROMPT = (
    "Describe what is happening in this manga panel: which objects or characters show "
    "implied motion (speed lines, drawn deformation, wind-blown hair/cloth), and which "
    "parts of the page are static background."
)

# A synthetic center-of-page box prompt, standing in for real grounding-stage output —
# no grounding candidate has been selected yet (see docs/decisions/0005-phase2-model-selection.md),
# so segmentation/inpainting are timed against a fixed placeholder region, not a real object.
_PLACEHOLDER_BOX_NORM = (0.30, 0.04, 0.70, 0.20)  # (x0, y0, x1, y1), normalized


def _box_px(image: Image.Image) -> tuple[int, int, int, int]:
    w, h = image.size
    x0, y0, x1, y1 = _PLACEHOLDER_BOX_NORM
    return (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))


# --- adapters: grounding -----------------------------------------------------


class GroundingDinoAdapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def infer(self, sample: Image.Image) -> Any:
        import torch

        inputs = self.processor(images=sample, text=GROUNDING_PROMPT, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # NOTE: transformers 5.0.0 renamed `box_threshold` -> `threshold` on this call
        # (confirmed via inspect.signature on a real Kaggle T4 run, 2026-08-12) — the
        # ADR 0004-era `box_threshold` kwarg raises TypeError on this version.
        return self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.25,
            text_threshold=0.2,
            target_sizes=[sample.size[::-1]],
        )

    def unload(self) -> None:
        import torch

        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OwlV2Adapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(self.source)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def infer(self, sample: Image.Image) -> Any:
        import torch

        prompts = [p.strip() for p in GROUNDING_PROMPT.rstrip(".").split(".") if p.strip()]
        inputs = self.processor(images=sample, text=[prompts], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.post_process_grounded_object_detection(
            outputs, threshold=0.1, target_sizes=torch.tensor([sample.size[::-1]])
        )

    def unload(self) -> None:
        import torch

        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Sam3GroundingAdapter:
    """VERIFY: SAM 3's concept-grounding API/class name against the installed `transformers`

    version — released Nov 2025 (per ADR 0004), after this script's authoring. Written to
    the documented text/concept-prompted detect+segment interface; adjust the class import
    if the installed library names it differently.
    """

    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor  # VERIFY: exact SAM 3 class names

        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model = AutoModel.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def infer(self, sample: Image.Image) -> Any:
        import torch

        inputs = self.processor(images=sample, text="hair, face, hand", return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            return self.model(**inputs)

    def unload(self) -> None:
        import torch

        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- adapters: vlm ------------------------------------------------------------


class Qwen25VLAdapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # CONFIRMED on a real Kaggle 2xT4 session (2026-08-12): `.to("cuda")` onto a
        # single T4 OOMs ("CUDA out of memory... 14.15 GiB is allocated by PyTorch" on
        # a 14.56 GiB card), contradicting ADR 0004's desk-research "fits comfortably on
        # a T4/L4" claim at float16. `device_map="auto"` (sharding across both T4s) DOES
        # fix it — confirmed loading (~101s from a warm HF cache) and, more importantly,
        # confirmed generating real output (see docs/phase2-benchmark-results.md's third
        # pass): visual encoder + early layers land on GPU0 (~7.9GB), later decoder layers
        # + lm_head on GPU1 (~6.5GB). This requires >1 GPU — a single-T4/L4 profile still
        # needs quantization or a smaller model, not yet tested.
        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype), device_map="auto"
        )
        self.model.eval()

    def infer(self, sample: Image.Image) -> Any:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Place inputs on the sharded model's first device, not a fixed `self.device` —
        # with device_map="auto" the model spans multiple devices, and `model.device`
        # resolves to wherever its first parameter (the embedding/visual stem) lives.
        inputs = self.processor(text=[text], images=[sample], return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            return self.model.generate(**inputs, max_new_tokens=200)

    def unload(self) -> None:
        import torch

        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3VLAdapter(Qwen25VLAdapter):
    """VERIFY: Qwen3-VL's exact model class/repo id — released after this script's

    authoring; ADR 0004 flags the exact small-variant repo id as still TBD. Reuses
    Qwen2.5-VL's chat-template inference shape as a starting point since Qwen's VL line has
    kept a consistent processor/generate interface across versions; the model class import
    below is the one detail most likely to need updating.
    """

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor  # VERIFY: class name

        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()


class InternVL3Adapter:
    """VERIFY: InternVL3 typically ships with custom modeling code

    (`trust_remote_code=True`) rather than a first-class `transformers` class — confirm the
    exact chat/generate call against the model card at integration time.
    """

    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.source, trust_remote_code=True, use_fast=False
        )
        self.model = (
            AutoModel.from_pretrained(
                self.source, torch_dtype=getattr(torch, self.dtype), trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

    def infer(self, sample: Image.Image) -> Any:
        # InternVL's `.chat(...)` convenience method takes a pixel-value tensor built by its
        # own preprocessing helper (model-card-specific) — placeholder call, verify at
        # integration time against the actual InternVL3 image-preprocessing utility.
        return self.model.chat(self.tokenizer, sample, VLM_PROMPT, {"max_new_tokens": 200})

    def unload(self) -> None:
        import torch

        del self.model, self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- adapters: segmentation ---------------------------------------------------


class Sam21Adapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor  # VERIFY: SAM2.1 vs. base SAM2 class

        self.processor = Sam2Processor.from_pretrained(self.source)
        self.model = Sam2Model.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def infer(self, sample: Image.Image) -> Any:
        import torch

        box = [list(_box_px(sample))]
        inputs = self.processor(sample, input_boxes=[box], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CONFIRMED signature on a real Kaggle run (2026-08-12, transformers 5.0.0):
        # `post_process_masks(masks, original_sizes, ...)` — no `reshaped_input_sizes`
        # argument exists on this version (an earlier guess here caused a real KeyError).
        return self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])

    def unload(self) -> None:
        import torch

        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Sam3SegAdapter(Sam3GroundingAdapter):
    """VERIFY: same caveat as `Sam3GroundingAdapter` — SAM 3 may collapse grounding and

    segmentation behind one call; this adapter reuses the concept-prompted entry point to
    time the segmentation-quality side specifically (see ADR 0004's architectural note).
    """


# --- adapters: inpainting (hidden-region reconstruction) ----------------------


def _placeholder_hole_mask(image: Image.Image) -> Image.Image:
    """A synthetic hole mask standing in for one a real motion-reveal would produce.

    Placed just outside the placeholder grounding box above, roughly where an object's
    motion might uncover background — no real segmentation/animation stage exists yet to
    derive this from (see the reconstruction ownership section in .claude/agents/cv-agent.md).
    """
    from PIL import ImageDraw

    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    x0, y0, x1, y1 = _box_px(image)
    ImageDraw.Draw(mask).rectangle([x0, y1, x1, min(y1 + (y1 - y0) // 3, h)], fill=255)
    return mask


class LamaAdapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        # `simple-lama-inpainting` (PyPI) wraps the manga-image-translator-proven LaMa
        # checkpoint behind a plain (image, mask) -> image call — see the candidate's notes
        # in configs/benchmark_candidates.yaml. VERIFY the exact checkpoint source at
        # integration time (ADR 0004: "exact checkpoint source TBD").
        #
        # ENVIRONMENT NOTE (real, 2026-08-12): importing `simple_lama_inpainting` AFTER
        # `cv2`/`numpy` were already imported elsewhere in the same process raised
        # `RuntimeError: empty_like method already has a different docstring` (a numpy/cv2
        # ABI conflict from installing this package mid-session). Import it first, or in a
        # fresh process, to avoid this — not a problem with the package itself.
        from simple_lama_inpainting import SimpleLama

        self.model = SimpleLama(device=self.device)

    def infer(self, sample: tuple[Image.Image, Image.Image]) -> Any:
        # CONFIRMED real inference on a real Kaggle T4 (2026-08-12): works, ~2.9s/image,
        # ~1.2GB peak VRAM. IMPORTANT finding: the raw output is NOT pixel-aligned with
        # the input (a 1778x1000 input came back 1784x1000 — the model pads to its
        # internal stride) — naively substituting the full raw output would silently
        # violate "Original Image Is the Source of Truth" (see docs/architecture.md).
        # This confirms compositing must resize/crop the output back to the source
        # resolution and blend ONLY the masked hole (via `cv-agent`'s compositing step,
        # see the reconstruction ownership section in .claude/agents/cv-agent.md) — never
        # use this adapter's return value as a full-frame replacement.
        image, mask = sample
        return self.model(image, mask)

    def unload(self) -> None:
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AotInpaintingAdapter:
    """VERIFY: `mayocream/aot-inpainting` is a community SafeTensors conversion with no

    standard `transformers`/`diffusers` pipeline class (ADR 0004: "license TBD... verify
    upstream"). Loading approach (raw state dict into the AOT-GAN generator architecture) is
    the most likely detail to need correcting against the actual checkpoint at integration
    time.
    """

    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(self.source, filename="model.safetensors")
        from safetensors.torch import load_file

        self.state_dict = load_file(weights_path)
        self.device_t = torch.device(self.device)
        # VERIFY: instantiate the actual AOT-GAN generator class here and load state_dict —
        # left unimplemented pending the model-card-specific architecture definition.
        raise NotImplementedError(
            "AOT inpainting generator architecture not yet wired up — verify against "
            "mayocream/aot-inpainting's model card before running this candidate."
        )

    def infer(self, sample: tuple[Image.Image, Image.Image]) -> Any:  # pragma: no cover
        raise NotImplementedError

    def unload(self) -> None:  # pragma: no cover
        pass


class SdxlInpaintingAdapter:
    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype

    def load(self) -> None:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)

    def infer(self, sample: tuple[Image.Image, Image.Image]) -> Any:
        image, mask = sample
        return self.pipe(
            prompt="",  # unconditional fill; manga line art has no natural text prompt
            image=image,
            mask_image=mask,
            num_inference_steps=25,
        ).images[0]

    def unload(self) -> None:
        import torch

        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- registry: stage -> candidate_id -> adapter factory ------------------------

AdapterFactory = Callable[[str, str, str], ModelAdapter]

ADAPTERS: dict[str, dict[str, AdapterFactory]] = {
    "grounding": {
        "grounding-dino-swin-l": GroundingDinoAdapter,
        "owlv2-vit-l14": OwlV2Adapter,
        "sam3-concept-grounding": Sam3GroundingAdapter,
    },
    "vlm": {
        "qwen2.5-vl-7b-instruct": Qwen25VLAdapter,
        "qwen3-vl-small": Qwen3VLAdapter,
        "internvl3-8b": InternVL3Adapter,
    },
    "segmentation": {
        "sam2.1-hiera-base": Sam21Adapter,
        "sam3": Sam3SegAdapter,
    },
    "inpainting": {
        "lama-large": LamaAdapter,
        "aot-inpainting-manga": AotInpaintingAdapter,
        "sdxl-inpainting": SdxlInpaintingAdapter,
    },
}


# --- sample building per stage -------------------------------------------------


def build_samples(stage: str, images: list[Image.Image]) -> list[Any]:
    if stage in ("grounding", "vlm", "segmentation"):
        return list(images)
    if stage == "inpainting":
        return [(img, _placeholder_hole_mask(img)) for img in images]
    raise ValueError(f"unknown stage {stage!r}")


# --- environment metadata + CLI -------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_metadata(device: str, dtype: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "dtype": dtype,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["gpu_total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (
                1024**2
            )
    except ImportError:
        meta["torch_version"] = None
    try:
        import transformers

        meta["transformers_version"] = transformers.__version__
    except ImportError:
        meta["transformers_version"] = None
    return meta


def load_sample_images(images_dir: Path, count: int) -> list[Image.Image]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not paths:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fetch_sample_pages import fetch  # sibling script, MangaDex Full Color pages only

        fetch(count, images_dir)
        paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no sample images found under {images_dir} and none could be fetched")
    return [Image.open(p).convert("RGB") for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=sorted(ADAPTERS))
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated candidate ids (default: all with a registered adapter)",
    )
    parser.add_argument("--images-dir", type=Path, default=Path("examples"))
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()

    all_candidates = load_candidates()[args.stage]
    wanted_ids = set(args.candidates.split(",")) if args.candidates else None
    candidates = [
        c
        for c in all_candidates
        if c.id in ADAPTERS[args.stage] and (wanted_ids is None or c.id in wanted_ids)
    ]
    if not candidates:
        raise SystemExit(f"no runnable candidates for stage={args.stage!r} (check --candidates)")

    images = load_sample_images(args.images_dir, args.count)
    samples = build_samples(args.stage, images)

    adapters: dict[str, ModelAdapter] = {
        c.id: ADAPTERS[args.stage][c.id](c.source, args.device, args.dtype) for c in candidates
    }

    results: list[BenchmarkResult] = run_sweep(
        candidates, adapters, samples, device=args.device, dtype=args.dtype
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"phase2_{args.stage}_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "environment": environment_metadata(args.device, args.dtype),
                "num_samples": len(samples),
                "results": [r.model_dump(mode="json") for r in results],
            },
            indent=2,
        )
    )

    print(f"wrote {out_path}\n")
    print(render_markdown(results))


if __name__ == "__main__":
    main()
