"""Wan2.2 TI2V-5B end-to-end GPU run: original panel + Qwen descriptions -> video.

Runs the production panel pipeline (`run_page_panels`) with the GENERATIVE animation engine
(ADR 0024) in TWO PHASES so the two heavy models never share a GPU:

  Phase 1 (Qwen phase): ONE Qwen3-VL-4B instance PER GPU processes the whole dataset --
    grounding (DINO) -> object description (Qwen pool) -> segmentation (SAM) -- and persists
    its checkpoints, then Qwen is released. `run_page_panels(stop_after_segmentation=True)`.
  Phase 2 (Wan2.2 phase): the restored checkpoints skip DINO/Qwen/SAM entirely, and ONE
    Wan2.2 worker per GPU animates each accepted panel from (original panel image, prompt
    built from the accepted Qwen descriptions) and renders H.264.
    `run_page_panels(animation_clients=wan2_pool)`.

No LaMa reconstruction and no deterministic CV animation are used -- Wan2.2-TI2V-5B is the
ONLY animation engine here.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

The Wan2.2 model needs its OWN Python environment on the worker because it requires
diffusers main branch (not the PyPI release) and specific torch/transformers versions.

Usage:

    python scripts/run_wan2_e2e.py \
        --pages examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --wan2-checkpoint /kaggle/working/models/Wan2.2-TI2V-5B \
        --wan2-python /kaggle/working/wan2-venv/bin/python \
        --out outputs/experiments/wan2_<ts>.json

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
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.orchestrator import DEFAULT_ANIMATION_LABELS
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.segmentation import Sam21Client
from manga_animation.wan2.client import Wan2Client

# Path to the isolated worker entrypoint, relative to this script (repo layout is canonical:
# scripts/ sits next to src/).
_WORKER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "manga_animation" / "wan2" / "worker.py"
)


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


def _prompt_report(page_result, page_dir: Path) -> list[dict]:
    """Per-panel (prompt, motion mask) provenance from the Wan2.2 work dirs."""
    report = []
    for panel in page_result.panels:
        workdir = page_dir / "wan2" / panel.panel_id
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
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True, help="Qwen3-VL-4B model dir on the worker")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--wan2-checkpoint", required=True, help="Wan2.2-TI2V-5B model dir")
    parser.add_argument("--wan2-python", required=True, help="isolated worker interpreter")
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
            "animation": "wan2.2-ti2v-5b",
            "grounding": "grounding-dino-swin-l",
            "segmentation": "sam2.1-hiera-base",
        }
    )
    setup_logging(False)
    import transformers  # noqa: F401 -- pre-load before worker threads spawn
    from transformers import Sam2Model, Sam2Processor  # noqa: F401

    labels = list(args.labels or DEFAULT_ANIMATION_LABELS)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0
    devices = [f"cuda:{i}" for i in range(max(1, n_gpus))]

    # Two-phase Wan2.2 run (ADR 0024):
    #   Phase 1 (Qwen phase): ONE Qwen3-VL-4B instance PER GPU processes the whole dataset
    #     (grounding -> object description -> segmentation) and persists its checkpoints,
    #     then Qwen is released.
    #   Phase 2 (Wan2.2 phase): the restored checkpoints skip DINO/Qwen/SAM entirely, and
    #     ONE Wan2.2 worker per GPU animates the accepted panels.
    qwen_pool: list[CountingVLM] = [
        CountingVLM(
            Qwen3VLClient(source=args.qwen, dtype="float16", max_new_tokens=4096, device=d)
        )
        for d in devices
    ]
    # DINO/SAM are stage-owned and small; they live on the first card during the Qwen phase.
    aux_device = devices[0]
    dino = GroundingDinoClient(source=args.dino, device=aux_device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=aux_device, dtype="float32")
    wan2_pool: list[Wan2Client] = [
        Wan2Client(
            source=args.wan2_checkpoint,
            python_bin=args.wan2_python,
            worker_script=_WORKER_SCRIPT,
            device=d,
            num_frames=config.animation_num_frames,
            fps=config.animation_fps,
            num_inference_steps=config.animation_num_inference_steps,
            guidance_scale=config.animation_guidance_scale,
            seed=config.seed,
        )
        for d in devices
    ]

    report: dict = {
        "phase": "wan2-e2e-two-phase",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
        "animation_engine": "wan2.2-ti2v-5b",
        "vlm_instances": len(qwen_pool),
        "vlm_devices": devices,
        "wan2_instances": len(wan2_pool),
        "wan2_devices": devices,
        "vlm_calls": 0,
    }
    started = time.perf_counter()
    try:
        # Phase 1: Qwen pool -> grounding/description/segmentation checkpoints (no animation).
        for page in args.pages:
            page_path = Path(page)
            run_page_panels(
                page_path,
                config,
                vlm_client=qwen_pool,
                grounding_client=dino,
                segmentation_client=sam,
                reconstruction_client=None,
                animation_clients=None,
                out_dir=out_dir / "videos",
                labels=labels,
                stop_after_segmentation=True,
            )
        print("phase 1 (Qwen -> checkpoints) done", flush=True)

        # Phase 2: Wan2.2 pool -> animate + render, resuming the persisted stages.
        for page in args.pages:
            page_path = Path(page)
            page_result = run_page_panels(
                page_path,
                config,
                vlm_client=qwen_pool,
                grounding_client=dino,
                segmentation_client=sam,
                reconstruction_client=None,
                animation_clients=wan2_pool,
                out_dir=out_dir / "videos",
                labels=labels,
            )
            page_dir = out_dir / "videos" / page_path.stem
            page_entry = {
                "page": page,
                "panels": _panel_report(page_result),
                "wan2": _prompt_report(page_result, page_dir),
            }
            report["pages"].append(page_entry)
            print(json.dumps(page_entry, indent=1), flush=True)
    finally:
        for client in (*qwen_pool, dino, sam, *wan2_pool):
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown at the very end
                    pass

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    report["vlm_calls"] = sum(c.call_count for c in qwen_pool)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
