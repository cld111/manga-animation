"""AnimateAnything PHASE 2 ONLY: resume from Qwen checkpoints and animate + render.

The two-phase generative run splits Qwen and AnimateAnything so they never share a GPU.
This script runs ONLY the second phase: it expects the Qwen phase to have already written
`grounding.json`/`descriptions.json` checkpoints for the pages (via
`run_animate_anything_e2e.py`, which runs both phases, or a prior
`run_page_panels(..., stop_after_description=True)`), and it resumes from those checkpoints
-- DINO and Qwen are NOT loaded. One AnimateAnything worker per GPU animates each ACCEPTED
candidate's DINO bbox crop (crop + prompt -> frames) and renders one H.264 MP4 per object.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

Usage:

    python scripts/run_animate_anything_phase2.py \
        --pages examples/realworld/wind_breaker_sprint.png \
        --aa-checkpoint /kaggle/working/models/animate_anything_512_v1.02 \
        --aa-python /kaggle/working/aa-venv/bin/python \
        --out outputs/experiments/animate_anything_phase2.json

Requires the Qwen phase checkpoints to already exist at `--out`'s parent
`videos/<page_stem>/grounding.json` and `descriptions.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.animation_anything.client import AnimateAnythingClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.pipeline.panels import run_page_panels

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "manga_animation" / \
    "animation_anything" / "worker.py"


def _panel_report(page_result) -> list[dict]:
    return [
        {
            "panel_id": p.panel_id,
            "status": p.status,
            "failure_stage": p.failure_stage,
            "failure_reason": p.failure_reason,
            "runtime_s": p.metrics.get("runtime_s"),
            "object_count": len(p.output_videos),
            "output_videos": [str(v) for v in p.output_videos],
        }
        for p in page_result.panels
    ]


def _prompt_report(page_dir: Path) -> list[dict]:
    report: list[dict] = []
    aa_dir = page_dir / "animate_anything"
    if not aa_dir.exists():
        return report
    for panel_dir in sorted(aa_dir.iterdir()):
        for object_dir in sorted(panel_dir.iterdir()):
            spec_path = object_dir / "spec.json"
            if not spec_path.exists():
                continue
            spec = json.loads(spec_path.read_text())
            report.append(
                {
                    "panel_id": panel_dir.name,
                    "object_id": object_dir.name,
                    "prompt": spec.get("prompt"),
                    "num_frames": spec.get("num_frames"),
                    "fps": spec.get("fps"),
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--aa-checkpoint", required=True, help="extracted AnimateAnything dir")
    parser.add_argument("--aa-python", required=True, help="isolated worker interpreter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": 1536})
    setup_logging(False)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0
    devices = [f"cuda:{i}" for i in range(max(1, n_gpus))]

    aa_pool: list[AnimateAnythingClient] = [
        AnimateAnythingClient(
            source=args.aa_checkpoint,
            python_bin=args.aa_python,
            worker_script=_WORKER_SCRIPT,
            device=d,
            num_frames=config.animation_num_frames,
            fps=config.animation_fps,
            num_inference_steps=config.animation_num_inference_steps,
            guidance_scale=config.animation_guidance_scale,
            motion_strength=config.animation_motion_strength,
            seed=config.seed,
        )
        for d in devices
    ]

    report: dict = {
        "phase": "animate-anything-phase2-only",
        "architecture": "dino-bbox-crop-no-sam",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
        "aa_instances": len(aa_pool),
        "aa_devices": devices,
    }
    started = time.perf_counter()
    try:
        for page in args.pages:
            page_path = Path(page)
            # vlm_client/grounding_client are required args but must NOT load on resume
            # (their checkpoints skip them); pass throwaway objects so resume works.
            page_result = run_page_panels(
                page_path,
                config,
                vlm_client=_NoopVLM(),
                grounding_client=_NoopGrounding(),
                segmentation_client=None,
                reconstruction_client=None,
                animation_clients=aa_pool,
                out_dir=out_dir / "videos",
            )
            page_dir = out_dir / "videos" / page_path.stem
            page_entry = {
                "page": page,
                "panels": _panel_report(page_result),
                "animate_anything": _prompt_report(page_dir),
            }
            report["pages"].append(page_entry)
            print(json.dumps(page_entry, indent=1), flush=True)
    finally:
        for client in aa_pool:
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown
                    pass

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


class _NoopVLM:
    """Phase 2 never reaches Qwen (descriptions resume from disk), but run_page_panels
    requires a vlm_client argument."""

    def generate(self, image, prompt: str) -> str:
        raise AssertionError("phase 2 must not call Qwen (descriptions are resumed)")

    def unload(self) -> None:
        pass


class _NoopGrounding:
    """Phase 2 never reaches DINO (grounding resumes from disk)."""

    model_id = "noop-grounding"

    def detect(self, image, text_prompt: str) -> list:
        raise AssertionError("phase 2 must not call DINO (grounding is resumed)")

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


if __name__ == "__main__":
    main()
