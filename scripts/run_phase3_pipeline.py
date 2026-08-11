"""Phase 3.1 real end-to-end run: one real manga page -> a playable, seamless-loop MP4.

Runs the complete pipeline (`manga_animation.pipeline.orchestrator.run_pipeline`) against
real models (Qwen2.5-VL-7B-Instruct, Grounding DINO Swin-L, SAM 2.1 Hiera Base, LaMa) on a
GPU. **Run this on the remote Kaggle/Jupyter GPU worker, never locally** — see ADR 0003 and
CLAUDE.md's standing policy (heavy model inference is remote-GPU-only). Kaggle GPU images
typically already have `torch`/`transformers` preinstalled; if not:
`pip install torch transformers accelerate simple-lama-inpainting` (or `uv sync --extra ml`).

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/run_phase3_pipeline.py
    uv run python scripts/run_phase3_pipeline.py --page examples/phase3_action_page.png --env kaggle

Writes `outputs/experiments/phase3_pipeline_<timestamp>.json` (environment metadata + full
`PipelineRunResult` summary — plan, grounding/segmentation/reconstruction findings, render
result) and the actual video/frames under `outputs/videos` / `outputs/frames` (all git-ignored
generated artifacts, per ADR 0002 — never committed, must be pulled back for inspection some
other way, e.g. downloading the file directly from this session).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.pipeline.orchestrator import build_default_clients, run_pipeline
from manga_animation.pipeline.types import PipelineStageError


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _environment_metadata(device: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            meta["gpu_count"] = gpu_count
            meta["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except ImportError:
        meta["torch_version"] = None
    try:
        import transformers

        meta["transformers_version"] = transformers.__version__
    except ImportError:
        meta["transformers_version"] = None
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--page", type=Path, default=Path("examples/phase3_action_page.png"))
    parser.add_argument("--env", default="kaggle", help="config profile (see configs/)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/videos/phase3"))
    parser.add_argument(
        "--experiments-dir", type=Path, default=Path("outputs/experiments"),
    )
    args = parser.parse_args()

    setup_logging(debug=False)

    if not args.page.exists():
        raise SystemExit(
            f"{args.page} not found — fetch it first: "
            "uv run python scripts/fetch_phase3_sample_page.py"
        )

    config = load_config(args.env)
    device = config.resolve_device()
    vlm_client, grounding_client, segmentation_client, reconstruction_client = (
        build_default_clients(config)
    )

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "environment": _environment_metadata(device),
        "config_env": args.env,
        "page": str(args.page),
        "model_variants": config.model_variants,
    }

    try:
        result = run_pipeline(
            args.page,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=args.out_dir,
        )
    except PipelineStageError as exc:
        summary["status"] = "FAILED"
        summary["failure"] = {
            "stage": exc.stage,
            "input_ref": exc.input_ref,
            "detail": exc.detail,
            "root_cause": exc.root_cause,
            "architectural": exc.architectural,
            "proposed_fix": exc.proposed_fix,
        }
        out_path = args.experiments_dir / f"phase3_pipeline_{timestamp}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"PIPELINE FAILED at stage={exc.stage}: {exc.detail}")
        print(f"wrote {out_path}")
        raise SystemExit(1) from exc

    summary["status"] = "COMPLETED"
    summary["plan"] = json.loads(result.plan.model_dump_json())
    summary["primary_object_id"] = result.primary_object.object_id
    summary["grounding"] = {
        "object_id": result.grounding.object_id,
        "bbox": result.grounding.bbox.as_xyxy(),
        "score": result.grounding.bbox.score,
        "model_id": result.grounding.model_id,
    }
    summary["segmentation"] = {
        "object_id": result.segmentation.object_id,
        "bbox": result.segmentation.bbox.as_xyxy(),
        "iou_score": result.segmentation.iou_score,
        "model_id": result.segmentation.model_id,
        "mask_coverage_fraction": float(
            (result.segmentation.mask > 0).sum() / result.segmentation.mask.size
        ),
    }
    summary["reconstruction"] = (
        None
        if result.reconstruction is None
        else {
            "object_id": result.reconstruction.object_id,
            "model_id": result.reconstruction.model_id,
            "hole_coverage_fraction": float(
                (result.reconstruction.hole_mask > 0).sum() / result.reconstruction.hole_mask.size
            ),
        }
    )
    summary["render"] = {
        "output_path": str(result.render.output_path),
        "frame_count": result.render.frame_count,
        "fps": result.render.fps,
        "resolution": result.render.resolution,
        "duration_s": result.render.duration_s,
        "codec": result.render.codec,
        "pixel_format": result.render.pixel_format,
        "seamless_loop_verified": result.render.seamless_loop_verified,
    }

    out_path = args.experiments_dir / f"phase3_pipeline_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("PIPELINE COMPLETED")
    print(
        f"  primary object: {result.primary_object.object_id} "
        f"({result.primary_object.semantic_label})"
    )
    print(f"  motion: {result.primary_object.motion}")
    print(f"  video: {result.render.output_path}")
    print(
        f"  frames: {result.render.frame_count} @ {result.render.fps}fps, "
        f"{result.render.duration_s}s"
    )
    print(f"  seamless_loop_verified: {result.render.seamless_loop_verified}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
