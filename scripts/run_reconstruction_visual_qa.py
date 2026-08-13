"""Phase 7.3.2: real hidden-region reconstruction (LaMa) visual QA -- committed, reusable
infrastructure, not a one-off diagnostic script.

Phase 4's `_compute_hole_mask` fix (docs/decisions/0010-multi-object-layer-decomposition.md's
"Revision") was confirmed correct on real data, but only with a *placeholder* fill -- no live
GPU worker was available at the time to run real LaMa inference. This script closes that gap:
it runs the real pipeline (real grounding, real validation, real segmentation, real LaMa
inpainting) against a real page and saves cropped debug images -- source, segmentation mask,
hole mask, raw LaMa fill, the fill isolated to just the hole region composited onto the source,
and composited frames from the actual render -- so reconstruction quality can be judged by
looking at it, not inferred from successful execution, tensor shapes, or loss values (see the
Phase 7 brief's explicit "do not infer visual quality from successful execution" instruction).

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** -- see ADR 0003 and
CLAUDE.md's standing policy.

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/run_reconstruction_visual_qa.py \\
        --page examples/sample_page_01.png --semantic-label character_hair \\
        --transform-kind translate --amplitude 0.03

If `--plan-source auto` (the default) and automatic `analyze_page` doesn't produce a usable
plan for `--semantic-label` (a real, disclosed possibility -- see ADR 0009's VLM cross-session
nondeterminism finding), pass `--plan-source fallback` to use a controlled single-object
AnimationPlan instead, isolating real LaMa reconstruction quality from analysis-stage
nondeterminism. Either path is disclosed explicitly in the output JSON's `plan_source` field --
never silently substituted.

Writes `outputs/experiments/reconstruction_visual_qa_<timestamp>.json` and cropped PNGs under
`outputs/debug/reconstruction_visual_qa_<timestamp>/` (both git-ignored generated artifacts,
per ADR 0002 -- must be pulled back for inspection, e.g. via the Jupyter Contents API, not
committed).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.analysis import analyze_page
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.pipeline.orchestrator import build_default_clients, run_pipeline
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import (
    AnimationPlan,
    BBox,
    Easing,
    LoopSpec,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PanelPlan,
    PivotSpec,
    SourceImage,
    TransformKind,
    Vector2,
)


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
            meta["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        meta["torch_version"] = None
    return meta


def _fallback_plan(
    page: Path, semantic_label: str, transform_kind: TransformKind, amplitude: float
) -> AnimationPlan:
    w, h = Image.open(page).size
    if transform_kind in (TransformKind.TRANSLATE, TransformKind.SHEAR):
        motion = MotionSpec(
            transform_kind=transform_kind,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=amplitude,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        )
    elif transform_kind == TransformKind.MESH_WARP:
        motion = MotionSpec(
            transform_kind=transform_kind,
            direction=Vector2(x=0.6, y=0.8),
            amplitude=amplitude,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        )
    elif transform_kind == TransformKind.OPACITY:
        motion = MotionSpec(transform_kind=transform_kind, amplitude=amplitude, speed=1.0)
    else:  # ROTATE, SCALE
        motion = MotionSpec(
            transform_kind=transform_kind,
            amplitude=amplitude,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        )
    return AnimationPlan(
        source=SourceImage(path=str(page), width=w, height=h),
        panels=[PanelPlan(panel_id="panel_1", bbox=BBox(x=0, y=0, width=1, height=1))],
        objects=[
            ObjectPlan(
                object_id=f"obj_{semantic_label}",
                panel_id="panel_1",
                semantic_label=semantic_label,
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=motion,
            )
        ],
        loop=LoopSpec(duration_s=4.0, fps=24, seamless=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--semantic-label", required=True)
    parser.add_argument(
        "--transform-kind", default="translate",
        choices=[k.value for k in TransformKind],
    )
    parser.add_argument("--amplitude", type=float, default=0.03)
    parser.add_argument(
        "--plan-source", choices=["auto", "fallback"], default="auto",
        help="'auto' runs real analyze_page() first; 'fallback' always uses a controlled "
        "single-object plan, skipping the analysis-stage VLM call entirely.",
    )
    parser.add_argument("--env", default="kaggle", help="config profile (see configs/)")
    parser.add_argument("--crop-padding", type=int, default=60)
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--debug-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.page.exists():
        raise SystemExit(f"{args.page} not found")

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()
    vlm_client, grounding_client, segmentation_client, reconstruction_client = (
        build_default_clients(config)
    )
    transform_kind = TransformKind(args.transform_kind)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    debug_dir = args.debug_dir or Path(f"outputs/debug/reconstruction_visual_qa_{timestamp}")
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(f"outputs/videos/reconstruction_visual_qa_{timestamp}")

    plan: AnimationPlan | None = None
    plan_source = "fallback"
    if args.plan_source == "auto":
        try:
            auto_plan = analyze_page(args.page, vlm_client, config=config)
            matches = [
                o
                for o in auto_plan.objects
                if args.semantic_label.lower() in o.semantic_label.lower()
                and o.motion_type != MotionType.STATIC
            ]
            if matches:
                plan = auto_plan
                plan_source = "auto"
        except PipelineStageError:
            pass  # falls through to the controlled fallback below

    if plan is None:
        plan = _fallback_plan(args.page, args.semantic_label, transform_kind, args.amplitude)
        plan_source = "fallback"

    summary: dict[str, Any] = {
        "environment": _environment_metadata(device),
        "page": str(args.page),
        "semantic_label": args.semantic_label,
        "transform_kind": args.transform_kind,
        "plan_source": plan_source,
    }

    try:
        result = run_pipeline(
            args.page,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=out_dir,
            plan=plan,
        )
    except PipelineStageError as exc:
        summary["status"] = "FAILED"
        summary["failure"] = {
            "stage": exc.stage, "detail": exc.detail, "root_cause": exc.root_cause,
        }
        args.experiments_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.experiments_dir / f"reconstruction_visual_qa_{timestamp}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"FAILED at stage={exc.stage}: {exc.detail}")
        print(f"wrote {out_path}")
        raise SystemExit(1) from exc

    summary["status"] = "COMPLETED"
    summary["primary_object_id"] = result.primary_object.object_id
    summary["primary_semantic_label"] = result.primary_object.semantic_label
    summary["motion"] = str(result.primary_object.motion)
    summary["segmentation_iou"] = result.segmentation.iou_score
    summary["reconstruction_ran"] = result.reconstruction is not None

    image = np.asarray(Image.open(args.page).convert("RGB"))
    x0, y0, x1, y1 = result.segmentation.bbox.as_xyxy()
    pad = args.crop_padding
    cy0, cx0 = max(0, y0 - pad), max(0, x0 - pad)
    cy1, cx1 = min(image.shape[0], y1 + pad), min(image.shape[1], x1 + pad)
    summary["crop_bbox"] = [cx0, cy0, cx1, cy1]

    Image.fromarray(image[cy0:cy1, cx0:cx1]).save(debug_dir / "01_source_crop.png")
    mask = result.segmentation.mask
    Image.fromarray(mask[cy0:cy1, cx0:cx1]).save(debug_dir / "02_segmentation_mask_crop.png")

    if result.reconstruction is not None:
        recon = result.reconstruction
        summary["hole_coverage_fraction"] = float(
            (recon.hole_mask > 0).sum() / recon.hole_mask.size
        )
        Image.fromarray(recon.hole_mask[cy0:cy1, cx0:cx1]).save(
            debug_dir / "03_hole_mask_crop.png"
        )
        Image.fromarray(recon.filled_pixels[cy0:cy1, cx0:cx1]).save(
            debug_dir / "04_lama_raw_fill_crop.png"
        )
        overlay = image.copy()
        hole_bool = recon.hole_mask > 0
        overlay[hole_bool] = recon.filled_pixels[hole_bool]
        Image.fromarray(overlay[cy0:cy1, cx0:cx1]).save(
            debug_dir / "05_lama_fill_isolated_to_hole_crop.png"
        )
    else:
        summary["hole_coverage_fraction"] = None

    frame_paths = sorted((out_dir / "frames").glob("frame_*.png"))
    summary["frame_count"] = len(frame_paths)
    if len(frame_paths) >= 2:
        crop_targets = (
            (0, "06_composited_frame0_crop.png"),
            (len(frame_paths) // 2, "07_composited_mid_loop_crop.png"),
        )
        for idx, name in crop_targets:
            frame = np.asarray(Image.open(frame_paths[idx]).convert("RGB"))
            Image.fromarray(frame[cy0:cy1, cx0:cx1]).save(debug_dir / name)

    summary["render"] = {
        "output_path": str(result.render.output_path),
        "frame_count": result.render.frame_count,
        "resolution": result.render.resolution,
        "seamless_loop_verified": result.render.seamless_loop_verified,
    }
    summary["debug_dir"] = str(debug_dir)

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.experiments_dir / f"reconstruction_visual_qa_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("RECONSTRUCTION VISUAL QA COMPLETE")
    print(f"  plan_source: {plan_source}")
    print(
        f"  primary object: {result.primary_object.object_id} "
        f"({result.primary_object.semantic_label})"
    )
    print(f"  reconstruction_ran: {summary['reconstruction_ran']}")
    print(f"  hole_coverage_fraction: {summary['hole_coverage_fraction']}")
    print(f"  debug crops: {debug_dir}")
    print(f"  video: {result.render.output_path}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
