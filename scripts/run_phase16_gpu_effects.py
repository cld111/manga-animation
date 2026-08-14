"""Phase 16 short GPU evaluation: Drawn Effect Track on real pages.

Answers one concrete question -- "on a real page with drawn effects, does the pipeline now
produce effect-specific animation (RADIAL_EXPAND / effect-aware MotionSpecs from
_MOTION_HEURISTICS) instead of the pre-Phase-16 generic translate, and does it render?" --
with real GPU evidence, not mocks.

This is deliberately a SHORT run per the project's adaptive evaluation policy: a few
effect-heavy pages, not a full regression. It records for every panel:
  - the semantic_label / motion_type / transform_kind of every ObjectPlan the analysis
    stage produced (the direct Phase 16 signal: are effects labeled and given effect
    motion specs?),
  - the panel status (PASS / STATIC / REJECTED / ERROR) and output video path,
  - stage release logs (unchanged lifecycle must still hold on this branch).

Run on the Kaggle/Jupyter GPU worker, never locally (ADR 0003):

    python scripts/run_phase16_gpu_effects.py \
        --pages examples/eval_weapon_effects.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase16_gpu_effects_<ts>.json

Writes one git-ignored experiment JSON per invocation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from manga_animation.analysis import Qwen25VLClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.reconstruction import LamaClient
from manga_animation.segmentation import Sam21Client


def _capture_stage_release_logs() -> tuple[io.StringIO, logging.Handler]:
    import manga_animation.core.logging as core_logging

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    core_logging.get_logger("manga_animation.pipeline").addHandler(handler)
    return buffer, handler


def _release_logs_from_buffer(buffer: io.StringIO) -> list[str]:
    lines = buffer.getvalue().splitlines()
    buffer.truncate(0)
    buffer.seek(0)
    return [line for line in lines if "released" in line]


def _capture_pipeline_logs() -> tuple[io.StringIO, logging.Handler]:
    import manga_animation.core.logging as core_logging

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    core_logging.get_logger("manga_animation.pipeline").addHandler(handler)
    return buffer, handler


def _logs_from_buffer(buffer: io.StringIO) -> list[str]:
    lines = buffer.getvalue().splitlines()
    buffer.truncate(0)
    buffer.seek(0)
    return lines


def _run_page(page: Path, config, vlm, dino, sam, lama, out_dir: Path) -> dict[str, object]:
    buffer, handler = _capture_pipeline_logs()
    try:
        result = run_page_panels(
            page,
            config,
            vlm_client=vlm,
            grounding_client=dino,
            segmentation_client=sam,
            reconstruction_client=lama,
            out_dir=out_dir,
        )
        pipeline_logs = _logs_from_buffer(buffer)
    finally:
        import manga_animation.core.logging as core_logging

        core_logging.get_logger("manga_animation.pipeline").removeHandler(handler)

    # Phase 16 signal: every analysis decision + the PRIMARY selection line carry the
    # semantic_label -> transform_kind mapping the new _MOTION_HEURISTICS should produce.
    decision_lines = [
        line for line in pipeline_logs
        if "transform_kind=" in line or "analysis selected PRIMARY" in line
    ]
    return {
        "page": str(page),
        "out_dir": str(out_dir),
        "panels": [
            {
                "panel_id": p.panel_id,
                "status": p.status,
                "output_video": p.output_video,
                "scene_crop": str(p.scene_crop_path) if p.scene_crop_path else None,
            }
            for p in result.panels
        ],
        "phase16_signal_lines": decision_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, nargs="+", default=[])
    parser.add_argument("--qwen", type=str, required=True)
    parser.add_argument("--dino", type=str, required=True)
    parser.add_argument("--sam", type=str, required=True)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()

    vlm = Qwen25VLClient(source=args.qwen, dtype=config.dtype)
    dino = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=device, dtype="float32")
    lama = LamaClient(device=device)

    import torch

    gpus = [
        {
            "device": i,
            "name": torch.cuda.get_device_name(i),
            "total_mb": round(torch.cuda.get_device_properties(i).total_memory / 2**20),
        }
        for i in range(torch.cuda.device_count())
    ]

    buffer, handler = _capture_stage_release_logs()
    release_logs: list[str] = []
    page_runs: list[dict[str, object]] = []
    try:
        for page in args.pages:
            page = page.resolve()
            out_dir = Path("outputs/videos/phase16_evidence") / page.stem
            run = _run_page(page, config, vlm, dino, sam, lama, out_dir)
            page_runs.append(run)
            release_logs.extend(_release_logs_from_buffer(buffer))
            panels_data = cast(list[dict[str, object]], run["panels"])
            statuses = [p["status"] for p in panels_data]
            print(
                f"[{page.name}] statuses={statuses}",
                flush=True,
            )
    finally:
        handler.flush()
        import manga_animation.core.logging as core_logging

        core_logging.get_logger("manga_animation.pipeline").removeHandler(handler)

    out_path = args.out or Path(
        f"outputs/experiments/phase16_gpu_effects_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "environment": environment_metadata(device),
        "config_env": args.env,
        "gpus": gpus,
        "pages": [str(p.resolve()) for p in args.pages],
        "page_runs": page_runs,
        "stage_release_logs": release_logs,
        "phase16_signal": (
            "effect labels should now map to effect-specific transform kinds "
            "(radial_expand / mesh_warp / translate-down) instead of the pre-Phase-16 "
            "generic translate"
        ),
    }
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
