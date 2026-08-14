"""Phase 13 targeted real GPU panel validation: run the new panel-first production entry point
(`run_page_panels`) on a single real example page, on a Kaggle/Jupyter GPU worker, and record
the per-panel outcome.

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md's standing
policy; see ADR 0003). The example pages are git-ignored generated artifacts (ADR 0002), so on a
fresh clone the page must be fetched first, e.g.:

    uv run python scripts/fetch_phase9_realworld_pages.py

then run:

    uv run python scripts/run_phase13_panel_smoke.py
        --page examples/realworld/villainess_ending_scuffle.png

The panel runner (`run_page_panels`, `src/manga_animation/pipeline/panels.py`) detects panels,
builds bounded scene crops, and runs the real analysis -> grounding -> validation ->
segmentation -> mask_semantics -> animation -> reconstruction -> compositing -> rendering stages
independently per crop. Every panel gets a stable unit and either an output video (PASS), an
all-STATIC read (STATIC), a safe rejection (REJECTED), or an ERROR, and the page manifest is
written after each panel.

Writes `outputs/experiments/phase13_panel_smoke_<timestamp>.json` with the environment, per-panel
statuses, crop/video paths, and measured runtime metrics, plus the rendered videos under
`outputs/videos/phase13_evidence/` (git-ignored artifacts; download locally for visual
inspection).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.pipeline.orchestrator import build_default_clients
from manga_animation.pipeline.panels import run_page_panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page",
        type=Path,
        default=Path("examples/realworld/villainess_ending_scuffle.png"),
    )
    parser.add_argument("--env", default="kaggle", help="config profile (see configs/)")
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()
    clients = build_default_clients(config)

    image_path = args.page.resolve()
    if not image_path.exists():
        raise SystemExit(f"example page not found: {image_path}")

    out_dir = Path("outputs/videos/phase13_evidence")
    experiments_dir = Path("outputs/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = experiments_dir / f"phase13_panel_smoke_{timestamp}.json"

    print(f"=== running run_page_panels on {image_path} ===", flush=True)
    result = run_page_panels(
        image_path,
        config,
        vlm_client=clients[0],
        grounding_client=clients[1],
        segmentation_client=clients[2],
        reconstruction_client=clients[3],
        out_dir=out_dir,
    )

    record = {
        "page_id": result.page_id,
        "source_image": str(result.source_image),
        "manifest_path": str(result.manifest_path),
        "detected_panel_count": len(result.panels),
        "panels": [panel.as_manifest_dict() for panel in result.panels],
        "performance": json.loads(result.manifest_path.read_text())["performance"],
    }
    out_path.write_text(
        json.dumps(
            {
                "environment": environment_metadata(device),
                "config_env": args.env,
                "model_variants": config.model_variants,
                "git_commit_note": "run against the commit checked out on this worker",
                "result": record,
            },
            indent=2,
            default=str,
        )
    )

    print(f"\nmanifest: {result.manifest_path}")
    for panel in result.panels:
        print(
            f"[{panel.panel_id}] status={panel.status} "
            f"crop={panel.scene_crop_bbox.as_xyxy()} "
            f"video={panel.output_video} "
            f"runtime_s={panel.metrics.get('runtime_s')}"
            + (
                f" failing_stage={panel.failure_stage} reason={panel.failure_reason}"
                if panel.status in ("REJECTED", "ERROR")
                else ""
            )
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
