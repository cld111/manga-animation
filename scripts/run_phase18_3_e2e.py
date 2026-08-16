"""Phase 18.3 end-to-end GPU validation: SAM masks + Qwen object descriptions -> animation.

Runs the production page entry point (`run_page_panels`) on REAL manga pages with the Phase
18.3 per-candidate VLM object-description stage ENABLED (default): for every grounded and
segmented object, Qwen2.5-VL sees the full panel image plus the accepted bbox as pixel
coordinates, produces a structured animation description, and the deterministically-mapped
MotionSpec drives the animation stage (with the SAM mask from segmentation). This is the
task brief's required integration: `masks from SAM + description from Qwen -> animation`,
inside the existing pipeline, with no isolated demo path.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

Usage (models downloaded to local dirs on the worker, pages fetched):

    python scripts/run_phase18_3_e2e.py \
        --pages examples/realworld/villainess_ending_scuffle.png \
                examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase18_3_e2e_<ts>.json

Writes one git-ignored experiment JSON per invocation with per-panel statuses, the
object-description verdicts that actually drove each render, and the render loop metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient
from manga_animation.core.config import load_config
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.segmentation import Sam21Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate semantic labels to ground (default: the pipeline default list)",
    )
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": args.resolution})
    config.model_variants.update(
        {
            "vlm": "qwen2.5-vl-7b-instruct",
            "grounding": "grounding-dino-swin-l",
            "segmentation": "sam2.1-hiera-base",
        }
    )
    assert config.enable_object_description_validation, "the Phase 18.3 stage must be enabled"

    vlm_client = Qwen25VLClient(source=args.qwen, dtype="float16")
    device = config.resolve_device()
    grounding_client = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    segmentation_client = Sam21Client(source=args.sam, device=device, dtype="float32")
    # Phase 14 lifecycle contract: exactly one model family resident at a time. The
    # reconstruction client is constructed here but only used (and therefore only loaded)
    # inside run_page_panels' own ModelStage.
    from manga_animation.reconstruction import LamaClient

    reconstruction_client = LamaClient(device=device, model_id="lama-large")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    report: dict = {
        "phase": "18.3-e2e",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
    }
    try:
        for page in args.pages:
            page_result = run_page_panels(
                Path(page),
                config,
                vlm_client=vlm_client,
                grounding_client=grounding_client,
                segmentation_client=segmentation_client,
                reconstruction_client=reconstruction_client,
                out_dir=out_dir / "videos",
                labels=args.labels,
            )
            page_entry = {
                "page": page,
                "panels": [
                    {
                        "panel_id": p.panel_id,
                        "status": p.status,
                        "failure_stage": p.failure_stage,
                        "failure_reason": p.failure_reason,
                        "output_video": str(p.output_video) if p.output_video else None,
                        "metrics": p.metrics,
                    }
                    for p in page_result.panels
                ],
            }
            report["pages"].append(page_entry)
            print(json.dumps(page_entry, indent=1))
    finally:
        for client in (vlm_client, grounding_client, segmentation_client, reconstruction_client):
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown at the very end
                    pass

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
