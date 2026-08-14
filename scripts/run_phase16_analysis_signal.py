"""Phase 16 cheap signal: what does the analysis stage produce for drawn effects?

Loads Qwen once, runs the real panel-aware analysis on each page, and prints EVERY
ObjectPlan's semantic_label / motion_type / transform_kind (the direct Phase 16 signal:
do effect labels now get effect-specific motion specs?). No grounding/SAM/animation --
analysis-only, so it is the cheapest possible GPU check of the plan_builder hypothesis.

Run on the GPU worker (ADR 0003). Usage:
    python scripts/run_phase16_analysis_signal.py --pages P1.png P2.png \
        --qwen /kaggle/working/models/qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient, analyze_page_panels
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, nargs="+", default=[])
    parser.add_argument("--qwen", type=str, required=True)
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    vlm = Qwen25VLClient(source=args.qwen, dtype=config.dtype)
    try:
        for page in args.pages:
            page = page.resolve()
            plan = analyze_page_panels(page, vlm, config=config)
            print(f"\n=== {page.name} === panels={len(plan.panels)}")
            for obj in sorted(plan.objects, key=lambda o: o.object_id):
                kind = obj.motion.transform_kind.value if obj.motion else "-"
                print(
                    f"  {obj.semantic_label:24s} {obj.motion_type.value:10s} "
                    f"transform_kind={kind:14s} confidence={obj.confidence:.2f}"
                )
    finally:
        vlm.unload()


if __name__ == "__main__":
    main()
