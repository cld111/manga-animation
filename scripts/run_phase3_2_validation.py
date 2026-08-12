"""Phase 3.2 real end-to-end validation run: VLM targeting + grounding-target validation,

aggregated across the existing real Phase 2/3.1 sample pages.

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** — see ADR 0003 and
CLAUDE.md's standing policy (heavy model inference is remote-GPU-only). Reuses
`manga_animation.pipeline.orchestrator.run_pipeline` per page; this script's only job is
aggregating the Phase 3.2 acceptance-criteria metrics across multiple real pages in one pass,
which a single-page run (`scripts/run_phase3_pipeline.py`) can't report on its own:

- VLM usable-target rate: fraction of pages where the analysis stage produced at least one
  non-STATIC candidate (did not raise `PipelineStageError(stage="analysis")`).
- Grounding candidate acceptance rate / rejection rate: computed from every
  `PipelineRunResult.validation_attempts` record across all pages that reached grounding.
- Fallback rate: fraction of *attempted* pages that only produced a result via a
  human-supplied `--fallback-plan`, not automatic analysis.
- False-positive semantic selections: this script cannot judge "was the ACCEPTed candidate
  actually correct" automatically — every ACCEPTed `ValidationResult` is listed in the summary
  under `needs_visual_review` so a human (or `qa-agent`) can check the rendered crop, per the
  Phase 3.2 brief's "visual QA findings" requirement. A REJECT is not flagged for review; a
  rejection is itself a successful outcome per the brief's acceptance criterion.

No page gets an automatic, fabricated fallback plan — if a page's automatic run fails (all-
STATIC analysis, or every grounding candidate rejected), it is recorded FAILED with the
failing stage/detail, exactly as `run_pipeline` reports it. Re-run that one page with
`--fallback-plan` after a human has reviewed the failure and authored one, mirroring how
Phase 3.1's own real run actually happened (see docs/phase3-results.md).

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/run_phase3_2_validation.py
    uv run python scripts/run_phase3_2_validation.py --page examples/phase3_action_page.png
    uv run python scripts/run_phase3_2_validation.py \\
        --page examples/phase3_action_page.png \\
        --fallback-plan outputs/experiments/phase3_2_fallback_plan.json

Writes `outputs/experiments/phase3_2_validation_<timestamp>.json` (per-page results + the
aggregate rates above) and, for every page that reaches rendering, the actual video under
`outputs/videos/phase3_2/<page-stem>/` (git-ignored generated artifacts, per ADR 0002 — never
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

from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.pipeline.orchestrator import build_default_clients, run_pipeline
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import AnimationPlan

DEFAULT_PAGES = [
    Path("examples/sample_page_01.png"),
    Path("examples/sample_page_02.png"),
    Path("examples/phase3_action_page.png"),
]


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


def _run_one_page(
    page: Path,
    config: Any,
    clients: tuple[Any, Any, Any, Any],
    out_dir: Path,
    fallback_plan: AnimationPlan | None,
) -> dict[str, Any]:
    vlm_client, grounding_client, segmentation_client, reconstruction_client = clients
    record: dict[str, Any] = {"page": str(page), "used_fallback_plan": fallback_plan is not None}

    try:
        result = run_pipeline(
            page,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=out_dir,
            plan=fallback_plan,
        )
    except PipelineStageError as exc:
        record["status"] = "FAILED"
        record["failing_stage"] = exc.stage
        record["failure_detail"] = exc.detail
        record["validation_attempts"] = []
        return record
    except Exception as exc:  # noqa: BLE001 -- one page's unexpected crash must not stop
        # the rest of the aggregate run (or lose already-collected pages' data); recorded
        # distinctly from a PipelineStageError so it's clear this was NOT one of the
        # pipeline's own classified failure modes -- see `failing_stage: "unexpected"`.
        record["status"] = "FAILED"
        record["failing_stage"] = "unexpected"
        record["failure_detail"] = f"{type(exc).__name__}: {exc}"
        record["validation_attempts"] = []
        return record

    record["status"] = "COMPLETED"
    record["primary_object"] = {
        "object_id": result.primary_object.object_id,
        "semantic_label": result.primary_object.semantic_label,
        "transform_kind": (
            result.primary_object.motion.transform_kind.value
            if result.primary_object.motion
            else None
        ),
    }
    record["accepted_grounding"] = {
        "bbox": result.grounding.bbox.as_xyxy(),
        "score": result.grounding.bbox.score,
    }
    record["validation_attempts"] = [
        {
            "candidate_rank": v.candidate_rank,
            "accepted": v.accepted,
            "grounding_score": v.grounding_score,
            "bbox_area_fraction": v.bbox_area_fraction,
            "bbox_plausible": v.bbox_plausible,
            "semantic_match": v.semantic_match,
            "semantic_confidence": v.semantic_confidence,
            "reason": v.reason,
        }
        for v in result.validation_attempts
    ]
    record["render"] = {
        "output_path": str(result.render.output_path),
        "seamless_loop_verified": result.render.seamless_loop_verified,
    }
    return record


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(records)
    vlm_usable = sum(
        1 for r in records if r["status"] == "COMPLETED" or r.get("failing_stage") != "analysis"
    )
    all_attempts = [v for r in records for v in r.get("validation_attempts", [])]
    accepted_attempts = [v for v in all_attempts if v["accepted"]]
    rejected_attempts = [v for v in all_attempts if not v["accepted"]]
    fallback_used = sum(1 for r in records if r["used_fallback_plan"])
    completed = [r for r in records if r["status"] == "COMPLETED"]

    return {
        "pages_attempted": attempted,
        "vlm_usable_target_rate": vlm_usable / attempted if attempted else None,
        "grounding_candidates_tried": len(all_attempts),
        "grounding_candidate_acceptance_rate": (
            len(accepted_attempts) / len(all_attempts) if all_attempts else None
        ),
        "grounding_candidate_rejection_rate": (
            len(rejected_attempts) / len(all_attempts) if all_attempts else None
        ),
        "pages_completed": len(completed),
        "fallback_rate": fallback_used / attempted if attempted else None,
        "needs_visual_review": [
            {"page": r["page"], "accepted_grounding": r.get("accepted_grounding")}
            for r in completed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--page", type=Path, action="append", dest="pages", default=None)
    parser.add_argument("--env", default="kaggle", help="config profile (see configs/)")
    parser.add_argument(
        "--fallback-plan",
        type=Path,
        default=None,
        help="AnimationPlan JSON to use as the controlled fallback -- only valid with exactly "
        "one --page",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/videos/phase3_2"))
    parser.add_argument(
        "--experiments-dir", type=Path, default=Path("outputs/experiments"),
    )
    args = parser.parse_args()

    pages = args.pages or DEFAULT_PAGES
    if args.fallback_plan is not None and len(pages) != 1:
        raise SystemExit("--fallback-plan requires exactly one --page")

    missing = [p for p in pages if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing sample page(s): {missing} -- fetch them first: "
            "uv run python scripts/fetch_sample_pages.py / "
            "uv run python scripts/fetch_phase3_sample_page.py"
        )

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()
    clients = build_default_clients(config)

    fallback_plan = (
        AnimationPlan.from_json_file(args.fallback_plan) if args.fallback_plan else None
    )

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    records = [
        _run_one_page(page, config, clients, args.out_dir / page.stem, fallback_plan)
        for page in pages
    ]

    summary: dict[str, Any] = {
        "environment": _environment_metadata(device),
        "config_env": args.env,
        "model_variants": config.model_variants,
        "pages": records,
        "aggregate": _aggregate(records),
    }

    out_path = args.experiments_dir / f"phase3_2_validation_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("PHASE 3.2 VALIDATION RUN COMPLETE")
    for r in records:
        stage_note = f" (stage={r.get('failing_stage')})" if r["status"] == "FAILED" else ""
        print(f"  {r['page']}: {r['status']}{stage_note}")
    print(f"  aggregate: {json.dumps(summary['aggregate'], indent=2, default=str)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
