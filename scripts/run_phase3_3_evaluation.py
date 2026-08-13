"""Phase 3.3 real evaluation run: page-level vs. panel-aware analysis, over the real evaluation

dataset (`configs/phase3_3_eval_dataset.yaml`), plus a repeated-run VLM nondeterminism check.

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** -- see ADR 0003 and
CLAUDE.md's standing policy. Reuses `manga_animation.pipeline.orchestrator.run_pipeline` per
page per `analysis_mode`, mirroring `scripts/run_phase3_2_validation.py`'s structure -- this
script's job is running BOTH analysis modes over every sample and computing the Phase 3.3
metrics (`manga_animation.evaluation.metrics.compute_metrics`) for each, so they can be
directly compared.

No page gets an automatic, fabricated fallback plan (same policy as
`run_phase3_2_validation.py`) -- every result here is from fully automatic operation. Panel
detection itself (`detect_panels`, no VLM call) is run directly against every sample up front,
independent of whether the full pipeline run succeeds or fails, so `panel_count` is real
evidence for every sample, not only the ones that happened to complete.

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/run_phase3_3_evaluation.py
    uv run python scripts/run_phase3_3_evaluation.py --env kaggle
    uv run python scripts/run_phase3_3_evaluation.py --nondeterminism-runs 5

Phase 7.2.2: the nondeterminism check's `--nondeterminism-samples` now defaults to EVERY
sample_id in the loaded dataset (previously a hardcoded 2-sample subset,
`sample_page_01`/`sample_page_02` -- the samples the real Phase 3.3/3.3.1/3.3.2 nondeterminism
finding was based on). Pass it explicitly for a smaller/cheaper subset:
    uv run python scripts/run_phase3_3_evaluation.py --nondeterminism-samples sample_page_01

Writes `outputs/experiments/phase3_3_evaluation_<timestamp>.json` (per-page, per-mode outcomes;
both modes' `EvaluationReport`s; nondeterminism summaries) and, for every page/mode that reaches
rendering, the actual video under `outputs/videos/phase3_3/<mode>/<page-stem>/` (git-ignored
generated artifacts, per ADR 0002 -- never committed).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation import (
    EvaluationReport,
    PageRunOutcome,
    classify_outcome,
    compute_metrics,
    load_eval_dataset,
)
from manga_animation.evaluation.harness import (
    environment_metadata,
    panel_detection_evidence,
    run_nondeterminism_check,
    run_one_sample,
)
from manga_animation.pipeline.orchestrator import build_default_clients

# Phase 7.2.2: previously a hardcoded 2-sample subset (sample_page_01/02) -- the real Phase
# 3.3/3.3.1/3.3.2 nondeterminism finding these two samples motivated. `main()` now defaults
# `--nondeterminism-samples` to every sample_id in the LOADED dataset (see the `None` sentinel
# below), so this constant only documents the historical/minimum floor and is used as a
# fallback if the dataset somehow loads empty -- it does not silently cap real coverage at 2
# samples going forward, and automatically includes any sample added later without editing
# this script.
DEFAULT_NONDETERMINISM_SAMPLE_IDS = ["sample_page_01", "sample_page_02"]


def _render_rates_in_place(
    reports: dict[str, EvaluationReport], serialized_reports: dict[str, dict[str, Any]]
) -> None:
    """Add a human-readable "rendered" string (e.g. "6/10 (60.0%)") to every `Rate`-shaped dict

    inside `serialized_reports` (the `dataclasses.asdict()` output of `reports`) -- `Rate`
    objects aren't JSON-serializable by default, and `asdict()` alone only keeps their raw
    `numerator`/`denominator` ints, not `Rate.__str__`'s formatting.

    Phase 7 audit fix: a prior version of this loop iterated `serialized_reports.values()`
    (discarding the mode key) and looked up `reports[mode]` using a `mode` variable left over
    from an unrelated, already-finished loop earlier in `main()` (which always held its LAST
    value, `"panel"`, by the time this ran) -- silently rendering EVERY mode's `Rate` strings
    from the panel report's own values, corrupting the page report's "rendered" strings in the
    saved JSON (the underlying `numerator`/`denominator` fields were unaffected, only this
    display string). Extracted into its own function, taking both dicts as explicit parameters
    instead of closing over an outer-scope loop variable, so this can't silently reoccur, and so
    it's independently unit-testable (see `tests/test_run_phase3_3_evaluation_script.py`).
    """
    for mode, mode_report in serialized_reports.items():
        for key, value in list(mode_report.items()):
            if isinstance(value, dict) and "numerator" in value and "denominator" in value:
                mode_report[key] = {**value, "rendered": str(reports[mode].__getattribute__(key))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default="kaggle", help="config profile (see configs/)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/videos/phase3_3"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument(
        "--nondeterminism-runs",
        type=int,
        default=3,
        help="repeated analyze_page() calls per nondeterminism sample",
    )
    parser.add_argument(
        "--nondeterminism-samples",
        nargs="*",
        default=None,
        help=(
            "sample_ids to repeat-run for the nondeterminism check. Defaults to EVERY "
            "sample_id in the loaded dataset (Phase 7.2.2 -- 'the full currently configured "
            "evaluation dataset, where practical'), not a fixed subset -- pass this "
            "explicitly to run a smaller/cheaper subset instead."
        ),
    )
    args = parser.parse_args()

    samples = load_eval_dataset()
    if args.nondeterminism_samples is None:
        args.nondeterminism_samples = [s.sample_id for s in samples] or (
            DEFAULT_NONDETERMINISM_SAMPLE_IDS
        )
    missing = [s for s in samples if not Path(s.image_path).exists()]
    if missing:
        fetchable = [s.image_path for s in missing if s.fetch_script]
        unfetchable = [s.image_path for s in missing if not s.fetch_script]
        detail = []
        if fetchable:
            detail.append(f"fetch first: {fetchable} (see fetch_script per sample)")
        if unfetchable:
            detail.append(
                f"no fetch script exists for: {unfetchable} -- obtain these directly from the "
                "project owner, they cannot be regenerated by any script"
            )
        raise SystemExit("missing evaluation sample image(s) -- " + "; ".join(detail))

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()
    clients = build_default_clients(config)
    vlm_client = clients[0]

    panel_evidence = {s.sample_id: panel_detection_evidence(Path(s.image_path)) for s in samples}
    samples_by_id = {s.sample_id: s for s in samples}

    outcomes: dict[str, list[PageRunOutcome]] = {"page": [], "panel": []}
    for mode in ("page", "panel"):
        for sample in samples:
            panel_count, panel_sources = panel_evidence[sample.sample_id]
            outcome = run_one_sample(
                sample,
                mode,
                config,
                clients,
                args.out_dir / mode / Path(sample.image_path).stem,
                panel_count,
                panel_sources,
            )
            outcomes[mode].append(outcome)
            status = classify_outcome(outcome, samples_by_id.get(sample.sample_id))
            print(
                f"[{mode}] {sample.sample_id}: {status} -- {outcome.status} "
                f"({outcome.failing_stage or 'ok'})"
            )

    reports = {mode: compute_metrics(outcomes[mode], samples_by_id) for mode in ("page", "panel")}

    nondeterminism_summaries = [
        run_nondeterminism_check(samples_by_id[sid], config, vlm_client, args.nondeterminism_runs)
        for sid in args.nondeterminism_samples
        if sid in samples_by_id
    ]

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "environment": environment_metadata(device),
        "config_env": args.env,
        "model_variants": config.model_variants,
        "outcomes": {
            mode: [json.loads(o.model_dump_json()) for o in outcomes[mode]] for mode in outcomes
        },
        "reports": {mode: asdict(reports[mode]) for mode in reports},
        "nondeterminism": [asdict(s) for s in nondeterminism_summaries],
    }
    _render_rates_in_place(reports, summary["reports"])

    out_path = args.experiments_dir / f"phase3_3_evaluation_{timestamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\nPHASE 3.3 EVALUATION RUN COMPLETE")
    for mode in ("page", "panel"):
        print(f"\n--- {mode}-level report ---")
        r = reports[mode]
        print(f"  usable_target_rate: {r.usable_target_rate}")
        print(f"  static_rate: {r.static_rate}")
        print(f"  grounding_success_rate: {r.grounding_success_rate}")
        print(f"  validation_acceptance_rate: {r.validation_acceptance_rate}")
        print(f"  validation_rejection_rate: {r.validation_rejection_rate}")
        print(f"  end_to_end_completion_rate: {r.end_to_end_completion_rate}")
        print(f"  semantic_false_positive_rate: {r.semantic_false_positive_rate}")
        print(f"  semantic_false_negative_rate: {r.semantic_false_negative_rate}")
        print(
            f"  unresolved_ground_truth_count: {r.unresolved_ground_truth_count}/"
            f"{r.sample_count} (excluded from the two rates above)"
        )
        # Reuses semantic_false_negative_rate/semantic_false_positive_rate's own denominators --
        # no parallel metric system -- to make the 3-way split Pre-Phase-3.4 requires visible:
        # verified positive controls / verified negative controls / unresolved samples.
        positive_controls = r.semantic_false_negative_rate.denominator
        negative_controls = r.semantic_false_positive_rate.denominator
        resolved_total = positive_controls + negative_controls
        print(
            "  ground-truth split: "
            f"positive_controls={positive_controls} negative_controls={negative_controls} "
            f"unresolved={r.unresolved_ground_truth_count} "
            f"(sum={resolved_total + r.unresolved_ground_truth_count} of "
            f"sample_count={r.sample_count})"
        )
        print(
            f"  regression_violations: {r.regression_violation_count}/"
            f"{r.regression_samples_checked}"
        )
        if r.panel_detection_multi_panel_rate is not None:
            print(f"  panel_detection_multi_panel_rate: {r.panel_detection_multi_panel_rate}")
        print(f"  secondary_object_render_rate: {r.secondary_object_render_rate}")
        print(f"  micro_object_render_rate: {r.micro_object_render_rate}")
        sb = r.status_breakdown
        print(
            "  status_breakdown: "
            f"PASS={sb.pass_count} PASS_WITH_FALLBACK={sb.pass_with_fallback_count} "
            f"REJECTED={sb.rejected_count} ERROR={sb.error_count} (total={sb.total})"
        )
    print("\n--- nondeterminism ---")
    for s in nondeterminism_summaries:
        print(
            f"  {s.sample_id}: outcome_stable={s.outcome_stable} "
            f"target_category_stable={s.target_category_stable} "
            f"outcomes={s.distinct_outcomes} labels={s.distinct_primary_labels}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
