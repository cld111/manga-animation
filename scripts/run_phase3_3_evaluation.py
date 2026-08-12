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
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from manga_animation.analysis.panels import detect_panels
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation import (
    EvalSample,
    NondeterminismSummary,
    ObjectAttemptOutcome,
    PageRunOutcome,
    RepeatedRunRecord,
    ValidationAttemptOutcome,
    compute_metrics,
    load_eval_dataset,
    summarize_repeated_runs,
)
from manga_animation.pipeline.orchestrator import build_default_clients, run_pipeline
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import MotionType

# Phase 7.2.2: previously a hardcoded 2-sample subset (sample_page_01/02) -- the real Phase
# 3.3/3.3.1/3.3.2 nondeterminism finding these two samples motivated. `main()` now defaults
# `--nondeterminism-samples` to every sample_id in the LOADED dataset (see the `None` sentinel
# below), so this constant only documents the historical/minimum floor and is used as a
# fallback if the dataset somehow loads empty -- it does not silently cap real coverage at 2
# samples going forward, and automatically includes any sample added later without editing
# this script.
DEFAULT_NONDETERMINISM_SAMPLE_IDS = ["sample_page_01", "sample_page_02"]


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
    return meta


def _panel_detection_evidence(image_path: Path) -> tuple[int, list[str]]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    panels = detect_panels(image)
    return len(panels), [p.source for p in panels]


def _run_one(
    sample: EvalSample,
    mode: Literal["page", "panel"],
    config: Any,
    clients: tuple[Any, Any, Any, Any],
    out_dir: Path,
    panel_count: int,
    panel_sources: list[str],
) -> PageRunOutcome:
    vlm_client, grounding_client, segmentation_client, reconstruction_client = clients
    image_path = Path(sample.image_path)
    try:
        result = run_pipeline(
            image_path,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=out_dir,
            analysis_mode=mode,
        )
    except PipelineStageError as exc:
        # No PipelineRunResult exists on this path (run_pipeline raised before returning), so
        # there is no visibility into which SECONDARY/MICRO objects might have been attempted
        # -- object_outcomes stays empty here, same as any other schema_version=2 page that
        # genuinely had none. schema_version is still bumped: this producer supports the
        # field, it just has nothing to report for a failed run.
        return PageRunOutcome(
            sample_id=sample.sample_id,
            analysis_mode=mode,
            status="failed",
            failing_stage=exc.stage,
            failure_detail=exc.detail,
            panel_count=panel_count,
            panel_sources=panel_sources,
            schema_version=2,
        )
    except Exception as exc:  # noqa: BLE001 -- one sample's unexpected crash must not stop
        # the rest of the evaluation run; recorded distinctly from a classified
        # PipelineStageError (failing_stage="unexpected"), same pattern as
        # scripts/run_phase3_2_validation.py.
        return PageRunOutcome(
            sample_id=sample.sample_id,
            analysis_mode=mode,
            status="failed",
            failing_stage="unexpected",
            failure_detail=f"{type(exc).__name__}: {exc}",
            panel_count=panel_count,
            panel_sources=panel_sources,
            schema_version=2,
        )

    object_outcomes = [
        ObjectAttemptOutcome(
            object_id=obj.object_plan.object_id,
            semantic_label=obj.object_plan.semantic_label,
            motion_type=obj.object_plan.motion_type.value,
            status="rendered",
            validation_attempts=[
                ValidationAttemptOutcome(
                    candidate_rank=v.candidate_rank,
                    accepted=v.accepted,
                    grounding_score=v.grounding_score,
                    reason=v.reason,
                )
                for v in obj.validation_attempts
            ],
        )
        for obj in result.secondary_objects
    ] + [
        ObjectAttemptOutcome(
            object_id=dropped.object_plan.object_id,
            semantic_label=dropped.object_plan.semantic_label,
            motion_type=dropped.object_plan.motion_type.value,
            status="dropped",
        )
        for dropped in result.dropped_objects
    ]

    return PageRunOutcome(
        sample_id=sample.sample_id,
        analysis_mode=mode,
        status="completed",
        panel_count=panel_count,
        panel_sources=panel_sources,
        primary_semantic_label=result.primary_object.semantic_label,
        primary_motion_type=result.primary_object.motion_type.value,
        validation_attempts=[
            ValidationAttemptOutcome(
                candidate_rank=v.candidate_rank,
                accepted=v.accepted,
                grounding_score=v.grounding_score,
                reason=v.reason,
            )
            for v in result.validation_attempts
        ],
        object_outcomes=object_outcomes,
        schema_version=2,
    )


def _run_nondeterminism_check(
    sample: EvalSample, config: Any, vlm_client: Any, run_count: int
) -> NondeterminismSummary:
    from manga_animation.analysis.plan_builder import analyze_page
    from manga_animation.pipeline.types import PipelineStageError as _PSE

    records: list[RepeatedRunRecord] = []
    for i in range(run_count):
        try:
            plan = analyze_page(Path(sample.image_path), vlm_client, config=config)
        except _PSE:
            records.append(
                RepeatedRunRecord(
                    sample_id=sample.sample_id, run_index=i, outcome="static_or_unusable"
                )
            )
            continue
        primaries = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
        primary = primaries[0] if primaries else None
        records.append(
            RepeatedRunRecord(
                sample_id=sample.sample_id,
                run_index=i,
                outcome="usable",
                primary_semantic_label=primary.semantic_label if primary else None,
                primary_motion_type=primary.motion_type.value if primary else None,
                object_count=len(plan.objects),
            )
        )
    return summarize_repeated_runs(records)


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

    panel_evidence = {s.sample_id: _panel_detection_evidence(Path(s.image_path)) for s in samples}

    outcomes: dict[str, list[PageRunOutcome]] = {"page": [], "panel": []}
    for mode in ("page", "panel"):
        for sample in samples:
            panel_count, panel_sources = panel_evidence[sample.sample_id]
            outcome = _run_one(
                sample,
                mode,
                config,
                clients,
                args.out_dir / mode / Path(sample.image_path).stem,
                panel_count,
                panel_sources,
            )
            outcomes[mode].append(outcome)
            print(
                f"[{mode}] {sample.sample_id}: {outcome.status} ({outcome.failing_stage or 'ok'})"
            )

    samples_by_id = {s.sample_id: s for s in samples}
    reports = {mode: compute_metrics(outcomes[mode], samples_by_id) for mode in ("page", "panel")}

    nondeterminism_summaries = [
        _run_nondeterminism_check(samples_by_id[sid], config, vlm_client, args.nondeterminism_runs)
        for sid in args.nondeterminism_samples
        if sid in samples_by_id
    ]

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "environment": _environment_metadata(device),
        "config_env": args.env,
        "model_variants": config.model_variants,
        "outcomes": {
            mode: [json.loads(o.model_dump_json()) for o in outcomes[mode]] for mode in outcomes
        },
        "reports": {mode: asdict(reports[mode]) for mode in reports},
        "nondeterminism": [asdict(s) for s in nondeterminism_summaries],
    }
    # Rate objects inside `reports` aren't JSON-serializable by default -- render explicitly.
    for mode_report in summary["reports"].values():
        for key, value in list(mode_report.items()):
            if isinstance(value, dict) and "numerator" in value and "denominator" in value:
                mode_report[key] = {**value, "rendered": str(reports[mode].__getattribute__(key))}

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
