"""Reproducible evaluation metrics, computed from `PageRunOutcome` records -- see

`evaluation/schemas.py`. Every rate is a `Rate` (numerator/denominator, never a bare float) so
a percentage can never be reported without its sample count, per the Phase 3.3 brief: "Never
present a percentage without its denominator."
"""

from __future__ import annotations

from dataclasses import dataclass

from manga_animation.evaluation.dataset import EvalSample
from manga_animation.evaluation.schemas import PageRunOutcome

_STATIC_DETAIL_MARKER = "every object STATIC"
"""Substring shared by both `_rank_candidates`'s and `_rank_panel_candidates`'s all-STATIC

`PipelineStageError.detail` text (`analysis/plan_builder.py`) -- the only way to distinguish
"the VLM read every candidate as STATIC" from "the VLM's output was unparseable JSON" within a
single `stage="analysis"` failure, since both currently share that one `Stage` value. See those
functions' docstrings for the exact wording this matches against."""


@dataclass(frozen=True, slots=True)
class Rate:
    """A `numerator/denominator` rate that always carries its own sample count -- `str(rate)`

    renders e.g. "6/10 (60.0%)", never a bare percentage. `denominator=0` is a valid, common
    case (e.g. no validation attempts were made at all) and renders as "0/0 (n/a)" rather than
    raising or silently reporting 0%.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator < 0:
            raise ValueError(
                f"Rate cannot be negative: numerator={self.numerator} "
                f"denominator={self.denominator}"
            )
        if self.numerator > self.denominator:
            raise ValueError(
                f"Rate numerator ({self.numerator}) cannot exceed denominator ({self.denominator})"
            )

    @property
    def value(self) -> float | None:
        """`None` (not 0.0 or NaN) when `denominator == 0` -- there is no rate to report, not a

        zero one."""
        return self.numerator / self.denominator if self.denominator else None

    def __str__(self) -> str:
        if self.denominator == 0:
            return f"{self.numerator}/{self.denominator} (n/a)"
        assert self.value is not None
        return f"{self.numerator}/{self.denominator} ({self.value:.1%})"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Every Phase 3.3-required metric, for one `analysis_mode`'s set of outcomes."""

    analysis_mode: str
    sample_count: int
    usable_target_rate: Rate
    """Pages where analysis produced >=1 non-STATIC candidate (did not fail with

    stage="analysis") / pages attempted."""
    static_rate: Rate
    """Pages where analysis specifically failed because every candidate was STATIC (a genuine

    all-STATIC read, not an unparseable-output failure) / pages attempted."""
    grounding_success_rate: Rate
    """Pages that reached grounding (analysis succeeded) where grounding found >=1 candidate

    (did not fail with stage="grounding") / pages that reached grounding."""
    validation_acceptance_rate: Rate
    """Accepted validation attempts / all validation attempts, pooled across every page."""
    validation_rejection_rate: Rate
    """Rejected validation attempts / all validation attempts, pooled across every page."""
    fallback_rate: Rate
    """Pages that used a human-supplied controlled-fallback plan / pages attempted."""
    end_to_end_completion_rate: Rate
    """Pages that reached a completed run (a real rendered video) / pages attempted."""
    semantic_false_positive_rate: Rate
    """Among samples with ground-truth animation_possible="no", the fraction that still

    COMPLETED (the pipeline animated something where a human found no justified target)."""
    semantic_false_negative_rate: Rate
    """Among samples with ground-truth animation_possible="yes", the fraction that FAILED (the

    pipeline found nothing where a human found a real, justified target)."""
    regression_violation_count: int
    """How many samples with a `regression_reference` (a specific, named, known-bad outcome)

    actually reproduced it this run -- see `_check_regression`. Always reported alongside
    `regression_samples_checked` (never a bare count)."""
    regression_samples_checked: int
    panel_detection_multi_panel_rate: Rate | None
    """Among `analysis_mode="panel"` outcomes only: pages where real gutter-based detection

    found >=2 panels / all panel-mode pages attempted. `None` for `analysis_mode="page"`
    reports, where this metric does not apply."""


def _is_static_failure(outcome: PageRunOutcome) -> bool:
    return (
        outcome.status == "failed"
        and outcome.failing_stage == "analysis"
        and outcome.failure_detail is not None
        and _STATIC_DETAIL_MARKER in outcome.failure_detail
    )


def _reached_grounding(outcome: PageRunOutcome) -> bool:
    """Analysis produced a usable plan -- the run got at least as far as attempting grounding

    (whether or not grounding itself then succeeded)."""
    return not (outcome.status == "failed" and outcome.failing_stage == "analysis")


def _check_regression(outcome: PageRunOutcome, sample: EvalSample | None) -> bool:
    """True if this outcome reproduced the specific, named, known-bad case a sample's

    `regression_reference` describes -- currently: a validated ACCEPT whose grounded candidate
    the ground-truth notes identify as the known-wrong one. Conservative by construction: only
    samples that both HAVE a `regression_reference` and whose ground truth says
    `animation_possible != "no"` (a completion isn't inherently the violation) are checked, and
    only a `status == "completed"` outcome can violate it -- absence of evidence is not treated
    as a violation.
    """
    if sample is None or not sample.regression_reference:
        return False
    return outcome.status == "completed"


def compute_metrics(
    outcomes: list[PageRunOutcome], samples: dict[str, EvalSample]
) -> EvaluationReport:
    """`outcomes` must all share one `analysis_mode` (raises `ValueError` otherwise) -- compute

    a page-level and a panel-level `EvaluationReport` separately and compare them, per the
    Phase 3.3 brief's "page-level vs panel-level" acceptance criterion, rather than pooling
    both modes into one misleading aggregate.
    """
    if not outcomes:
        raise ValueError("compute_metrics requires at least one outcome")
    modes = {o.analysis_mode for o in outcomes}
    if len(modes) > 1:
        raise ValueError(
            f"compute_metrics requires all outcomes to share one analysis_mode, got {modes}"
        )
    analysis_mode = next(iter(modes))
    n = len(outcomes)

    usable_target = sum(1 for o in outcomes if _reached_grounding(o))
    static_count = sum(1 for o in outcomes if _is_static_failure(o))

    reached_grounding = [o for o in outcomes if _reached_grounding(o)]
    grounding_success = sum(
        1
        for o in reached_grounding
        if not (o.status == "failed" and o.failing_stage == "grounding")
    )

    all_attempts = [v for o in outcomes for v in o.validation_attempts]
    accepted_attempts = sum(1 for v in all_attempts if v.accepted)
    rejected_attempts = len(all_attempts) - accepted_attempts

    fallback_used = sum(1 for o in outcomes if o.used_fallback_plan)
    completed = sum(1 for o in outcomes if o.status == "completed")

    fp_num = fp_den = fn_num = fn_den = 0
    regression_checked = regression_violated = 0
    for outcome in outcomes:
        sample = samples.get(outcome.sample_id)
        if sample is not None and sample.regression_reference:
            regression_checked += 1
            if _check_regression(outcome, sample):
                regression_violated += 1

        if sample is None or sample.animation_possible == "uncertain":
            continue
        if sample.animation_possible == "no":
            fp_den += 1
            if outcome.status == "completed":
                fp_num += 1
        elif sample.animation_possible == "yes":
            fn_den += 1
            if outcome.status == "failed":
                fn_num += 1

    panel_rate: Rate | None = None
    if analysis_mode == "panel":
        multi_panel = sum(1 for o in outcomes if (o.panel_count or 0) >= 2)
        panel_rate = Rate(multi_panel, n)

    return EvaluationReport(
        analysis_mode=analysis_mode,
        sample_count=n,
        usable_target_rate=Rate(usable_target, n),
        static_rate=Rate(static_count, n),
        grounding_success_rate=Rate(grounding_success, len(reached_grounding)),
        validation_acceptance_rate=Rate(accepted_attempts, len(all_attempts)),
        validation_rejection_rate=Rate(rejected_attempts, len(all_attempts)),
        fallback_rate=Rate(fallback_used, n),
        end_to_end_completion_rate=Rate(completed, n),
        semantic_false_positive_rate=Rate(fp_num, fp_den),
        semantic_false_negative_rate=Rate(fn_num, fn_den),
        regression_violation_count=regression_violated,
        regression_samples_checked=regression_checked,
        panel_detection_multi_panel_rate=panel_rate,
    )
