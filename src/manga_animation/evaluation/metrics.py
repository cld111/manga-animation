"""Reproducible evaluation metrics, computed from `PageRunOutcome` records -- see

`evaluation/schemas.py`. Every rate is a `Rate` (numerator/denominator, never a bare float) so
a percentage can never be reported without its sample count, per the Phase 3.3 brief: "Never
present a percentage without its denominator."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    """Among samples with ground-truth animation_possible="no" AND ground_truth_uncertain=False,

    the fraction that still COMPLETED (the pipeline animated something where a human found no
    justified target). Samples with ground_truth_uncertain=True are excluded from this
    denominator regardless of their animation_possible value -- see
    `unresolved_ground_truth_count` (Phase 3.4 baseline cleanup)."""
    semantic_false_negative_rate: Rate
    """Among samples with ground-truth animation_possible="yes" AND ground_truth_uncertain=False,

    the fraction that FAILED (the pipeline found nothing where a human found a real, justified
    target). Same exclusion as `semantic_false_positive_rate` above."""
    unresolved_ground_truth_count: int
    """How many of this report's `sample_count` outcomes belong to a sample whose

    `EvalSample.ground_truth_uncertain` is `True` (e.g. `sample_page_01`/`sample_page_02` as of
    Phase 3.3.2) -- these never contribute to `semantic_false_positive_rate`/
    `semantic_false_negative_rate`'s numerators or denominators. Reported explicitly so an
    unresolved sample's exclusion is visible, not an invisible side effect of the rate
    denominators being smaller than `sample_count` (Phase 3.4 baseline cleanup)."""
    regression_violation_count: int
    """How many samples with a `regression_reference` (a specific, named, known-bad outcome)

    actually reproduced it this run -- see `_check_regression`. Always reported alongside
    `regression_samples_checked` (never a bare count)."""
    regression_samples_checked: int
    panel_detection_multi_panel_rate: Rate | None
    """Among `analysis_mode="panel"` outcomes only: pages where real gutter-based detection

    found >=2 panels / all panel-mode pages attempted. `None` for `analysis_mode="page"`
    reports, where this metric does not apply."""
    secondary_object_render_rate: Rate
    """Phase 7.2.1 (closes the evaluation gap ADR 0010 explicitly deferred to Phase 7): among

    every `PageRunOutcome.object_outcomes` entry with `motion_type == "secondary"`, pooled
    across every outcome in this report, the fraction with `status == "rendered"` -- i.e. of
    every SECONDARY object the VLM proposed on an attempted page, how many actually made it
    into the final render (grounding+validation both succeeded), vs. how many were dropped per
    ADR 0010's non-fatal SECONDARY/MICRO failure policy. `denominator=0` (rendered as "0/0
    (n/a)", per `Rate`) whenever every outcome in this report has `schema_version < 2` (see
    `PageRunOutcome.schema_version`) or genuinely proposed no SECONDARY object at all -- these
    two cases are NOT distinguished by this rate alone."""
    micro_object_render_rate: Rate
    """Same as `secondary_object_render_rate`, for `motion_type == "micro"`."""
    status_breakdown: StatusBreakdown
    """Phase 8: every outcome in this report classified into the brief's required PASS /

    PASS_WITH_FALLBACK / REJECTED / ERROR vocabulary -- see `classify_outcome`. Counts sum to
    `sample_count` exactly (`StatusBreakdown.total`)."""


E2EStatus = Literal["PASS", "PASS_WITH_FALLBACK", "REJECTED", "ERROR"]


@dataclass(frozen=True, slots=True)
class StatusBreakdown:
    """How many outcomes in one `EvaluationReport` fell into each `E2EStatus` bucket -- the

    Phase 8 brief's explicit requirement ("must clearly distinguish PASS; PASS WITH FALLBACK;
    REJECTED; ERROR. Do not hide rejected cases."), built from per-outcome `classify_outcome`
    calls rather than a new parallel counting pass.
    """

    pass_count: int
    pass_with_fallback_count: int
    rejected_count: int
    error_count: int

    def __post_init__(self) -> None:
        if any(
            c < 0
            for c in (
                self.pass_count,
                self.pass_with_fallback_count,
                self.rejected_count,
                self.error_count,
            )
        ):
            raise ValueError(f"StatusBreakdown counts cannot be negative: {self}")

    @property
    def total(self) -> int:
        return (
            self.pass_count + self.pass_with_fallback_count + self.rejected_count + self.error_count
        )


def classify_outcome(outcome: PageRunOutcome, sample: EvalSample | None) -> E2EStatus:
    """The Phase 8 brief's required PASS/PASS_WITH_FALLBACK/REJECTED/ERROR vocabulary for one

    outcome, built entirely from signals this module already computes elsewhere
    (`_check_regression`, and the same semantic false-positive/false-negative logic
    `compute_metrics` uses) -- no new ground-truth interpretation invented for this function
    alone.

    - **ERROR**: real, structured evidence of a wrong or broken result -- this outcome
      reproduced a sample's own named `regression_reference` (`_check_regression`), OR it
      contradicts the sample's confident (`ground_truth_uncertain=False`) `animation_possible`
      ground truth (a semantic false positive: completed on a confident "no"; or false
      negative: failed on a confident "yes" -- UNLESS the sample's own
      `honest_failure_acceptable` explicitly allows an attributed failure here, see below), OR
      the run failed for a reason the harness could not attribute to any real pipeline stage at
      all (`failing_stage is None` -- see `evaluation.schemas.FailingStage`'s own "unexpected"
      value for the *attributed* case, which is REJECTED, not ERROR, below).
    - **REJECTED**: the run failed at a specific, identifiable reason (an all-STATIC read, an
      empty grounding result, every validation candidate rejected, or even an unexpected
      exception the harness still recorded as `failing_stage="unexpected"`) and nothing above
      flags it as contradicting ground truth -- a correct, honest negative result per this
      project's "Static Is a Valid Result" principle (docs/architecture.md), not a defect.
    - **PASS_WITH_FALLBACK**: the run completed using a human-authored controlled-fallback plan
      (`used_fallback_plan=True`) rather than fully automatic operation.
    - **PASS**: the run completed fully automatically, with nothing above flagging a problem.

    Order matters: regression/ground-truth violations are checked before `status`, so a
    "completed" run that actually animated the wrong object is never miscategorized as PASS
    just because a video was produced.

    `sample.honest_failure_acceptable` (Phase 8.3): a small, evidenced carve-out of the
    confident-"yes"-but-failed branch. Two real samples (`phase3_action_page`,
    `eval_weapon_effects`) have confident ground truth (something real and animatable IS
    present) but their own `acceptable_outcome` prose has always explicitly allowed an honest
    grounding/validation failure too, because the target is an effect-heavy motion cue rather
    than one concrete, easily-prompted object -- a real, previously-undocumented mismatch
    between that prose and this function's structured-fields-only logic, which classified both
    as ERROR on real Kaggle GPU output (`docs/phase8-results.md` section 6.2) despite neither
    being a defect per the sample's own written acceptance criterion. Only an *attributed*
    failure (`failing_stage is not None`) counts as "honest" here -- a genuinely unattributed
    failure still falls through to the unconditional ERROR check below, same as any other
    sample.
    """
    if _check_regression(outcome, sample):
        return "ERROR"
    if sample is not None and not sample.ground_truth_uncertain:
        if sample.animation_possible == "no" and outcome.status == "completed":
            return "ERROR"
        honest_failure = (
            sample.honest_failure_acceptable
            and outcome.status == "failed"
            and outcome.failing_stage is not None
        )
        if sample.animation_possible == "yes" and outcome.status == "failed" and not honest_failure:
            return "ERROR"
    if outcome.status == "failed":
        return "ERROR" if outcome.failing_stage is None else "REJECTED"
    return "PASS_WITH_FALLBACK" if outcome.used_fallback_plan else "PASS"


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


def _target_matches_expectation(
    primary_semantic_label: str | None, expected_target_category: str
) -> bool:
    """Loose, case-insensitive substring match -- mirrors the same style of keyword matching

    `analysis/plan_builder.py::_motion_spec_for` already uses for this project's semantic
    labels (e.g. `expected_target_category="hair"` matches a VLM-produced
    `primary_semantic_label="character_hair"`), rather than requiring exact string equality
    that would never match in practice.
    """
    if not primary_semantic_label:
        return False
    return expected_target_category.strip().lower() in primary_semantic_label.strip().lower()


def _check_regression(outcome: PageRunOutcome, sample: EvalSample | None) -> bool:
    """True if this outcome reproduced the specific, named, known-bad case a sample's

    `regression_reference` describes.

    A completion is NOT inherently a violation -- several flagged samples' own
    `acceptable_outcome` explicitly allow a validated ACCEPT as a correct result (e.g.
    `phase3_action_page`: "a validated ACCEPT on a real weapon-shaped or cloth-banner-shaped
    region" is explicitly acceptable). This only flags a violation when there is real,
    structured evidence the WRONG target was selected: `status == "completed"` AND
    `sample.expected_target_category` is recorded AND the outcome's `primary_semantic_label`
    does not match it (Phase 3.4 baseline cleanup fix -- the previous implementation flagged
    *any* completed outcome on a flagged sample, which would have falsely flagged a genuinely
    correct `phase3_action_page` completion as a regression).

    When no `expected_target_category` is recorded (a genuinely open question for some samples
    -- see `phase3_action_page`'s own manifest entry, which deliberately does not assert a
    single correct target among several plausible ones), this function cannot distinguish a
    correct completion from a regression automatically and returns `False`; that specific
    historical defect is instead re-verified by deliberate reproduction (see
    docs/phase3.3-results.md's "Phase 3.1/3.2 regression re-verification"), not by this
    automatic per-run check.
    """
    if sample is None or not sample.regression_reference:
        return False
    if outcome.status != "completed":
        return False
    if sample.expected_target_category is None:
        return False
    return not _target_matches_expectation(
        outcome.primary_semantic_label, sample.expected_target_category
    )


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
    unresolved_ground_truth = 0
    for outcome in outcomes:
        sample = samples.get(outcome.sample_id)
        if sample is not None and sample.regression_reference:
            regression_checked += 1
            if _check_regression(outcome, sample):
                regression_violated += 1

        if sample is None:
            continue
        if sample.ground_truth_uncertain:
            # An unresolved ground truth must never be silently treated as a known positive or
            # negative label -- gated on the dedicated `ground_truth_uncertain` flag, not on
            # `animation_possible == "uncertain"` (Phase 3.4 baseline cleanup fix): a sample
            # could in principle carry a hedged "yes"/"no" that is still marked uncertain, and
            # the previous check would have let it silently contaminate fp/fn.
            unresolved_ground_truth += 1
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

    all_object_outcomes = [oo for o in outcomes for oo in o.object_outcomes]
    secondary_outcomes = [oo for oo in all_object_outcomes if oo.motion_type == "secondary"]
    micro_outcomes = [oo for oo in all_object_outcomes if oo.motion_type == "micro"]
    secondary_rendered = sum(1 for oo in secondary_outcomes if oo.status == "rendered")
    micro_rendered = sum(1 for oo in micro_outcomes if oo.status == "rendered")

    statuses = [classify_outcome(o, samples.get(o.sample_id)) for o in outcomes]
    status_breakdown = StatusBreakdown(
        pass_count=statuses.count("PASS"),
        pass_with_fallback_count=statuses.count("PASS_WITH_FALLBACK"),
        rejected_count=statuses.count("REJECTED"),
        error_count=statuses.count("ERROR"),
    )

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
        unresolved_ground_truth_count=unresolved_ground_truth,
        regression_violation_count=regression_violated,
        regression_samples_checked=regression_checked,
        panel_detection_multi_panel_rate=panel_rate,
        secondary_object_render_rate=Rate(secondary_rendered, len(secondary_outcomes)),
        micro_object_render_rate=Rate(micro_rendered, len(micro_outcomes)),
        status_breakdown=status_breakdown,
    )
