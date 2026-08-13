"""Typed records of what actually happened running one evaluation sample through the real

pipeline -- the input the metrics in `evaluation/metrics.py` are computed from. Deliberately
independent of `PipelineRunResult`/`PipelineStageError` (which carry live, non-serializable
objects like numpy arrays and `AnimationPlan`s) so a real run's outcome can be recorded to JSON
(`scripts/run_phase3_3_evaluation.py`) and the metrics recomputed later without re-running the
pipeline, and so metric computation itself is testable with plain fixtures -- no torch, no
real model clients, matching every other stage's fake-client test style in this project.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from manga_animation.pipeline.types import Stage

FailingStage = Stage | Literal["unexpected"]
"""`Stage` (the exact 8 real `PipelineStageError.stage` values) plus one evaluation-harness-only

value: `"unexpected"`, for a bare (non-`PipelineStageError`) exception the harness could not
attribute to a specific pipeline stage at all (`scripts/run_phase3_3_evaluation.py`'s
`except Exception` path). Phase 8 fix: `PageRunOutcome.failing_stage` was previously typed as
plain `Stage | None`, which does not include `"unexpected"` -- a real, disclosed, latent bug
(see docs/phase7-results.md section 13): constructing `PageRunOutcome(failing_stage=
"unexpected", ...)` would have raised `pydantic.ValidationError` the first time that exception
path actually fired for real, masking the original exception with an unrelated validation
error. Never fixed in `pipeline.types.Stage` itself -- that type stays exact/closed, since every
real `PipelineStageError.stage` value genuinely is one of the 8 real stages; this is a strictly
evaluation-reporting concern, kept local to this module."""


class ValidationAttemptOutcome(BaseModel):
    """Mirrors the fields of `pipeline.types.ValidationResult` that matter for evaluation --

    not the type itself, since that dataclass carries no serialization guarantee and this
    needs to round-trip through JSON (see this module's docstring).
    """

    candidate_rank: int
    accepted: bool
    grounding_score: float | None
    reason: str


ObjectOutcomeStatus = Literal["rendered", "dropped"]
ObjectOutcomeMotionType = Literal["secondary", "micro"]


class ObjectAttemptOutcome(BaseModel):
    """One non-PRIMARY (SECONDARY/MICRO) object's own attempt outcome within a
    `PageRunOutcome` (Phase 7.2.1) -- extends single-PRIMARY-only evaluation reporting to
    every other object a plan proposed, closing the gap ADR 0010 explicitly deferred to
    Phase 7 ("extending evaluation to report on secondary/micro objects too is real future
    work... at Phase 7, not here").

    PRIMARY is deliberately NOT represented here -- it stays exactly where it already was
    (`PageRunOutcome.primary_semantic_label`/`primary_motion_type`/`validation_attempts`),
    unchanged in meaning: a PRIMARY failure already fails the whole run (ADR 0010's PRIMARY
    failure policy), so there is at most one PRIMARY per outcome and it was already fully
    reported by those pre-existing fields. This type only ever describes the objects that
    could legitimately be dropped without failing the run.
    """

    object_id: str
    semantic_label: str
    motion_type: ObjectOutcomeMotionType
    status: ObjectOutcomeStatus
    """"rendered" -- grounded, validated, segmented, and included in the final composited
    output (mirrors one of `PipelineRunResult.secondary_objects`). "dropped" -- failed at
    grounding or validation and was excluded per ADR 0010's failure policy (mirrors one of
    `PipelineRunResult.dropped_objects`) -- this does NOT mean the whole page run failed."""
    validation_attempts: list[ValidationAttemptOutcome] = []


class LoopMetricsOutcome(BaseModel):
    """Mirrors `pipeline.types.LoopMetrics` for JSON round-tripping -- see this module's

    docstring for why a serializable mirror type is used instead of the dataclass itself.
    """

    ordinary_adjacent_step_mean_abs_diff: float
    wrap_step_mean_abs_diff: float
    wrap_step_within_2x_ordinary: bool
    ordinary_adjacent_step_ssim: float
    wrap_step_ssim: float
    wrap_ssim_within_tolerance: bool


class RenderSummary(BaseModel):
    """Mirrors the fields of `pipeline.types.RenderResult` that matter for evaluation -- the

    Phase 8 brief's required "output dimensions; frame count; FPS; duration; codec/container;
    loop metrics" fields, for one completed `PageRunOutcome`.
    """

    frame_count: int
    fps: float
    resolution: tuple[int, int]
    duration_s: float
    codec: str
    pixel_format: str
    seamless_loop_verified: bool
    loop_metrics: LoopMetricsOutcome | None = None
    seam_artifact_suspected: bool | None = None
    """Phase 9: `evaluation.artifacts.detect_seam_like_artifacts` run on the actual decoded

    rendered frames -- `True`/`False` when computed, `None` when not (e.g. a producer with
    `schema_version < 4`, or a decode failure). A real, evidence-validated signal for one
    specific, narrow defect class only (a rigid, axis-aligned changed-region edge -- the real
    "vertical seam" defect `docs/phase8.3-results.md` root-caused and fixed at the segmentation
    stage) -- NOT a general artifact-free claim, and does not replace `seamless_loop_verified`
    (loop-wrap continuity) or human visual QA. See `evaluation.artifacts`'s own module
    docstring for the real validation evidence behind this check."""


class PageRunOutcome(BaseModel):
    """One real (or faked, in tests) `run_pipeline` invocation's outcome for one evaluation

    sample, under one `analysis_mode`.
    """

    sample_id: str
    analysis_mode: Literal["page", "panel"]
    status: Literal["completed", "failed"]
    failing_stage: FailingStage | None = None
    failure_detail: str | None = None
    used_fallback_plan: bool = False
    panel_count: int | None = None
    panel_sources: list[str] = []
    primary_semantic_label: str | None = None
    primary_motion_type: str | None = None
    validation_attempts: list[ValidationAttemptOutcome] = []
    object_outcomes: list[ObjectAttemptOutcome] = []
    """Phase 7.2.1: one entry per SECONDARY/MICRO object the plan proposed for this page (both

    rendered and dropped ones) -- see `ObjectAttemptOutcome`. Empty (the schema default) for
    every outcome recorded before this field existed (`schema_version < 2`) AND for a
    single-object plan with no SECONDARY/MICRO candidates at all -- `schema_version`
    distinguishes those two empty-list cases; this field alone cannot.
    """
    render_summary: RenderSummary | None = None
    """Phase 8: what was actually measured about the rendered output, for every `status ==

    "completed"` outcome -- `None` for a failed run (nothing was rendered) and for every
    outcome recorded before this field existed (`schema_version < 3`, see below). Populated
    from `PipelineRunResult.render` (`pipeline/types.py::RenderResult`) at the same point
    `primary_semantic_label`/`validation_attempts`/etc. already are."""
    schema_version: int = Field(
        default=1,
        ge=1,
        description=(
            "1 = pre-Phase-7.2.1 producers (object_outcomes was never populated -- PRIMARY-"
            "only reporting, matching every PageRunOutcome recorded before this field "
            "existed). 2 = Phase 7.2.1 onward: a producer that populates object_outcomes for "
            "every SECONDARY/MICRO object the plan proposed, even when the resulting list is "
            "empty because none existed. 3 = Phase 8 onward: a producer that also populates "
            "render_summary for every completed outcome. 4 = Phase 9 onward: a producer that "
            "also populates render_summary.seam_artifact_suspected for every completed "
            "outcome (see manga_animation.evaluation.harness.run_one_sample). This is a "
            "PREDICTION-schema version a producer sets when it constructs a record -- unlike "
            "EvalSample.annotation_version (ADR 0009), which is a ground-truth-revision signal "
            "only a human bumps by hand -- so it is set programmatically wherever a "
            "PageRunOutcome is actually constructed (see "
            "manga_animation.evaluation.harness.run_one_sample), not left to drift."
        ),
    )
