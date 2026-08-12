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

from pydantic import BaseModel

from manga_animation.pipeline.types import Stage


class ValidationAttemptOutcome(BaseModel):
    """Mirrors the fields of `pipeline.types.ValidationResult` that matter for evaluation --

    not the type itself, since that dataclass carries no serialization guarantee and this
    needs to round-trip through JSON (see this module's docstring).
    """

    candidate_rank: int
    accepted: bool
    grounding_score: float | None
    reason: str


class PageRunOutcome(BaseModel):
    """One real (or faked, in tests) `run_pipeline` invocation's outcome for one evaluation

    sample, under one `analysis_mode`.
    """

    sample_id: str
    analysis_mode: Literal["page", "panel"]
    status: Literal["completed", "failed"]
    failing_stage: Stage | None = None
    failure_detail: str | None = None
    used_fallback_plan: bool = False
    panel_count: int | None = None
    panel_sources: list[str] = []
    primary_semantic_label: str | None = None
    primary_motion_type: str | None = None
    validation_attempts: list[ValidationAttemptOutcome] = []
