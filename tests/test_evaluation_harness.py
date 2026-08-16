"""Tests for src/manga_animation/evaluation/harness.py -- the shared per-sample runner logic

Phase 9 extracted out of `scripts/run_phase3_3_evaluation.py` so both that script and
`scripts/run_phase9_evaluation.py` reuse it instead of duplicating it. Only the pure,
model-free helpers are tested here (`render_summary_from_result`, `object_outcome_motion_type`)
-- `run_one_sample`/`run_nondeterminism_check` themselves call the real pipeline/VLM and are
only exercised for real on the remote GPU worker (ADR 0003), matching this project's existing
`scripts/run_phase3_3_evaluation.py` testing boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import manga_animation.evaluation.harness as harness
from manga_animation.evaluation.dataset import EvalSample
from manga_animation.evaluation.harness import (
    object_outcome_motion_type,
    render_summary_from_result,
)
from manga_animation.pipeline.types import LoopMetrics, PipelineStageError, RenderResult
from manga_animation.schemas.animation_plan import MotionType


def test_render_summary_from_result_carries_loop_metrics_through():
    render = RenderResult(
        output_path=Path("out.mp4"),
        frame_count=48,
        fps=24.0,
        resolution=(640, 480),
        duration_s=2.0,
        codec="h264",
        pixel_format="yuv420p",
        seamless_loop_verified=True,
        loop_metrics=LoopMetrics(
            ordinary_adjacent_step_mean_abs_diff=1.2,
            wrap_step_mean_abs_diff=1.4,
            wrap_step_within_2x_ordinary=True,
            ordinary_adjacent_step_ssim=0.99,
            wrap_step_ssim=0.98,
            wrap_ssim_within_tolerance=True,
        ),
    )
    summary = render_summary_from_result(render)
    assert summary.frame_count == 48
    assert summary.resolution == (640, 480)
    assert summary.loop_metrics is not None
    assert summary.loop_metrics.wrap_step_ssim == pytest.approx(0.98)


def test_render_summary_from_result_handles_a_missing_loop_metrics():
    render = RenderResult(
        output_path=Path("out.mp4"),
        frame_count=2,
        fps=24.0,
        resolution=(640, 480),
        duration_s=0.1,
        codec="h264",
        pixel_format="yuv420p",
        seamless_loop_verified=True,
        loop_metrics=None,
    )
    summary = render_summary_from_result(render)
    assert summary.loop_metrics is None
    assert summary.seam_artifact_suspected is None  # not computed unless the caller passes it


def test_render_summary_from_result_threads_seam_artifact_suspected_through():
    render = RenderResult(
        output_path=Path("out.mp4"),
        frame_count=2,
        fps=24.0,
        resolution=(640, 480),
        duration_s=0.1,
        codec="h264",
        pixel_format="yuv420p",
        seamless_loop_verified=True,
        loop_metrics=None,
    )
    assert (
        render_summary_from_result(render, seam_artifact_suspected=True).seam_artifact_suspected
        is True
    )
    assert (
        render_summary_from_result(render, seam_artifact_suspected=False).seam_artifact_suspected
        is False
    )


def test_seam_artifact_suspected_returns_none_on_an_undecodable_path(tmp_path):
    from manga_animation.evaluation.harness import _seam_artifact_suspected

    render = RenderResult(
        output_path=tmp_path / "does_not_exist.mp4",
        frame_count=2,
        fps=24.0,
        resolution=(640, 480),
        duration_s=0.1,
        codec="h264",
        pixel_format="yuv420p",
        seamless_loop_verified=True,
        loop_metrics=None,
    )
    assert _seam_artifact_suspected(render) is None


def test_object_outcome_motion_type_maps_secondary_and_micro():
    assert object_outcome_motion_type(MotionType.SECONDARY) == "secondary"
    assert object_outcome_motion_type(MotionType.MICRO) == "micro"


def test_object_outcome_motion_type_rejects_primary_and_static():
    with pytest.raises(ValueError, match="unexpected motion_type"):
        object_outcome_motion_type(MotionType.PRIMARY)
    with pytest.raises(ValueError, match="unexpected motion_type"):
        object_outcome_motion_type(MotionType.STATIC)


def test_object_description_outcome_from_result_returns_none_for_none():
    """The description stage didn't run for this object -- `None` in, `None` out."""
    assert harness.object_description_outcome_from_result(None) is None


def test_object_description_outcome_from_result_mirrors_the_real_result():
    from manga_animation.pipeline.types import ObjectDescriptionResult
    from manga_animation.schemas.animation_plan import (
        MotionSpec,
        TransformKind,
        Vector2,
    )

    result = ObjectDescriptionResult(
        object_id="obj_1",
        accepted=True,
        assessment="pass",
        matches_semantic_label=True,
        animatable=True,
        object_identity="character_hair",
        motion_spec=MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.03,
            speed=1.0,
        ),
        movable_parts=("hair",),
        static_parts=("face",),
        constraints=("keep the face static",),
        neighbor_conflicts=(),
        confidence=0.9,
        reason="hair sways in the sprint",
        rejection_reason=None,
        model_id="fake-qwen",
        raw_responses=('{"bbox_assessment": "pass"}',),
    )
    outcome = harness.object_description_outcome_from_result(result)
    assert outcome is not None
    assert outcome.accepted is True
    assert outcome.assessment == "pass"
    assert outcome.object_identity == "character_hair"
    assert outcome.motion is not None
    assert outcome.motion["transform_kind"] == "translate"
    assert outcome.confidence == pytest.approx(0.9)
    assert outcome.rejection_reason is None


def test_run_one_sample_preserves_primary_object_description_on_failure(monkeypatch, tmp_path):
    from manga_animation.pipeline.types import ObjectDescriptionResult

    description = ObjectDescriptionResult(
        object_id="primary",
        accepted=False,
        assessment="ambiguous",
        matches_semantic_label=None,
        animatable=None,
        object_identity=None,
        motion_spec=None,
        reason="two characters in the box",
        rejection_reason="bbox_assessment=ambiguous",
        model_id="fake-qwen",
        raw_responses=("{}",),
    )

    def fail(*args, **kwargs):
        raise PipelineStageError(
            stage="object_description",
            input_ref="primary",
            detail="semantic rejection",
            object_description=description,
        )

    monkeypatch.setattr(harness, "run_pipeline", fail)
    sample = EvalSample(
        sample_id="sample",
        image_path="missing.png",
        source_citation="test",
        diversity_tag="test",
        acceptable_outcome="reject",
    )
    outcome = harness.run_one_sample(
        sample,
        "panel",
        config=None,
        clients=(None, None, None, None),
        out_dir=tmp_path,
        panel_count=1,
        panel_sources=["fallback_full_page"],
    )
    assert outcome.status == "failed"
    assert outcome.failing_stage == "object_description"
    assert outcome.primary_object_description is not None
    assert outcome.primary_object_description.assessment == "ambiguous"
