"""Tests for src/manga_animation/evaluation/visual_qa.py -- the Phase 9 human/AI visual-quality

scoring protocol and capability-matrix tooling.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_animation.evaluation.visual_qa import (
    CAPABILITY_DIMENSIONS,
    VISUAL_QA_DIMENSIONS,
    VISUAL_QA_SCALE,
    CapabilityMatrixEntry,
    VisualQAScore,
    build_capability_matrix,
)


def _score(**overrides) -> VisualQAScore:
    defaults = dict(
        sample_id="realworld_wind_breaker_sprint",
        analysis_mode="panel",
        evaluator="claude (test fixture)",
        evaluated_at="2026-08-13T00:00:00Z",
        has_video=True,
    )
    defaults.update(overrides)
    return VisualQAScore(**defaults)


def test_visual_qa_scale_covers_every_score_value_0_through_5():
    assert set(VISUAL_QA_SCALE.keys()) == {0, 1, 2, 3, 4, 5}
    for text in VISUAL_QA_SCALE.values():
        assert text  # every definition is real, non-empty text


def test_visual_qa_score_defaults_to_no_scores_and_no_failures():
    score = _score()
    assert score.mean_score is None
    assert score.failure_categories == []
    assert set(score.dimension_scores.keys()) == set(VISUAL_QA_DIMENSIONS)


def test_visual_qa_score_mean_score_averages_only_scored_dimensions():
    score = _score(target_correctness=5, motion_correctness=3, mask_quality=4)
    assert score.mean_score == pytest.approx((5 + 3 + 4) / 3)


def test_visual_qa_score_rejects_an_out_of_range_value():
    with pytest.raises(ValidationError):
        _score(target_correctness=6)
    with pytest.raises(ValidationError):
        _score(loop_quality=-1)


def test_visual_qa_score_rejects_an_unknown_failure_category():
    with pytest.raises(ValidationError):
        _score(failure_categories=["not_a_real_category"])


def test_capability_matrix_entry_requires_evidence_for_a_non_unknown_verdict():
    with pytest.raises(ValueError, match="cites no evidence_sample_ids"):
        CapabilityMatrixEntry(dimension="rotation", verdict="WORKS_WELL")


def test_capability_matrix_entry_allows_unknown_with_no_evidence():
    entry = CapabilityMatrixEntry(dimension="rotation", verdict="UNKNOWN")
    assert entry.evidence_sample_ids == []


def test_build_capability_matrix_defaults_every_dimension_to_unknown():
    matrix = build_capability_matrix([])
    assert set(matrix.keys()) == set(CAPABILITY_DIMENSIONS)
    assert all(entry.verdict == "UNKNOWN" for entry in matrix.values())


def test_build_capability_matrix_applies_explicit_entries():
    entries = [
        CapabilityMatrixEntry(
            dimension="rotation", verdict="WORKS_WELL", evidence_sample_ids=["a", "b"]
        ),
        CapabilityMatrixEntry(dimension="occlusion", verdict="FAILS", evidence_sample_ids=["c"]),
    ]
    matrix = build_capability_matrix(entries)
    assert matrix["rotation"].verdict == "WORKS_WELL"
    assert matrix["rotation"].evidence_sample_ids == ["a", "b"]
    assert matrix["occlusion"].verdict == "FAILS"
    assert matrix["single_object"].verdict == "UNKNOWN"  # untouched dimensions stay UNKNOWN


def test_build_capability_matrix_rejects_a_duplicate_dimension():
    entries = [
        CapabilityMatrixEntry(
            dimension="rotation", verdict="WORKS_WELL", evidence_sample_ids=["a"]
        ),
        CapabilityMatrixEntry(dimension="rotation", verdict="FAILS", evidence_sample_ids=["b"]),
    ]
    with pytest.raises(ValueError, match="duplicate capability entry"):
        build_capability_matrix(entries)


def test_build_capability_matrix_rejects_an_unknown_dimension_name():
    bad_entry = CapabilityMatrixEntry.__new__(CapabilityMatrixEntry)
    object.__setattr__(bad_entry, "dimension", "not_a_real_dimension")
    object.__setattr__(bad_entry, "verdict", "UNKNOWN")
    object.__setattr__(bad_entry, "evidence_sample_ids", [])
    object.__setattr__(bad_entry, "notes", "")
    with pytest.raises(ValueError, match="unknown capability dimension"):
        build_capability_matrix([bad_entry])
