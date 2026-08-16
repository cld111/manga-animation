"""Phase 18.4: per-stage disk persistence round-trips.

The batch entry point (`run_pages`) checkpoints each model stage to disk so a killed
session resumes without re-loading completed models. These tests verify the
serialization/deserialization layer itself round-trips every dataclass/pydantic field
exactly -- independent of the orchestration tests in tests/test_lifecycle.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manga_animation.pipeline.orchestrator import DroppedObjectResult
from manga_animation.pipeline.persistence import (
    load_descriptions,
    load_grounding,
    load_segmentation,
    save_descriptions,
    save_grounding,
    save_segmentation,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    ObjectDescriptionResult,
    SegmentationResult,
)
from manga_animation.schemas.animation_plan import MotionSpec, MotionType, ObjectPlan


def _object_plan(object_id: str) -> ObjectPlan:
    return ObjectPlan(
        object_id=object_id,
        panel_id="panel_001",
        semantic_label="character",
        confidence=0.9,
        motion_type=MotionType.SECONDARY,
        motion=MotionSpec(transform_kind="mesh_warp", amplitude=0.01),
    )


def _grounding(object_id: str) -> GroundingResult:
    return GroundingResult(
        object_id=object_id,
        bbox=BBoxPx(x0=10, y0=20, x1=60, y1=90, score=0.87),
        model_id="fake-dino",
    )


def _description(box_index: int, *, accepted: bool) -> ObjectDescriptionResult:
    return ObjectDescriptionResult(
        object_id=f"obj_{box_index}",
        accepted=accepted,
        assessment="pass" if accepted else "reject",
        matches_semantic_label=True,
        animatable=accepted,
        object_identity="character" if accepted else None,
        motion_spec=(
            MotionSpec(transform_kind="mesh_warp", amplitude=2.5) if accepted else None
        ),
        movable_parts=("hair", "arm"),
        static_parts=("face",),
        constraints=("keep the face static",),
        neighbor_conflicts=("speech bubble behind",),
        confidence=0.93 if accepted else None,
        reason="fake description",
        rejection_reason=None if accepted else "bbox_assessment=reject",
        model_id="fake-qwen",
        raw_responses=('{"box_index": 0}',),
    )


def _segmentation(object_id: str) -> SegmentationResult:
    return SegmentationResult(
        object_id=object_id,
        mask=np.zeros((160, 120), dtype=np.uint8),
        bbox=BBoxPx(x0=10, y0=20, x1=60, y1=90),
        model_id="fake-sam",
        iou_score=0.91,
    )


def test_grounding_round_trip(tmp_path: Path):
    candidates = {
        "panel_001": {
            "obj_character_0": [_grounding("obj_character_0"), _grounding("obj_character_0")],
            "obj_flag_1": [_grounding("obj_flag_1")],
        },
        "panel_002": {"obj_character_0": [_grounding("obj_character_0")]},
    }
    plans = {
        "panel_001": {
            "obj_character_0": _object_plan("obj_character_0"),
            "obj_flag_1": _object_plan("obj_flag_1"),
        },
        "panel_002": {"obj_character_0": _object_plan("obj_character_0")},
    }
    dropped = {
        "panel_001": [
            DroppedObjectResult(
                object_plan=_object_plan("obj_weapon_2"),
                failing_stage="grounding",
                reason="no detection above threshold",
            )
        ]
    }
    save_grounding(tmp_path, candidates, plans, dropped)
    loaded_candidates, loaded_plans, loaded_dropped = load_grounding(tmp_path)

    assert loaded_candidates == candidates
    assert loaded_plans == plans
    assert len(loaded_dropped["panel_001"]) == 1
    restored = loaded_dropped["panel_001"][0]
    assert restored.failing_stage == "grounding"
    assert restored.reason == "no detection above threshold"
    assert restored.object_plan.object_id == "obj_weapon_2"


def test_description_round_trip_preserves_motion_spec_and_audit_trail(tmp_path: Path):
    descriptions = {
        "panel_001": {
            ("obj_character_0", 0): _description(0, accepted=True),
            ("obj_character_0", 1): _description(1, accepted=False),
        }
    }
    save_descriptions(tmp_path, descriptions)
    loaded = load_descriptions(tmp_path)

    assert set(loaded["panel_001"]) == {("obj_character_0", 0), ("obj_character_0", 1)}
    accepted = loaded["panel_001"][("obj_character_0", 0)]
    assert accepted.accepted is True
    assert accepted.assessment == "pass"
    assert accepted.motion_spec == descriptions["panel_001"][("obj_character_0", 0)].motion_spec
    assert accepted.motion_spec.transform_kind == "mesh_warp"  # type: ignore[union-attr]
    assert accepted.movable_parts == ("hair", "arm")
    assert accepted.static_parts == ("face",)
    assert accepted.constraints == ("keep the face static",)
    assert accepted.neighbor_conflicts == ("speech bubble behind",)
    assert accepted.raw_responses == ('{"box_index": 0}',)
    assert accepted.model_id == "fake-qwen"

    rejected = loaded["panel_001"][("obj_character_0", 1)]
    assert rejected.accepted is False
    assert rejected.assessment == "reject"
    assert rejected.rejection_reason == "bbox_assessment=reject"
    assert rejected.motion_spec is None


def test_description_json_is_human_readable(tmp_path: Path):
    descriptions = {"panel_001": {("obj_character_0", 0): _description(0, accepted=True)}}
    save_descriptions(tmp_path, descriptions)
    payload = json.loads((tmp_path / "descriptions.json").read_text())
    assert "panel_001" in payload
    entry = payload["panel_001"]["obj_character_0|0"]
    assert entry["assessment"] == "pass"
    assert entry["motion_spec"]["transform_kind"] == "mesh_warp"


def test_segmentation_round_trip_preserves_mask_pixels(tmp_path: Path):
    mask = np.zeros((160, 120), dtype=np.uint8)
    mask[30:80, 40:90] = 255
    segmentation = {
        "panel_001": {
            ("obj_character_0", 0): SegmentationResult(
                object_id="obj_character_0",
                mask=mask,
                bbox=BBoxPx(x0=40, y0=30, x1=90, y1=80),
                model_id="fake-sam",
                iou_score=0.91,
            )
        }
    }
    save_segmentation(tmp_path, segmentation)
    loaded = load_segmentation(tmp_path)

    assert set(loaded["panel_001"]) == {("obj_character_0", 0)}
    result = loaded["panel_001"][("obj_character_0", 0)]
    assert result.object_id == "obj_character_0"
    assert result.bbox == segmentation["panel_001"][("obj_character_0", 0)].bbox
    assert result.model_id == "fake-sam"
    assert result.iou_score == 0.91
    np.testing.assert_array_equal(result.mask, mask)
