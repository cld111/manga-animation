from __future__ import annotations

import numpy as np
import pytest

from manga_animation.grounding.client import Detection
from manga_animation.grounding.ground import (
    _prompt_from_label,
    ground_object,
    ground_object_candidates,
)
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import MotionType, ObjectPlan


class FakeGroundingClient:
    """A `GroundingClient` double: canned detections, no torch/transformers/network."""

    model_id = "fake-grounding"

    def __init__(self, detections: list[Detection]):
        self.detections = detections
        self.last_prompt: str | None = None
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.last_prompt = text_prompt
        return self.detections

    def unload(self) -> None:
        self.unloaded = True


def make_object_plan(semantic_label: str = "flag_cloth", object_id: str = "obj_1") -> ObjectPlan:
    return ObjectPlan(
        object_id=object_id,
        panel_id="panel_1",
        semantic_label=semantic_label,
        confidence=0.9,
        motion_type=MotionType.STATIC,
        motion=None,
    )


def make_image(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- prompt conversion ---------------------------------------------------------


def test_prompt_from_label_converts_underscores_to_spaces():
    assert _prompt_from_label("flag_cloth") == "flag cloth."


def test_prompt_from_label_single_word():
    assert _prompt_from_label("hair") == "hair."


# --- ground_object ---------------------------------------------------------


def test_ground_object_picks_the_highest_scoring_detection():
    detections = [
        Detection(label="flag cloth", score=0.31, box=(5, 5, 20, 20)),
        Detection(label="flag cloth", score=0.82, box=(10, 10, 60, 60)),
    ]
    client = FakeGroundingClient(detections)
    plan = make_object_plan()

    result = ground_object(make_image(), plan, client)

    assert result.bbox.score == pytest.approx(0.82)
    assert result.bbox.as_xyxy() == (10, 10, 60, 60)
    assert result.object_id == "obj_1"
    assert result.model_id == "fake-grounding"


def test_ground_object_sends_the_converted_prompt_to_the_client():
    client = FakeGroundingClient([Detection(label="hair", score=0.5, box=(0, 0, 10, 10))])
    ground_object(make_image(), make_object_plan(semantic_label="character_hair"), client)
    assert client.last_prompt == "character hair."


def test_ground_object_raises_when_nothing_detected():
    client = FakeGroundingClient([])
    with pytest.raises(PipelineStageError) as exc_info:
        ground_object(make_image(), make_object_plan(), client)
    assert exc_info.value.stage == "grounding"
    assert exc_info.value.input_ref == "obj_1"


def test_ground_object_clips_box_to_image_bounds():
    client = FakeGroundingClient([Detection(label="hair", score=0.9, box=(90, 90, 150, 150))])
    result = ground_object(make_image(h=100, w=100), make_object_plan(), client)
    assert result.bbox.as_xyxy() == (90, 90, 100, 100)


def test_ground_object_raises_when_box_lies_entirely_outside_image():
    client = FakeGroundingClient([Detection(label="hair", score=0.9, box=(200, 200, 250, 250))])
    with pytest.raises(PipelineStageError) as exc_info:
        ground_object(make_image(h=100, w=100), make_object_plan(), client)
    assert exc_info.value.stage == "grounding"


def test_ground_object_clips_negative_coordinates():
    client = FakeGroundingClient([Detection(label="hair", score=0.9, box=(-10, -10, 50, 50))])
    result = ground_object(make_image(h=100, w=100), make_object_plan(), client)
    assert result.bbox.as_xyxy() == (0, 0, 50, 50)


# --- ground_object_candidates (Phase 3.2: ranked, not just best-of) ------------------------


def test_ground_object_candidates_returns_every_detection_ranked_by_score():
    detections = [
        Detection(label="flag cloth", score=0.31, box=(5, 5, 20, 20)),
        Detection(label="flag cloth", score=0.82, box=(10, 10, 60, 60)),
        Detection(label="flag cloth", score=0.55, box=(15, 15, 40, 40)),
    ]
    client = FakeGroundingClient(detections)

    candidates = ground_object_candidates(make_image(), make_object_plan(), client)

    assert [c.bbox.score for c in candidates] == pytest.approx([0.82, 0.55, 0.31])
    assert all(c.object_id == "obj_1" for c in candidates)


def test_ground_object_candidates_caps_at_max_candidates():
    detections = [
        Detection(label="hair", score=0.9 - 0.1 * i, box=(i, i, i + 10, i + 10)) for i in range(6)
    ]
    client = FakeGroundingClient(detections)

    candidates = ground_object_candidates(
        make_image(), make_object_plan(), client, max_candidates=3
    )

    assert len(candidates) == 3
    assert [c.bbox.score for c in candidates] == pytest.approx([0.9, 0.8, 0.7])


def test_ground_object_candidates_skips_degenerate_boxes_but_keeps_usable_ones():
    detections = [
        Detection(label="hair", score=0.9, box=(200, 200, 250, 250)),  # entirely outside
        Detection(label="hair", score=0.5, box=(10, 10, 40, 40)),  # usable
    ]
    client = FakeGroundingClient(detections)

    candidates = ground_object_candidates(make_image(h=100, w=100), make_object_plan(), client)

    assert len(candidates) == 1
    assert candidates[0].bbox.as_xyxy() == (10, 10, 40, 40)


def test_ground_object_candidates_raises_when_every_detection_is_degenerate():
    detections = [Detection(label="hair", score=0.9, box=(200, 200, 250, 250))]
    client = FakeGroundingClient(detections)

    with pytest.raises(PipelineStageError) as exc_info:
        ground_object_candidates(make_image(h=100, w=100), make_object_plan(), client)
    assert exc_info.value.stage == "grounding"


def test_ground_object_delegates_to_candidates_and_returns_the_top_one():
    detections = [
        Detection(label="hair", score=0.31, box=(5, 5, 20, 20)),
        Detection(label="hair", score=0.82, box=(10, 10, 60, 60)),
    ]
    client = FakeGroundingClient(detections)

    result = ground_object(make_image(), make_object_plan(), client)

    assert result.bbox.score == pytest.approx(0.82)
