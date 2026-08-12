from __future__ import annotations

import json

import numpy as np
import pytest

from manga_animation.pipeline.types import BBoxPx, GroundingResult
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PivotSpec,
    TransformKind,
    Vector2,
)
from manga_animation.validation.validate import validate_target


class FakeVLMClient:
    """A `VLMClient` double for the validation stage: returns a canned response string,

    records every (image, prompt) call it received.
    """

    def __init__(self, response: str):
        self._response = response
        self.prompts: list[str] = []
        self.images: list = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        self.images.append(image)
        return self._response


def _verification_json(matches: bool, confidence: float = 0.9, reason: str = "fake reason") -> str:
    return json.dumps({"matches": matches, "confidence": confidence, "reason": reason})


def make_image(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_object_plan(
    semantic_label: str = "flag_banner", transform_kind: TransformKind = TransformKind.MESH_WARP
) -> ObjectPlan:
    return ObjectPlan(
        object_id="obj_1",
        panel_id="panel_1",
        semantic_label=semantic_label,
        confidence=0.8,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=transform_kind,
            direction=Vector2(x=1.0, y=0.0) if transform_kind == TransformKind.TRANSLATE else None,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    )


def make_grounding(bbox: BBoxPx | None = None, score: float = 0.5) -> GroundingResult:
    resolved = bbox or BBoxPx(x0=50, y0=50, x1=100, y1=100, score=score)
    return GroundingResult(object_id="obj_1", bbox=resolved, model_id="fake-grounding")


# --- accept / reject on semantic agreement ------------------------------------------------


def test_validate_target_accepts_when_bbox_plausible_and_vlm_agrees():
    client = FakeVLMClient(_verification_json(True, confidence=0.92, reason="clearly a banner"))
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is True
    assert result.semantic_match is True
    assert result.semantic_confidence == pytest.approx(0.92)
    assert result.bbox_plausible is True
    assert result.reason == "clearly a banner"


def test_validate_target_rejects_when_vlm_disagrees_even_at_high_grounding_score():
    """Real Phase 3.1 finding this stage exists to catch: a technically valid, high-scoring

    detection is not automatically correct.
    """
    client = FakeVLMClient(_verification_json(False, confidence=0.85, reason="this is a face"))
    high_score_grounding = make_grounding(score=0.9)

    result = validate_target(make_image(), make_object_plan(), high_score_grounding, client)

    assert result.accepted is False
    assert result.semantic_match is False
    assert result.grounding_score == pytest.approx(0.9)
    assert "face" in result.reason


# --- bbox plausibility pre-filter (no model call needed) ----------------------------------


def test_validate_target_rejects_implausible_bbox_without_calling_the_vlm():
    client = FakeVLMClient(_verification_json(True))
    # ~99% of a 200x200 image -- a real grounded object practically never covers this much
    huge_box = BBoxPx(x0=0, y0=0, x1=199, y1=199, score=0.5)

    result = validate_target(make_image(), make_object_plan(), make_grounding(huge_box), client)

    assert result.accepted is False
    assert result.bbox_plausible is False
    assert result.semantic_match is None
    assert result.semantic_confidence is None
    assert client.prompts == []  # never spent a VLM call on an already-implausible box


def test_validate_target_accepts_a_small_plausible_bbox_when_vlm_agrees():
    client = FakeVLMClient(_verification_json(True))
    small_box = BBoxPx(x0=90, y0=90, x1=110, y1=110, score=0.4)  # 1% of a 200x200 image

    result = validate_target(make_image(), make_object_plan(), make_grounding(small_box), client)

    assert result.accepted is True
    assert result.bbox_plausible is True


# --- fail-closed on an unparseable VLM response --------------------------------------------


def test_validate_target_rejects_unparseable_vlm_response():
    client = FakeVLMClient("this is not json at all {{{")
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is False
    assert result.semantic_match is None
    assert result.bbox_plausible is True  # got past the pre-filter; the VLM call is what failed


def test_validate_target_accepts_json_wrapped_in_markdown_fences():
    wrapped = f"Sure, here you go:\n```json\n{_verification_json(True)}\n```\n"
    client = FakeVLMClient(wrapped)
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is True


# --- crop construction ----------------------------------------------------------------------


def test_validate_target_crops_with_margin_around_the_bbox():
    client = FakeVLMClient(_verification_json(True))
    box = BBoxPx(x0=50, y0=50, x1=100, y1=100, score=0.5)

    validate_target(make_image(), make_object_plan(), make_grounding(box), client)

    seen_w, seen_h = client.images[0].size  # PIL Image.size == (width, height)
    assert seen_w > box.x1 - box.x0
    assert seen_h > box.y1 - box.y0


# --- diagnostics -----------------------------------------------------------------------------


def test_validate_target_records_the_candidate_rank_it_was_given():
    client = FakeVLMClient(_verification_json(True))
    result = validate_target(
        make_image(), make_object_plan(), make_grounding(), client, candidate_rank=2
    )
    assert result.candidate_rank == 2


def test_validate_target_never_raises_on_reject():
    """A REJECT is a normal outcome, not an exception -- see the Phase 3.2 acceptance

    criterion ("a correct REJECT is a successful result").
    """
    client = FakeVLMClient(_verification_json(False))
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)
    assert result.accepted is False  # returned, not raised
