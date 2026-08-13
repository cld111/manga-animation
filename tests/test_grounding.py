from __future__ import annotations

import numpy as np
import pytest

from manga_animation.grounding.client import Detection, _detections_from_scores_boxes_labels
from manga_animation.grounding.ground import (
    _prompt_from_label,
    ground_object,
    ground_object_candidates,
)
from manga_animation.pipeline.types import BBoxPx, PipelineStageError
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


# --- _detections_from_scores_boxes_labels (real Phase 3.2 finding) -------------------------


def test_detections_from_scores_boxes_labels_normal_aligned_case():
    detections = _detections_from_scores_boxes_labels(
        scores=[0.9, 0.5],
        boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
        text_labels=["hair", "face"],
        fallback_label="fallback",
    )
    assert [d.label for d in detections] == ["hair", "face"]
    assert [d.score for d in detections] == pytest.approx([0.9, 0.5])
    assert detections[0].box == (1, 2, 3, 4)


def test_detections_from_scores_boxes_labels_handles_zero_detections_with_placeholder_label():
    """Real, reproduced Phase 3.2 finding: a zero-detection `post_process_grounded_object_

    detection` result can still return `text_labels=['']` (length 1) while `scores`/`boxes`
    are correctly length 0 -- must not raise, must return an empty list.
    """
    detections = _detections_from_scores_boxes_labels(
        scores=[], boxes=[], text_labels=[""], fallback_label="weapon"
    )
    assert detections == []


def test_detections_from_scores_boxes_labels_falls_back_when_labels_run_short():
    detections = _detections_from_scores_boxes_labels(
        scores=[0.9, 0.5],
        boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
        text_labels=["hair"],  # shorter than scores/boxes
        fallback_label="fallback_prompt",
    )
    assert [d.label for d in detections] == ["hair", "fallback_prompt"]


def test_detections_from_scores_boxes_labels_still_raises_when_scores_and_boxes_disagree():
    """scores/boxes mismatching each other (unlike a labels mismatch) is not a known real

    case -- keep failing loudly on it rather than silently truncating.
    """
    with pytest.raises(ValueError, match="zip"):
        _detections_from_scores_boxes_labels(
            scores=[0.9, 0.5], boxes=[[1, 2, 3, 4]], text_labels=["hair"], fallback_label="x"
        )


def test_ground_object_delegates_to_candidates_and_returns_the_top_one():
    detections = [
        Detection(label="hair", score=0.31, box=(5, 5, 20, 20)),
        Detection(label="hair", score=0.82, box=(10, 10, 60, 60)),
    ]
    client = FakeGroundingClient(detections)

    result = ground_object(make_image(), make_object_plan(), client)

    assert result.bbox.score == pytest.approx(0.82)


# --- Phase 5.1: panel-aware grounding (docs/decisions/0011-panel-aware-grounding.md) --------


class SpyGroundingClient:
    """Like `FakeGroundingClient`, but also records the actual image array it was handed --

    lets a test assert exactly what region (full page or panel crop) grounding ran against,
    not just what candidates came back. `detections` are given in CROP-LOCAL coordinates (i.e.
    relative to whatever image `detect()` actually receives), matching the real
    `GroundingDinoClient.detect`'s contract of returning boxes relative to its input.
    """

    model_id = "spy-grounding"

    def __init__(self, detections: list[Detection]):
        self.detections = detections
        self.last_prompt: str | None = None
        self.last_image_shape: tuple[int, ...] | None = None

    def load(self) -> None:
        pass

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.last_prompt = text_prompt
        self.last_image_shape = image.shape
        return self.detections

    def unload(self) -> None:
        pass


def test_ground_object_candidates_crops_to_the_given_panel_bbox_px():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=20, y0=30, x1=80, y1=90)  # 60x60 region, offset (20, 30)

    ground_object_candidates(
        make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
    )

    assert client.last_image_shape == (60, 60, 3)


def test_ground_object_candidates_translates_local_box_to_page_coordinates():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=20, y0=30, x1=80, y1=90)

    candidates = ground_object_candidates(
        make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
    )

    assert candidates[0].bbox.as_xyxy() == (25, 35, 35, 45)


def test_ground_object_candidates_clips_translated_box_to_the_page_not_just_the_crop():
    """A crop flush against the page's own edge, with a local box that overshoots the crop's

    own bounds (a real, observed Grounding DINO behavior near resized-image edges) -- the
    translated box must be clipped against the FULL PAGE, exactly like the pre-Phase-5.1
    full-page case already had to handle, not silently left oversized.
    """
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(40, 40, 70, 70))])
    panel = BBoxPx(x0=150, y0=150, x1=200, y1=200)  # 50x50 crop flush against the page edge

    candidates = ground_object_candidates(
        make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
    )

    # local (40,40)-(70,70) + offset (150,150) = (190,190)-(220,220), clipped to the 200x200 page
    assert candidates[0].bbox.as_xyxy() == (190, 190, 200, 200)


def test_ground_object_candidates_with_panel_bbox_px_none_grounds_the_full_page_unchanged():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    page = make_image(h=100, w=120)

    candidates = ground_object_candidates(page, make_object_plan(), client, panel_bbox_px=None)

    assert client.last_image_shape == page.shape
    assert candidates[0].bbox.as_xyxy() == (5, 5, 15, 15)  # offset (0, 0) -- unchanged


def test_ground_object_candidates_full_page_panel_bbox_is_equivalent_to_none():
    """A `panel_bbox_px` that already covers the whole page (page-mode's synthetic (0,0,1,1)

    panel, or panel-detection's `fallback_full_page` candidate -- see ADR 0011's "Fallback
    behavior") must produce byte-identical results to omitting `panel_bbox_px` entirely.
    """
    page = make_image(h=100, w=120)
    detections = [Detection(label="hair", score=0.9, box=(5, 5, 15, 15))]

    client_none = SpyGroundingClient(list(detections))
    result_none = ground_object_candidates(
        page, make_object_plan(), client_none, panel_bbox_px=None
    )

    client_full = SpyGroundingClient(list(detections))
    full_page_region = BBoxPx(x0=0, y0=0, x1=120, y1=100)
    result_full = ground_object_candidates(
        page, make_object_plan(), client_full, panel_bbox_px=full_page_region
    )

    assert result_none[0].bbox.as_xyxy() == result_full[0].bbox.as_xyxy()
    assert client_none.last_image_shape == client_full.last_image_shape


def test_ground_object_passes_panel_bbox_px_through_to_candidates():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=10, y0=10, x1=50, y1=50)

    result = ground_object(
        make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
    )

    assert result.bbox.as_xyxy() == (15, 15, 25, 25)


def test_ground_object_candidates_raises_with_the_region_in_the_error_detail():
    client = SpyGroundingClient([])
    panel = BBoxPx(x0=10, y0=10, x1=50, y1=50)

    with pytest.raises(PipelineStageError) as exc_info:
        ground_object_candidates(
            make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
        )

    assert exc_info.value.stage == "grounding"
    assert "(10, 10, 50, 50)" in exc_info.value.detail


# --- Phase 7.1.4: defensive panel_bbox_px/image consistency check ---------------------------
#
# ADR 0011's "Known limitations" flagged this directly: every real call site derives
# panel_bbox_px and image from the same page_shape, so this is structurally unreachable in
# production today -- but ground_object_candidates had no defensive check for it, only numpy's
# silent out-of-range slice truncation. These tests protect the fix, not a redesign.


def test_ground_object_candidates_raises_on_panel_bbox_wider_than_image():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=0, y0=0, x1=250, y1=100)  # x1 exceeds the 200-wide image

    with pytest.raises(ValueError, match="inconsistent with the actual image dimensions"):
        ground_object_candidates(
            make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
        )


def test_ground_object_candidates_raises_on_panel_bbox_taller_than_image():
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=0, y0=0, x1=100, y1=250)  # y1 exceeds the 200-tall image

    with pytest.raises(ValueError, match="inconsistent with the actual image dimensions"):
        ground_object_candidates(
            make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
        )


def test_ground_object_candidates_boundary_exact_panel_bbox_stays_valid():
    """The ordinary, real, full-page-equivalent case (panel_bbox_px exactly matching the
    image's own bounds, per ADR 0011's fallback_full_page/page-mode synthetic panel) must NOT
    be rejected by the new defensive check -- only a genuinely out-of-range box should raise.
    """
    client = SpyGroundingClient([Detection(label="hair", score=0.9, box=(5, 5, 15, 15))])
    panel = BBoxPx(x0=0, y0=0, x1=200, y1=200)  # exactly the image's own bounds

    candidates = ground_object_candidates(
        make_image(h=200, w=200), make_object_plan(), client, panel_bbox_px=panel
    )
    assert candidates[0].bbox.as_xyxy() == (5, 5, 15, 15)
