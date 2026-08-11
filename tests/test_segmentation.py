from __future__ import annotations

import numpy as np
import pytest

from manga_animation.pipeline.types import BBoxPx, GroundingResult, PipelineStageError
from manga_animation.segmentation.client import MaskCandidate
from manga_animation.segmentation.segment import segment_object


class FakeSegmentationClient:
    """A `SegmentationClient` double: canned mask candidates, no torch/transformers/network."""

    model_id = "fake-segmentation"

    def __init__(self, candidates: list[MaskCandidate]):
        self.candidates = candidates
        self.last_box: BBoxPx | None = None
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def segment(self, image, box: BBoxPx) -> list[MaskCandidate]:
        self.last_box = box
        return self.candidates

    def unload(self) -> None:
        self.unloaded = True


def make_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_mask(
    h: int = 64, w: int = 64, region: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """A full-image-shape uint8 0/255 mask, nonzero only inside `region` (x0,y0,x1,y1)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if region is not None:
        x0, y0, x1, y1 = region
        mask[y0:y1, x0:x1] = 255
    return mask


def make_grounding(bbox: BBoxPx | None = None) -> GroundingResult:
    resolved_bbox = bbox or BBoxPx(x0=10, y0=10, x1=40, y1=40)
    return GroundingResult(object_id="obj_1", bbox=resolved_bbox, model_id="fake-grounding")


# --- segment_object: box prompting ---------------------------------------------------


def test_segment_object_passes_the_grounding_box_to_the_client():
    grounding = make_grounding(BBoxPx(x0=5, y0=6, x1=20, y1=22))
    candidate = MaskCandidate(mask=make_mask(region=(5, 6, 20, 22)), iou_score=0.9)
    client = FakeSegmentationClient([candidate])

    segment_object(make_image(), grounding, client)

    assert client.last_box == grounding.bbox


# --- segment_object: best-IoU selection ----------------------------------------------


def test_segment_object_picks_the_highest_iou_candidate():
    low = MaskCandidate(mask=make_mask(region=(10, 10, 15, 15)), iou_score=0.4)
    high = MaskCandidate(mask=make_mask(region=(10, 10, 30, 30)), iou_score=0.95)
    client = FakeSegmentationClient([low, high])

    result = segment_object(make_image(), make_grounding(), client)

    assert result.iou_score == pytest.approx(0.95)
    np.testing.assert_array_equal(result.mask, high.mask)


# --- segment_object: mask shape/value convention --------------------------------------


def test_segment_object_returns_full_image_shape_uint8_0_255_mask():
    candidate = MaskCandidate(mask=make_mask(region=(10, 10, 30, 30)), iou_score=0.9)
    client = FakeSegmentationClient([candidate])
    result = segment_object(make_image(h=64, w=64), make_grounding(), client)

    assert result.mask.shape == (64, 64)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})


def test_segment_object_bbox_is_the_masks_tight_extent_not_the_grounding_box():
    # grounding box is (10,10,40,40) but the actual mask only covers (15,20,25,30) —
    # the returned bbox should reflect the mask's real extent, not the input box.
    grounding = make_grounding(BBoxPx(x0=10, y0=10, x1=40, y1=40))
    candidate = MaskCandidate(mask=make_mask(region=(15, 20, 25, 30)), iou_score=0.9)
    client = FakeSegmentationClient([candidate])

    result = segment_object(make_image(), grounding, client)

    assert result.bbox.as_xyxy() == (15, 20, 25, 30)


# --- segment_object: validation ---------------------------------------------------


def test_segment_object_raises_on_empty_mask():
    client = FakeSegmentationClient([MaskCandidate(mask=make_mask(), iou_score=0.9)])
    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), make_grounding(), client)
    assert exc_info.value.stage == "segmentation"


def test_segment_object_raises_on_full_page_mask():
    full = np.full((64, 64), 255, dtype=np.uint8)
    client = FakeSegmentationClient([MaskCandidate(mask=full, iou_score=0.9)])
    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), make_grounding(), client)
    assert exc_info.value.stage == "segmentation"


def test_segment_object_raises_when_client_returns_no_candidates():
    client = FakeSegmentationClient([])
    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), make_grounding(), client)
    assert exc_info.value.stage == "segmentation"
    assert exc_info.value.input_ref == "obj_1"


def test_segment_object_model_id_comes_from_the_client():
    candidate = MaskCandidate(mask=make_mask(region=(10, 10, 30, 30)), iou_score=0.9)
    client = FakeSegmentationClient([candidate])
    result = segment_object(make_image(), make_grounding(), client)
    assert result.model_id == "fake-segmentation"
