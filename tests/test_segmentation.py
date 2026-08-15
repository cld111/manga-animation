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
    """A full-image-shape uint8 0/255 mask: a diamond inscribed in `region` (x0,y0,x1,y1),

    nonzero only inside it, touching each of its 4 edges at exactly its midpoint -- same
    tight-bbox-equals-`region` property a solid rectangle fill has, but without a solid
    rectangle's 100%-of-every-edge touch fraction, which the real, evidenced
    `segmentation/segment.py::_validate_mask_shape` check (Phase 8.3, see
    docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md) now correctly rejects as
    implausible for a real object silhouette.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if region is not None:
        x0, y0, x1, y1 = region
        cx, cy = (x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0
        ax, ay = max((x1 - x0) / 2.0, 1e-9), max((y1 - y0) / 2.0, 1e-9)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local = (np.abs(xx - cx) / ax + np.abs(yy - cy) / ay) <= 1.0
        mask[y0:y1, x0:x1] = local.astype(np.uint8) * 255
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


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((32, 64), dtype=np.uint8),
        np.zeros((64, 64, 1), dtype=np.uint8),
        np.zeros((64, 64), dtype=np.float32),
    ],
)
def test_segment_object_rejects_masks_that_violate_source_geometry_or_dtype(mask):
    mask[10, 10] = 255
    client = FakeSegmentationClient([MaskCandidate(mask=mask, iou_score=0.9)])
    with pytest.raises(PipelineStageError, match="mask"):
        segment_object(make_image(), make_grounding(), client)


def test_segment_object_raises_on_full_page_mask():
    full = np.full((64, 64), 255, dtype=np.uint8)
    client = FakeSegmentationClient([MaskCandidate(mask=full, iou_score=0.9)])
    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), make_grounding(), client)
    assert exc_info.value.stage == "segmentation"


def _one_sided_mask(h: int, w: int, region: tuple[int, int, int, int]) -> np.ndarray:
    """A mask matching the real Defect B evidence exactly: a diamond silhouette (see `make_mask`)

    UNIONED with a solid strip along the region's left third -- the left edge is hugged for its
    full length while the opposite (right) edge stays low, same asymmetric signature the real
    downloaded SAM mask had (LEFT=45.5%, RIGHT=0.56%, see
    docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md).
    """
    mask = make_mask(h, w, region=region)
    x0, y0, x1, y1 = region
    strip_x1 = x0 + max(1, (x1 - x0) // 3)
    mask[y0:y1, x0:strip_x1] = 255
    return mask


def test_segment_object_raises_on_a_mask_that_hugs_one_bbox_edge_but_not_the_opposite_one():
    """Phase 8.3, Defect B regression: `phase3_action_page`'s real "vertical seam" defect was

    traced to a real SAM 2.1 mask (downloaded from a live Kaggle run and independently
    re-verified by reproducing it through the real production compositing/animation code) whose
    own tight bbox's LEFT edge was mask-covered for 45.5% of its height while the OPPOSITE
    (right) edge was only 0.56% -- an over-segmentation into adjacent background on just one
    side, invisible while the object sits at rest but producing a hard, duplicate-looking seam
    once TRANSLATE displaces it. Five OTHER real masks from the same investigation (a sword, two
    eyes, two hair regions with no visual defect) ranged 2.2%-20.2% on their own worst edge. See
    docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md for the full evidence and for why
    the check requires this LEFT-high/RIGHT-low asymmetry rather than flagging any one edge
    alone (which would also flag a genuinely rectangular object, e.g. a real banner/flag).
    """
    region = (10, 10, 40, 40)
    x0, y0, x1, y1 = region
    candidate = MaskCandidate(mask=_one_sided_mask(64, 64, region), iou_score=0.9)
    client = FakeSegmentationClient([candidate])

    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), make_grounding(BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1)), client)
    assert exc_info.value.stage == "segmentation"
    assert "hugs" in exc_info.value.detail
    assert "left" in exc_info.value.detail


def test_segment_object_accepts_a_mask_that_only_touches_bbox_edges_near_their_midpoint():
    """Negative control for the check above: a real object silhouette's tight bbox is touched

    by the mask only near each edge's midpoint (an extremal point of the silhouette), not along
    a long run -- this must not be rejected.
    """
    region = (10, 10, 40, 40)
    candidate = MaskCandidate(mask=make_mask(region=region), iou_score=0.9)
    client = FakeSegmentationClient([candidate])
    x0, y0, x1, y1 = region

    grounding = make_grounding(BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1))
    result = segment_object(make_image(), grounding, client)

    assert result.bbox.as_xyxy() == region


def test_segment_object_accepts_a_genuinely_rectangular_mask():
    """Negative control found by independent review: a solid rectangle (BOTH opposite edges of

    an axis hugged together, e.g. a real banner/flag/sign with a genuinely straight silhouette
    -- explicitly a valid target per this project's own dataset, `configs/
    phase3_3_eval_dataset.yaml`'s `phase3_action_page`/`eval_weapon_effects` acceptable_outcome
    entries naming "cloth-banner-shaped region"/"energy-effect-shaped region") must NOT be
    rejected -- only a ONE-sided hug (see the test above) is the real, evidenced defect shape.
    """
    region = (10, 10, 40, 40)
    x0, y0, x1, y1 = region
    solid_rectangle = np.zeros((64, 64), dtype=np.uint8)
    solid_rectangle[y0:y1, x0:x1] = 255
    client = FakeSegmentationClient([MaskCandidate(mask=solid_rectangle, iou_score=0.9)])

    grounding = make_grounding(BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1))
    result = segment_object(make_image(), grounding, client)

    assert result.bbox.as_xyxy() == region


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


def test_segment_object_accepts_sparse_effect_mask_under_density_bound():
    """Phase 16: a SPARSE drawn-effect mask (thin rays radiating from a center -- the real
    signature of an impact burst / speed lines) inside a large bbox must pass
    `max_mask_density`: the density check exists to reject dense "select everything" masks,
    not sparse effect artwork. Here the mask is 8 thin rays in a 40x40 box, so its tight-bbox
    density is far under the 0.70 bound even though the box itself is large."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    cx = cy = 30
    for angle in range(0, 180, 22):  # 8 rays from the center
        import math

        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for t in range(1, 16):
            mask[cy + int(round(dy * t)), cx + int(round(dx * t))] = 255
            mask[cy - int(round(dy * t)), cx - int(round(dx * t))] = 255
    assert (mask[10:50, 10:50] > 0).sum() / (40 * 40) < 0.70  # sanity: genuinely sparse
    client = FakeSegmentationClient([MaskCandidate(mask=mask, iou_score=0.9)])
    grounding = make_grounding(BBoxPx(x0=10, y0=10, x1=50, y1=50))

    result = segment_object(make_image(), grounding, client, max_mask_density=0.70)

    assert result.mask is not None


def test_segment_object_rejects_dense_filled_effect_mask_over_density_bound():
    """Phase 16: a DENSE filled mask -- the confirmed "select everything" signature (real
    defective cloth_5 at density 0.902, character_hair_7 at 0.843, docs/phase11-results.md)
    -- must be REJECTED by `max_mask_density` when a RADIAL_EXPAND effect expects sparse
    artwork: animating a nearly-solid mask would drag a large filled region (including panel
    background) when the rim breathes."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:48, 12:48] = 255  # 36x36 solid fill: density = 1296/1600 = 0.81 > 0.70
    client = FakeSegmentationClient([MaskCandidate(mask=mask, iou_score=0.9)])
    grounding = make_grounding(BBoxPx(x0=10, y0=10, x1=50, y1=50))

    with pytest.raises(PipelineStageError) as exc_info:
        segment_object(make_image(), grounding, client, max_mask_density=0.70)
    assert exc_info.value.stage == "segmentation"
    assert "density" in exc_info.value.detail.lower()


def test_segment_object_without_density_bound_accepts_dense_mask():
    """Phase 16: `max_mask_density` is opt-in -- an ordinary (non-effect) object's dense mask
    must pass exactly as before when no bound is given (the density check is the drawn-effect
    safety net, not a general segmentation restriction)."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:48, 12:48] = 255
    client = FakeSegmentationClient([MaskCandidate(mask=mask, iou_score=0.9)])

    result = segment_object(
        make_image(), make_grounding(BBoxPx(x0=10, y0=10, x1=50, y1=50)), client
    )

    assert result.mask is not None
