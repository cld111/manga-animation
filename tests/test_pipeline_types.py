"""Tests for src/manga_animation/pipeline/types.py -- specifically the Phase 3.3 additions:

`PanelCandidate`'s invariants and the `BBoxPx <-> BBox` coordinate-mapping utilities
(`normalized_bbox_to_px` / `bbox_px_to_normalized`). See
docs/decisions/0007-panel-aware-analysis.md for why this mapping is a "critical, deterministic,
tested" concern for panel-aware analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.pipeline.types import (
    BBoxPx,
    PanelCandidate,
    bbox_px_to_normalized,
    normalized_bbox_to_px,
)
from manga_animation.schemas.animation_plan import BBox

# --- normalized_bbox_to_px -------------------------------------------------------------------


def test_normalized_bbox_to_px_full_page():
    bbox = BBox(x=0.0, y=0.0, width=1.0, height=1.0)
    px = normalized_bbox_to_px(bbox, page_width=800, page_height=2000)
    assert px.as_xyxy() == (0, 0, 800, 2000)


def test_normalized_bbox_to_px_sub_region_reaches_expected_pixels():
    bbox = BBox(x=0.25, y=0.5, width=0.5, height=0.25)
    px = normalized_bbox_to_px(bbox, page_width=1000, page_height=1000)
    assert px.as_xyxy() == (250, 500, 750, 750)


def test_normalized_bbox_to_px_far_edge_uses_ceil_not_truncation():
    """A normalized box that should reach the page's far edge must not lose a row/column to

    `int()` truncation -- e.g. x=0.1 + width=0.9 on a 3px-wide page should reach x=3, not 2.
    """
    bbox = BBox(x=0.1, y=0.1, width=0.9, height=0.9)
    px = normalized_bbox_to_px(bbox, page_width=3, page_height=3)
    assert px.x1 == 3
    assert px.y1 == 3


def test_normalized_bbox_to_px_clamps_to_page_bounds():
    bbox = BBox(x=0.99, y=0.99, width=0.01, height=0.01)
    px = normalized_bbox_to_px(bbox, page_width=10, page_height=10)
    assert px.x1 <= 10
    assert px.y1 <= 10


# --- bbox_px_to_normalized -------------------------------------------------------------------


def test_bbox_px_to_normalized_full_page():
    px = BBoxPx(x0=0, y0=0, x1=800, y1=2000)
    bbox = bbox_px_to_normalized(px, page_width=800, page_height=2000)
    assert bbox.x == pytest.approx(0.0)
    assert bbox.y == pytest.approx(0.0)
    assert bbox.width == pytest.approx(1.0)
    assert bbox.height == pytest.approx(1.0)


def test_bbox_px_to_normalized_sub_region():
    px = BBoxPx(x0=250, y0=500, x1=750, y1=750)
    bbox = bbox_px_to_normalized(px, page_width=1000, page_height=1000)
    assert bbox.x == pytest.approx(0.25)
    assert bbox.y == pytest.approx(0.5)
    assert bbox.width == pytest.approx(0.5)
    assert bbox.height == pytest.approx(0.25)


def test_bbox_px_to_normalized_never_exceeds_schema_bounds_at_page_edge():
    """A pixel box that touches the page's far edge must normalize to something the schema's

    own BBox bounds validator accepts (x + width <= 1, not 1.0000000002 from float division).
    """
    px = BBoxPx(x0=333, y0=777, x1=1000, y1=1000)
    bbox = bbox_px_to_normalized(px, page_width=1000, page_height=1000)  # constructs BBox itself
    assert bbox.x + bbox.width <= 1.0 + 1e-9
    assert bbox.y + bbox.height <= 1.0 + 1e-9


# --- round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "px_box,page_w,page_h",
    [
        ((0, 0, 800, 2000), 800, 2000),
        ((100, 200, 700, 1900), 800, 2000),
        ((0, 0, 720, 5062), 720, 5062),
        ((9, 3360, 716, 4912), 720, 5062),
        ((1, 1, 2, 2), 10, 10),
    ],
)
def test_px_to_normalized_to_px_round_trip_within_one_pixel(px_box, page_w, page_h):
    original = BBoxPx(x0=px_box[0], y0=px_box[1], x1=px_box[2], y1=px_box[3])
    normalized = bbox_px_to_normalized(original, page_width=page_w, page_height=page_h)
    back = normalized_bbox_to_px(normalized, page_width=page_w, page_height=page_h)

    assert abs(back.x0 - original.x0) <= 1
    assert abs(back.y0 - original.y0) <= 1
    assert abs(back.x1 - original.x1) <= 1
    assert abs(back.y1 - original.y1) <= 1


def test_normalized_to_px_to_normalized_round_trip_is_stable():
    original = BBox(x=0.1234, y=0.5678, width=0.25, height=0.1)
    px = normalized_bbox_to_px(original, page_width=4000, page_height=6000)
    back = bbox_px_to_normalized(px, page_width=4000, page_height=6000)

    assert back.x == pytest.approx(original.x, abs=1e-3)
    assert back.y == pytest.approx(original.y, abs=1e-3)
    assert back.width == pytest.approx(original.width, abs=1e-3)
    assert back.height == pytest.approx(original.height, abs=1e-3)


# --- PanelCandidate invariants ----------------------------------------------------------------


def test_panel_candidate_accepts_matching_crop_and_bbox():
    bbox = BBoxPx(x0=10, y0=20, x1=60, y1=120)
    crop = np.zeros((100, 50, 3), dtype=np.uint8)  # (height, width, 3) matching bbox extent
    candidate = PanelCandidate(
        id="panel_0", bbox=bbox, crop=crop, confidence=0.9, source="gutter_xy_cut"
    )
    assert candidate.crop.shape[:2] == (bbox.height, bbox.width)
    assert candidate.metadata == {}


def test_panel_candidate_rejects_crop_shape_mismatch():
    bbox = BBoxPx(x0=0, y0=0, x1=50, y1=100)
    wrong_crop = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="crop shape"):
        PanelCandidate(
            id="panel_0", bbox=bbox, crop=wrong_crop, confidence=0.9, source="gutter_xy_cut"
        )


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_panel_candidate_rejects_confidence_out_of_range(bad_confidence):
    bbox = BBoxPx(x0=0, y0=0, x1=10, y1=10)
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="confidence"):
        PanelCandidate(
            id="panel_0", bbox=bbox, crop=crop, confidence=bad_confidence, source="gutter_xy_cut"
        )
