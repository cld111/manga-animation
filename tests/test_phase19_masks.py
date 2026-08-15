"""Phase 19 mask/coordinate transform tests: the expand2square padding geometry, the padded-
canvas -> original crop, and bbox/geometry helpers. All pure numpy/PIL."""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.benchmarking.phase19.masks import (
    SquarePad,
    expand2square_pad_color,
    mask_from_canvas,
    page_to_square_canvas,
    tight_bbox_from_mask,
    verify_mask_geometry,
)


def test_square_pad_equal_dimensions():
    pad = SquarePad.from_page_size((100, 100))
    assert pad.canvas_size == 100
    assert (pad.sx, pad.sy, pad.ex, pad.ey) == (0, 0, 100, 100)


def test_square_pad_width_greater_than_height():
    # page (H=60, W=100): canvas 100x100, page vertically centered at y=20.
    pad = SquarePad.from_page_size((60, 100))
    assert pad.canvas_size == 100
    assert (pad.sx, pad.sy) == (0, 20)
    assert (pad.ex, pad.ey) == (100, 80)
    assert (pad.ey - pad.sy, pad.ex - pad.sx) == (60, 100)


def test_square_pad_height_greater_than_width():
    pad = SquarePad.from_page_size((100, 60))
    assert pad.canvas_size == 100
    assert (pad.sx, pad.sy) == (20, 0)
    assert (pad.ex, pad.ey) == (80, 100)


def test_expand2square_pad_color_matches_official_mean():
    assert expand2square_pad_color((0.4814, 0.4578, 0.4082)) == (122, 116, 104)


def test_page_to_square_canvas_places_page_centered():
    image = np.full((60, 100, 3), 255, dtype=np.uint8)
    pad = SquarePad.from_page_size((60, 100))
    canvas = page_to_square_canvas(image, pad, fill=(0, 0, 0))
    assert canvas.shape == (100, 100, 3)
    assert np.all(canvas[0:20, :, :] == 0)  # top padding
    assert np.all(canvas[20:80, :, :] == 255)  # the page band
    assert np.all(canvas[80:100, :, :] == 0)  # bottom padding


def test_page_to_square_canvas_geometry_mismatch_raises():
    image = np.full((10, 10, 3), 0, dtype=np.uint8)
    pad = SquarePad.from_page_size((60, 100))
    with pytest.raises(ValueError):
        page_to_square_canvas(image, pad)


def test_mask_from_canvas_crops_padding():
    canvas = np.zeros((100, 100), dtype=bool)
    canvas[20:80, 0:100] = True  # the page band for (60, 100)
    pad = SquarePad.from_page_size((60, 100))
    cropped = mask_from_canvas(canvas, pad)
    assert cropped.shape == (60, 100)
    assert np.all(cropped)


def test_mask_from_canvas_roundtrip():
    # A mask on original geometry survives pad -> crop exactly.
    orig = np.zeros((60, 100), dtype=bool)
    orig[10:50, 20:90] = True
    pad = SquarePad.from_page_size((60, 100))
    canvas = np.zeros((100, 100), dtype=bool)
    canvas[pad.sy : pad.sy + 60, pad.sx : pad.sx + 100] = orig
    assert np.array_equal(mask_from_canvas(canvas, pad), orig)


def test_mask_from_canvas_wrong_size_raises():
    pad = SquarePad.from_page_size((60, 100))
    with pytest.raises(ValueError):
        mask_from_canvas(np.zeros((50, 50), dtype=bool), pad)


def test_tight_bbox():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 40:70] = True
    assert tight_bbox_from_mask(mask) == (40, 10, 70, 30)


def test_tight_bbox_empty_raises():
    with pytest.raises(ValueError):
        tight_bbox_from_mask(np.zeros((10, 10), dtype=bool))


def test_verify_mask_geometry():
    assert verify_mask_geometry(np.zeros((60, 100), dtype=bool), (60, 100))
    assert verify_mask_geometry(np.ones((60, 100), dtype=np.uint8) * 255, (60, 100))
    assert not verify_mask_geometry(np.zeros((50, 100), dtype=bool), (60, 100))
    assert not verify_mask_geometry(np.zeros((60, 100, 3), dtype=bool), (60, 100))
