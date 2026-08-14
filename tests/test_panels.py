"""Tests for src/manga_animation/analysis/panels.py -- the deterministic, model-free gutter-

based panel detector (see docs/decisions/0007-panel-aware-analysis.md). All synthetic: pure
numpy image construction, no real sample pages or model calls needed to exercise the geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.analysis.panels import detect_panels


def _blank_page(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)):
    page = np.zeros((height, width, 3), dtype=np.uint8)
    page[:, :] = color
    return page


def _rng_noise_block(height: int, width: int, seed: int) -> np.ndarray:
    """A non-uniform (high local variance) block, so it never reads as a gutter."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _fill(page: np.ndarray, x0: int, y0: int, x1: int, y1: int, seed: int):
    """Fill a region with textured (internally varying) content, seeded for determinism.

    Deliberately NOT a flat solid color: real manga art always has internal variation along a
    row/column (linework, shading, texture), which is exactly what lets the row/column
    standard-deviation gutter test tell "real panel content" apart from "blank gutter" (both a
    flat color block and a blank gutter would otherwise read as equally "uniform" -- see
    `_MAX_GUTTER_RUN_FRACTION`'s docstring in `analysis/panels.py` for this same, real,
    documented limitation of a pure-variance signal).
    """
    page[y0:y1, x0:x1] = _rng_noise_block(y1 - y0, x1 - x0, seed)


# --- zero panels ------------------------------------------------------------------------------


def test_too_small_image_returns_zero_panels():
    tiny = _blank_page(8, 8)
    assert detect_panels(tiny) == []


def test_degenerate_one_pixel_tall_image_returns_zero_panels():
    tiny = _blank_page(20, 1)
    assert detect_panels(tiny) == []


# --- one panel (fallback: no internal gutters) ------------------------------------------------


def test_page_with_no_internal_gutters_returns_one_fallback_panel():
    page = _rng_noise_block(400, 300, seed=1)  # one continuous "busy" panel, no gutter anywhere
    panels = detect_panels(page)
    assert len(panels) == 1
    assert panels[0].source == "fallback_full_page"
    assert panels[0].bbox.as_xyxy() == (0, 0, 300, 400)
    assert panels[0].crop.shape == (400, 300, 3)
    assert 0.0 <= panels[0].confidence <= 1.0


def test_solid_uniform_page_returns_one_fallback_panel():
    """An entirely blank page has no internal gutter *boundary* to split on (uniform

    everywhere) -- still one valid whole-page candidate, not a crash or empty result.
    """
    page = _blank_page(300, 500)
    panels = detect_panels(page)
    assert len(panels) == 1
    assert panels[0].source == "fallback_full_page"


# --- multiple panels (real gutter split) -------------------------------------------------------


def test_two_panels_stacked_vertically_are_split_at_the_gutter():
    page = _blank_page(300, 800)  # white background = gutter
    _fill(page, 0, 0, 300, 300, 1)  # panel A: rows 0-300
    _fill(page, 0, 400, 300, 800, 2)  # panel B: rows 400-800 (100px white gutter)

    panels = detect_panels(page)

    assert len(panels) == 2
    assert all(p.source == "gutter_xy_cut" for p in panels)
    # sorted top-to-bottom
    assert panels[0].bbox.y0 < panels[1].bbox.y0
    # each candidate's crop matches its own bbox exactly
    for p in panels:
        assert p.crop.shape[:2] == (p.bbox.height, p.bbox.width)


def test_three_panels_stacked_vertically():
    page = _blank_page(300, 1200)
    _fill(page, 0, 0, 300, 300, 3)
    _fill(page, 0, 400, 300, 700, 4)
    _fill(page, 0, 800, 300, 1200, 5)

    panels = detect_panels(page)

    assert len(panels) == 3
    ys = [p.bbox.y0 for p in panels]
    assert ys == sorted(ys)


def test_side_by_side_panels_within_a_tall_band_are_column_split():
    page = _blank_page(600, 500)
    _fill(page, 0, 0, 280, 500, 6)  # left column
    _fill(page, 320, 0, 600, 500, 7)  # right column (40px white gutter)

    panels = detect_panels(page)

    assert len(panels) == 2
    xs = sorted(p.bbox.x0 for p in panels)
    assert xs[0] < 100  # left panel starts near the page's left edge
    assert xs[1] > 280  # right panel starts at/past the gutter (context margin may pull it
    # back slightly toward the gutter's own midpoint, but never back into the left panel)


def test_panels_are_returned_in_top_to_bottom_left_to_right_reading_order():
    page = _blank_page(600, 900)
    _fill(page, 0, 0, 280, 300, 8)  # band 1, col 0
    _fill(page, 320, 0, 600, 300, 9)  # band 1, col 1
    _fill(page, 0, 500, 600, 900, 10)  # band 2 (full width)

    panels = detect_panels(page)

    assert len(panels) == 3
    positions = [(p.bbox.y0, p.bbox.x0) for p in panels]
    assert positions == sorted(positions)


# --- tall pages ---------------------------------------------------------------------------------


def test_very_tall_page_splits_into_proportionate_bands():
    """Mirrors the real Phase 3.1 720x5062 (~7:1) page shape -- a tall page with several

    stacked panels must split into real bands, not collapse into one page-spanning candidate
    (which is exactly the problem panel-aware analysis exists to fix, see ADR 0007).
    """
    page = _blank_page(720, 5000)
    bands = [(0, 600), (700, 2400), (2500, 3400), (3500, 4900)]
    for i, (y0, y1) in enumerate(bands):
        _fill(page, 0, y0, 720, y1, 100 + i)

    panels = detect_panels(page)

    assert len(panels) == len(bands)
    for p in panels:
        assert p.bbox.height < 5000  # no candidate collapsed back to the whole page
        assert p.confidence > 0.0


# --- malformed / degenerate candidate geometry never reaches the caller ------------------------


def test_all_candidates_below_minimum_area_fall_back_to_whole_page():
    """Tiny slivers (e.g. a few stray dark pixels creating spurious thin gutters) must not be

    reported as real panels -- the area floor should reject them all and fall back cleanly.
    """
    page = _blank_page(300, 300)
    # a few 1px-tall dark rows -- too thin to be real gutters/panels either way
    for y in (50, 51, 150, 151, 250, 251):
        page[y, :] = (0, 0, 0)

    panels = detect_panels(page)

    assert len(panels) >= 1
    for p in panels:
        area_fraction = (p.bbox.width * p.bbox.height) / (300 * 300)
        assert area_fraction >= 0.01 or p.source == "fallback_full_page"


def test_every_candidate_bbox_is_well_formed_and_within_page_bounds():
    page = _blank_page(400, 1000)
    _fill(page, 0, 0, 400, 300, 12)
    _fill(page, 0, 400, 400, 1000, 13)

    panels = detect_panels(page)

    assert len(panels) >= 1
    for p in panels:
        assert 0 <= p.bbox.x0 < p.bbox.x1 <= 400
        assert 0 <= p.bbox.y0 < p.bbox.y1 <= 1000
        assert p.crop.shape[:2] == (p.bbox.height, p.bbox.width)


# --- confidence reflects real evidence strength ------------------------------------------------


def test_thicker_more_uniform_gutter_yields_higher_confidence_than_a_borderline_one():
    thick_gutter_page = _blank_page(300, 1000)
    _fill(thick_gutter_page, 0, 0, 300, 300, 14)
    _fill(thick_gutter_page, 0, 700, 300, 1000, 15)  # 400px clean white gutter

    thin_gutter_page = _blank_page(300, 620)
    _fill(thin_gutter_page, 0, 0, 300, 300, 16)
    _fill(thin_gutter_page, 0, 310, 300, 620, 17)  # ~10px gutter, just above the floor

    thick_panels = detect_panels(thick_gutter_page)
    thin_panels = detect_panels(thin_gutter_page)

    assert len(thick_panels) == 2
    assert len(thin_panels) == 2
    assert min(p.confidence for p in thick_panels) >= min(p.confidence for p in thin_panels)


# --- determinism ---------------------------------------------------------------------------------


def test_detection_is_deterministic_across_repeated_calls():
    page = _blank_page(300, 900)
    _fill(page, 0, 0, 300, 300, 18)
    _fill(page, 0, 500, 300, 900, 19)

    first = detect_panels(page)
    second = detect_panels(page)

    assert [p.bbox.as_xyxy() for p in first] == [p.bbox.as_xyxy() for p in second]
    assert [p.confidence for p in first] == [p.confidence for p in second]
    assert [p.source for p in first] == [p.source for p in second]
    assert [p.id for p in first] == [p.id for p in second]


@pytest.mark.parametrize("width,height", [(720, 5062), (800, 2305), (800, 2216)])
def test_detector_runs_on_real_sample_page_dimensions_without_crashing(width, height):
    """Uses synthetic content at the real dimensions of this project's real sample pages

    (see examples/) -- proves the detector handles these exact aspect ratios/sizes cleanly
    without needing the real (git-ignored, not-committed) image bytes in the test suite.
    """
    page = _rng_noise_block(height, width, seed=7)
    panels = detect_panels(page)
    assert len(panels) >= 1
    for p in panels:
        assert p.bbox.x1 <= width
        assert p.bbox.y1 <= height
