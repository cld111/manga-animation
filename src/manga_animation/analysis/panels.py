"""Deterministic, model-free panel detection: full page -> `PanelCandidate` regions.

See `docs/decisions/0007-panel-aware-analysis.md` for why this is a classical recursive
gutter-based ("XY-cut") splitter rather than a learned detector: every real sample page this
project has ever used is a digital, full-color, single-column webtoon-style page with wide,
near-uniform-color gutters between panels -- a solved, well-understood segmentation problem
that doesn't need model weights, GPU residency, or a training/calibration dataset this project
doesn't have.

Nothing here imports torch/transformers -- this stays a pure numpy/opencv module, importable
without the `ml` extra (mirrors `analysis/client.py`'s `VLMClient`-only import boundary).

Only `detect_panels` is the public entry point; everything else is an implementation detail of
turning row/column uniformity profiles into `PanelCandidate`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from manga_animation.pipeline.types import BBoxPx, ImageArray, PanelCandidate

# --- tuning constants -------------------------------------------------------------------------
# All of these are documented, evidenced-but-not-statistically-calibrated choices (see ADR
# 0007's "Open questions") -- consistent with every other uncalibrated-but-documented threshold
# already in this codebase (pipeline/types.py's coverage-fraction bounds,
# validation/validate.py's margin fraction).

_GUTTER_STD_THRESHOLD = 8.0
"""A row/column is "uniform" (part of a gutter) if its pixel standard deviation is below this,
regardless of its actual color -- catches white gutters (the real, observed case on every
sample page this project has) and would equally catch a solid-black gutter, without hardcoding
a specific background color."""

_MIN_GUTTER_FRACTION = 0.006
"""Minimum contiguous run of uniform rows/columns, as a fraction of the page's relevant
dimension, to count as a real gutter rather than a thin highlight/antialiasing artifact."""

_MIN_GUTTER_PX = 6
"""Absolute floor under `_MIN_GUTTER_FRACTION` so a small image still requires a few real
pixels of uniformity, not a fraction that rounds to nothing."""

_MIN_PANEL_AREA_FRACTION = 0.01
"""A candidate region smaller than this fraction of the full page's area is dropped -- almost
certainly a sliver produced by a false-positive gutter, not a real panel."""

_CONTEXT_MARGIN_FRACTION = 0.02
"""Context padding added around a gutter-derived tight region before it becomes the panel's
final `bbox`, as a fraction of that region's own dimension -- mirrors
`validation/validate.py`'s `_MARGIN_FRACTION` crop-padding pattern (image prep, not a
pass/fail threshold)."""

_MIN_BAND_HEIGHT_FOR_COLUMN_SPLIT_PX = 40
"""A row band shorter than this is not considered for a further column (side-by-side) split --
too little vertical extent for a second real panel to plausibly fit beside the first."""

_MIN_IMAGE_DIM_PX = 16
"""Below this in either dimension, the image is too small for gutter analysis to mean anything
-- `detect_panels` returns zero candidates rather than guessing."""

_MAX_GUTTER_RUN_FRACTION = 0.85
"""A uniform run spanning nearly this much of the region's own dimension almost certainly is
not a real inter-panel gutter -- it is more likely a single, genuinely flat/plain-colored panel
(e.g. a solid dark splash panel, or a large flat-color background) that happens to be uniform
along this axis too. Per-row/per-column standard deviation alone cannot otherwise distinguish
"blank gutter" from "flat panel content" (both have ~zero internal variance) -- excluding runs
this long avoids a spurious cut straight down the middle of a real, single, flat-colored panel.
See ADR 0007's "Open questions" -- this is a documented limitation, not a solved case."""


@dataclass(frozen=True, slots=True)
class _Leaf:
    """One candidate region, before margin expansion / area filtering -- internal only."""

    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _find_gutter_runs(uniform: np.ndarray, min_len_px: int) -> list[tuple[int, int]]:
    """Contiguous index ranges where `uniform` is True, at least `min_len_px` long."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    values = uniform.tolist()
    for i, is_uniform in enumerate(values):
        if is_uniform and start is None:
            start = i
        elif not is_uniform and start is not None:
            if i - start >= min_len_px:
                runs.append((start, i))
            start = None
    if start is not None and len(values) - start >= min_len_px:
        runs.append((start, len(values)))
    return runs


def _run_confidence(run_std: np.ndarray, min_len_px: int) -> float:
    """How trustworthy one detected gutter run is: more uniform and thicker -> more confident."""
    uniformity_margin = float(np.clip(1.0 - run_std.mean() / _GUTTER_STD_THRESHOLD, 0.0, 1.0))
    thickness_factor = float(np.clip(len(run_std) / (2 * min_len_px), 0.0, 1.0))
    return 0.5 * uniformity_margin + 0.5 * thickness_factor


def _gutter_runs_along(std_profile: np.ndarray) -> list[tuple[int, int, float]]:
    """(start, end, confidence) for each detected gutter run along a 1D std profile."""
    length = len(std_profile)
    min_len = max(_MIN_GUTTER_PX, round(length * _MIN_GUTTER_FRACTION))
    max_len = _MAX_GUTTER_RUN_FRACTION * length
    uniform = std_profile < _GUTTER_STD_THRESHOLD
    runs = _find_gutter_runs(uniform, min_len)
    runs = [(s, e) for s, e in runs if (e - s) <= max_len]
    return [(s, e, _run_confidence(std_profile[s:e], min_len)) for s, e in runs]


def _segments(
    length_px: int, runs: list[tuple[int, int, float]]
) -> list[tuple[int, int, float, float]]:
    """Non-gutter segments between/around `runs`, as `(start, end, start_confidence,
    end_confidence)`. A page edge is treated as perfectly certain evidence (1.0); an internal
    cut's confidence on each side is the bordering gutter run's own confidence. The cut point
    is each run's midpoint, so the (separately applied) context margin still lands inside the
    neighboring segment on both sides rather than back inside the gutter.
    """
    if length_px <= 0:
        return []
    # A run touching a true axis edge (s == 0 or e == length_px) is leading/trailing margin,
    # not a boundary *between* two pieces of content -- it must not generate a cut (that would
    # carve a spurious sliver "segment" out of pure edge margin with no real content on its
    # far side). Only interior runs -- ones with real space, however uniform, on both sides --
    # are genuine candidate boundaries.
    interior_runs = [(s, e, conf) for s, e, conf in runs if s > 0 and e < length_px]
    cuts = sorted({(s + e) // 2 for s, e, _ in interior_runs if 0 < (s + e) // 2 < length_px})
    if not cuts:
        return [(0, length_px, 1.0, 1.0)]

    conf_by_mid = {(s + e) // 2: conf for s, e, conf in interior_runs}
    segments: list[tuple[int, int, float, float]] = []
    prev_pos, prev_conf = 0, 1.0
    for pos in cuts:
        conf = conf_by_mid.get(pos, 1.0)
        if pos > prev_pos:
            segments.append((prev_pos, pos, prev_conf, conf))
        prev_pos, prev_conf = pos, conf
    if prev_pos < length_px:
        segments.append((prev_pos, length_px, prev_conf, 1.0))
    return segments


def _leaf_regions(gray: np.ndarray) -> list[_Leaf]:
    """Row-split the page into bands via horizontal-gutter detection, then attempt one
    column-split pass within each sufficiently tall band -- see ADR 0007's "row-split, then one
    column-split pass within each row band"."""
    h, w = gray.shape
    row_runs = _gutter_runs_along(gray.std(axis=1))
    row_segments = _segments(h, row_runs)

    leaves: list[_Leaf] = []
    for band_index, (by0, by1, top_conf, bottom_conf) in enumerate(row_segments):
        band = gray[by0:by1, :]
        col_segments = [(0, w, 1.0, 1.0)]
        did_column_split = False
        if (by1 - by0) >= _MIN_BAND_HEIGHT_FOR_COLUMN_SPLIT_PX:
            col_runs = _gutter_runs_along(band.std(axis=0))
            candidate = _segments(w, col_runs)
            if len(candidate) > 1:
                col_segments = candidate
                did_column_split = True

        for col_index, (bx0, bx1, left_conf, right_conf) in enumerate(col_segments):
            leaves.append(
                _Leaf(
                    x0=bx0,
                    y0=by0,
                    x1=bx1,
                    y1=by1,
                    confidence=(top_conf + bottom_conf + left_conf + right_conf) / 4.0,
                    metadata={
                        "row_band_index": band_index,
                        "column_index": col_index if did_column_split else None,
                        "top_confidence": top_conf,
                        "bottom_confidence": bottom_conf,
                        "left_confidence": left_conf,
                        "right_confidence": right_conf,
                    },
                )
            )
    return leaves


def _whole_page_candidate(
    image: ImageArray, width: int, height: int, *, reason: str, confidence: float
) -> PanelCandidate:
    bbox = BBoxPx(x0=0, y0=0, x1=width, y1=height)
    return PanelCandidate(
        id="panel_00_fallback_full_page",
        bbox=bbox,
        crop=image[0:height, 0:width].copy(),
        confidence=confidence,
        source="fallback_full_page",
        metadata={"reason": reason},
    )


def _to_candidate(image: ImageArray, leaf: _Leaf, index: int) -> PanelCandidate:
    h, w = image.shape[0], image.shape[1]
    region_w, region_h = leaf.x1 - leaf.x0, leaf.y1 - leaf.y0
    mx = max(0, round(region_w * _CONTEXT_MARGIN_FRACTION))
    my = max(0, round(region_h * _CONTEXT_MARGIN_FRACTION))
    x0, y0 = max(0, leaf.x0 - mx), max(0, leaf.y0 - my)
    x1, y1 = min(w, leaf.x1 + mx), min(h, leaf.y1 + my)
    bbox = BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1)
    metadata = dict(leaf.metadata)
    metadata["context_margin_px"] = {"x": mx, "y": my}
    metadata["tight_bbox"] = (leaf.x0, leaf.y0, leaf.x1, leaf.y1)
    return PanelCandidate(
        id=f"panel_{index:02d}",
        bbox=bbox,
        crop=image[y0:y1, x0:x1].copy(),
        confidence=leaf.confidence,
        source="gutter_xy_cut",
        metadata=metadata,
    )


def detect_panels(image: ImageArray) -> list[PanelCandidate]:
    """Detect panel/region candidates on a full manga page, in page-space pixel coordinates.

    Returns:
    - `[]` if the image is too small for gutter analysis to mean anything (a genuine detector
      failure, not "one panel").
    - A single `source="fallback_full_page"` candidate if no internal gutters were found (a
      valid splash-page read -- see the `manga-analysis` skill) or if every gutter-derived
      region was too small to trust (below `_MIN_PANEL_AREA_FRACTION`).
    - Otherwise, one `source="gutter_xy_cut"` candidate per detected panel, sorted in
      top-to-bottom, then left-to-right reading order (this project's real sample pages are all
      single-column webtoon-style layouts -- see ADR 0007's open question on traditional,
      right-to-left multi-column manga grids).

    Never raises -- an unreliable page produces a low-confidence fallback candidate, not an
    exception; callers decide whether/when to additionally fall back to whole-page VLM analysis
    (see `analysis/plan_builder.py::analyze_page_panels`).
    """
    h, w = image.shape[0], image.shape[1]
    if h < _MIN_IMAGE_DIM_PX or w < _MIN_IMAGE_DIM_PX:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    leaves = _leaf_regions(gray)

    if len(leaves) <= 1:
        return [
            _whole_page_candidate(
                image, w, h, reason="no internal gutters detected", confidence=0.5
            )
        ]

    page_area = w * h
    kept = [
        leaf
        for leaf in leaves
        if ((leaf.x1 - leaf.x0) * (leaf.y1 - leaf.y0)) / page_area >= _MIN_PANEL_AREA_FRACTION
    ]
    if not kept:
        return [
            _whole_page_candidate(
                image,
                w,
                h,
                reason="all candidate regions were below the minimum panel area fraction",
                confidence=0.4,
            )
        ]

    candidates = [_to_candidate(image, leaf, index) for index, leaf in enumerate(kept)]
    candidates.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return candidates
