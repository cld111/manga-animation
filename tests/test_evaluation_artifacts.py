"""Tests for src/manga_animation/evaluation/artifacts.py -- the Phase 9 seam-artifact detector.

Synthetic, deterministic fixtures only (no dependency on the real, git-ignored Phase 8 evidence
videos this check was originally validated against -- see that module's own docstring and
docs/phase9-results.md for the real-data validation numbers). These fixtures reproduce the same
qualitative shapes: a rigid, page-aligned rectangle hugging one edge (the real seam defect's
signature) vs. an organic, centered blob (a healthy object) and a symmetric rectangle
(a legitimately rectangular object, e.g. a banner -- must NOT be flagged, mirroring
`segmentation.segment`'s own `_validate_mask_shape` asymmetry refinement).
"""

from __future__ import annotations

import numpy as np

from manga_animation.evaluation.artifacts import (
    detect_changed_region_shapes,
    detect_seam_like_artifacts,
)

_H, _W = 200, 300


def _blank() -> np.ndarray:
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def _with_rect(y0: int, y1: int, x0: int, x1: int, value: int = 200) -> np.ndarray:
    frame = _blank()
    frame[y0:y1, x0:x1] = value
    return frame


def _with_circle(cy: int, cx: int, radius: int, value: int = 200) -> np.ndarray:
    frame = _blank()
    yy, xx = np.ogrid[:_H, :_W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    frame[mask] = value
    return frame


def _with_seam_shape(value: int = 200) -> np.ndarray:
    """A shape reproducing the real seam defect's own signature: a dead-straight left edge

    (flush against x=0 for every row, like an over-segmented mask's rigid leaked boundary)
    combined with a naturally varying right edge (a triangular taper peaking at one row, like an
    organic silhouette) -- only the row(s) right at the peak actually touch the tight bbox's
    right edge, so most rows do not. A solid rectangle (constant width) would trivially hug BOTH
    edges at 100% for every row, which is NOT the real defect's shape -- see the module
    docstring's real evidence (one edge ~90%, the opposite ~2-5%).
    """
    frame = _blank()
    peak_y = 99
    for y in range(20, 180):
        width = 40 + (30 - min(abs(y - peak_y), 30))  # rises to 70 at peak_y, 40 elsewhere
        frame[y, 0:width] = value
    return frame


def test_detect_changed_region_shapes_finds_nothing_between_identical_frames():
    frame = _with_circle(100, 150, 40)
    assert detect_changed_region_shapes(frame, frame) == []


def test_seam_like_shape_flush_against_left_edge_is_flagged():
    """A shape with a dead-straight left edge and a naturally varying right edge -- the real

    seam defect's own asymmetric-edge shape (one side hugged at ~90%+, the opposite side clean).
    """
    frame_a = _blank()
    frame_b = _with_seam_shape()
    report = detect_seam_like_artifacts([frame_a, frame_b, frame_a])
    assert report is not None
    assert report.seam_suspected is True
    assert report.worst_component is not None
    assert report.worst_component.left_edge_fraction > 0.8
    assert report.worst_component.right_edge_fraction < 0.15


def test_organic_centered_blob_is_not_flagged():
    frame_a = _blank()
    frame_b = _with_circle(cy=100, cx=150, radius=50)
    report = detect_seam_like_artifacts([frame_a, frame_b, frame_a])
    assert report is not None
    assert report.seam_suspected is False


def test_symmetric_rectangle_spanning_the_full_bbox_is_not_flagged():
    """A genuinely rectangular changed region (both left AND right edges hugged together, e.g.

    a real banner/flag object) must not be flagged -- it has no asymmetry, unlike the real seam
    defect. Mirrors `segmentation.segment`'s own reviewed false-positive fix.
    """
    frame_a = _blank()
    frame_b = _with_rect(y0=20, y1=180, x0=0, x1=_W)
    report = detect_seam_like_artifacts([frame_a, frame_b, frame_a])
    assert report is not None
    assert report.worst_component is not None
    assert report.worst_component.left_edge_fraction > 0.8
    assert report.worst_component.right_edge_fraction > 0.8  # both hugged -> symmetric
    assert report.seam_suspected is False


def test_detect_seam_like_artifacts_returns_none_for_fewer_than_two_frames():
    assert detect_seam_like_artifacts([_blank()]) is None
    assert detect_seam_like_artifacts([]) is None


def test_detect_seam_like_artifacts_samples_evenly_across_a_longer_sequence():
    frames = [_blank() for _ in range(48)]
    frames[24] = _with_seam_shape()  # a real defect only at mid-cycle
    report = detect_seam_like_artifacts(frames, sample_count=12)
    assert report is not None
    assert report.seam_suspected is True
