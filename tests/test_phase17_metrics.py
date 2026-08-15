"""Phase 17 metric verification on synthetic masks/boxes -- the phase brief section 12 gate:
the metric implementation must be independently verified before its results are trusted.

Verified cases (mask metrics): identical masks (1.0), no overlap (0.0), partial overlap
(known closed-form values), contained mask, shifted mask. Same families for bbox IoU plus GT
coverage and area-ratio semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.benchmarking.phase17.metrics import (
    bbox_area_ratio,
    bbox_gt_coverage,
    bbox_iou,
    compute_distribution,
    mask_metrics,
)


def _h(w: int, h: int = 60) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _rect(x0: int, y0: int, x1: int, y1: int, canvas: np.ndarray) -> np.ndarray:
    m = canvas.copy()
    m[y0:y1, x0:x1] = True
    return m


# --- mask metrics ------------------------------------------------------------------------


def test_identical_masks_give_perfect_metrics():
    gt = _rect(10, 10, 50, 40, _h(60))
    m = mask_metrics(gt, gt.copy())
    assert m.iou == pytest.approx(1.0)
    assert m.dice == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)


def test_no_overlap_gives_zero_metrics():
    gt = _rect(5, 5, 25, 25, _h(60))
    pred = _rect(35, 35, 55, 55, _h(60))
    m = mask_metrics(gt, pred)
    assert m.iou == 0.0 and m.dice == 0.0 and m.precision == 0.0 and m.recall == 0.0


def test_partial_overlap_closed_form():
    # gt = left half of [10,30]x[10,30] area (20x20=400); pred = right half, half-overlap.
    gt = _rect(10, 10, 30, 30, _h(60))  # 400 px
    pred = _rect(20, 10, 40, 30, _h(60))  # 400 px
    m = mask_metrics(gt, pred)
    assert m.iou == pytest.approx(200 / (400 + 400 - 200))  # 200/600
    assert m.recall == pytest.approx(200 / 400)
    assert m.precision == pytest.approx(200 / 400)
    assert m.dice == pytest.approx(2 * 200 / (400 + 400))


def test_contained_prediction_scores_low_precision_full_recall():
    gt = _rect(10, 10, 50, 50, _h(60))  # 1600 px
    pred = _rect(20, 20, 40, 40, _h(60))  # 400 px, fully inside gt
    m = mask_metrics(gt, pred)
    assert m.recall == pytest.approx(400 / 1600)
    assert m.precision == pytest.approx(1.0)
    assert m.iou == pytest.approx(400 / 1600)


def test_shifted_mask_metrics_reduce_recall_and_iou():
    gt = _rect(10, 10, 30, 30, _h(60))
    pred = _rect(15, 10, 35, 30, _h(60))  # shifted right by 5
    inter = 15 * 20
    m = mask_metrics(gt, pred)
    assert m.iou == pytest.approx(inter / (2 * 400 - inter))


def test_empty_prediction_scores_zero_not_error():
    gt = _rect(10, 10, 30, 30, _h(60))
    m = mask_metrics(gt, np.zeros_like(gt))
    assert m.iou == 0.0 and m.precision == 0.0 and m.recall == 0.0 and m.dice == 0.0


def test_empty_gt_is_a_caller_bug():
    with pytest.raises(ValueError, match="gt mask is empty"):
        mask_metrics(np.zeros((20, 20), dtype=bool), np.ones((20, 20), dtype=bool))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mask_metrics(np.ones((20, 20), dtype=bool), np.ones((21, 20), dtype=bool))


# --- bbox metrics ------------------------------------------------------------------------


def test_bbox_iou_identical_boxes():
    assert bbox_iou((10, 10, 30, 30), (10, 10, 30, 30)) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes():
    assert bbox_iou((10, 10, 20, 20), (50, 50, 60, 60)) == 0.0


def test_bbox_iou_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)  # 50% overlap
    inter = 50
    assert bbox_iou(a, b) == pytest.approx(inter / (100 + 100 - inter))


def test_bbox_iou_contained():
    a = (0, 0, 20, 20)
    b = (5, 5, 15, 15)  # fully inside a
    assert bbox_iou(a, b) == pytest.approx(100 / 400)


def test_bbox_gt_coverage_distinguishes_overprediction():
    gt = (0, 0, 20, 20)
    # Small box inside the GT: full coverage possible only if it covers GT -- test both
    # over-prediction (high coverage, low iou) and under-prediction (low coverage).
    big = (0, 0, 40, 40)
    small = (5, 5, 10, 10)
    assert bbox_gt_coverage(gt, big) == pytest.approx(1.0)
    assert bbox_gt_coverage(gt, small) == pytest.approx(25 / 400)
    # Symmetric check: a prediction half the GT on one side.
    half = (0, 0, 20, 10)
    assert bbox_gt_coverage(gt, half) == pytest.approx(0.5)


def test_bbox_area_ratio():
    gt = (0, 0, 20, 20)
    assert bbox_area_ratio(gt, gt) == pytest.approx(1.0)
    assert bbox_area_ratio(gt, (0, 0, 40, 40)) == pytest.approx(4.0)
    assert bbox_area_ratio(gt, (0, 0, 10, 10)) == pytest.approx(0.25)


def test_bbox_zero_area_gt_raises():
    with pytest.raises(ValueError, match="zero area"):
        bbox_gt_coverage((5, 5, 5, 5), (0, 0, 10, 10))


# --- distribution helper ------------------------------------------------------------------


def test_compute_distribution_mean_median_percentiles():
    # np.percentile uses linear interpolation: with n=5 the 5th/95th percentiles interpolate
    # toward the extremes rather than landing exactly on min/max.
    d = compute_distribution([0.1, 0.2, 0.3, 0.4, 0.5])
    assert d.mean == pytest.approx(0.3)
    assert d.median == pytest.approx(0.3)
    assert d.p05 == pytest.approx(0.12)
    assert d.p95 == pytest.approx(0.48)
    assert d.minimum == pytest.approx(0.1)
    assert d.maximum == pytest.approx(0.5)


def test_compute_distribution_empty_with_failures():
    d = compute_distribution([], failures=3)
    assert d.count == 0
    assert d.mean is None
    assert d.failures == 3
