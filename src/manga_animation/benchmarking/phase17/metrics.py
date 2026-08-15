"""Mask/bbox metrics for the Phase 17 diagnostic, independently verified before use.

Every metric here is a pure function of numpy arrays/tuples -- no model, no state. They are
deliberately tiny and independently testable (see `tests/test_phase17_metrics.py` for the
synthetic-mask verification required by the phase brief section 12: identical masks, no
overlap, partial overlap, contained mask, shifted mask, plus the same cases for bbox IoU).

Coordinate convention matches the rest of the project: boxes are `(x0, y0, x1, y1)` pixel
tuples (half-open, x1/y1 exclusive); masks are `(H, W)` with a nonzero pixel meaning "inside".
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class MaskMetrics:
    """Per-sample mask overlap metrics against the GT mask.

    precision = |GT ∩ pred| / |pred|, recall = |GT ∩ pred| / |GT|, IoU = |GT ∩ pred| / |GT ∪
    pred|, Dice = 2|GT ∩ pred| / (|GT| + |pred|). All in [0, 1]; a fully empty prediction has
    precision = recall = iou = dice = 0 (no true positives), which is the honest measurement
    of "SAM proposed nothing".
    """

    iou: float
    dice: float
    precision: float
    recall: float


def _area(binary: np.ndarray) -> int:
    return int(np.count_nonzero(binary))


def mask_metrics(gt: np.ndarray, pred: np.ndarray) -> MaskMetrics:
    """Metrics of `pred` vs the ground-truth mask `gt`. Both `(H, W)` boolean-ish arrays.

    `gt` must be non-empty (the benchmark only samples instances with a real human mask); an
    empty `gt` is a caller bug and raises. `pred` may be empty -- that is a real, measurable
    outcome (SAM produced nothing) and is scored as all-zero overlap, not an exception.
    """
    gt_bin = gt > 0
    pred_bin = pred > 0
    if gt_bin.shape != pred_bin.shape:
        raise ValueError(
            f"mask shape mismatch: gt {gt_bin.shape} vs pred {pred_bin.shape} -- masks must "
            "share the source image geometry"
        )
    gt_area = _area(gt_bin)
    if gt_area == 0:
        raise ValueError("gt mask is empty -- the benchmark never samples an empty GT")
    pred_area = _area(pred_bin)
    intersection = int(np.count_nonzero(gt_bin & pred_bin))
    if intersection == 0:
        return MaskMetrics(iou=0.0, dice=0.0, precision=0.0, recall=0.0)
    union = gt_area + pred_area - intersection
    return MaskMetrics(
        iou=intersection / union if union else 0.0,
        dice=2 * intersection / (gt_area + pred_area) if (gt_area + pred_area) else 0.0,
        precision=intersection / pred_area if pred_area else 0.0,
        recall=intersection / gt_area,
    )


def _bbox_area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _bbox_intersection_area(a: BBox, b: BBox) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two pixel boxes. 0.0 for disjoint boxes."""
    inter = _bbox_intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union else 0.0


def bbox_gt_coverage(gt: BBox, pred: BBox) -> float:
    """Fraction of the GT box covered by the predicted box (`inter / |gt|`) -- "how much of the
    target did DINO actually find". Distinct from IoU: a huge over-prediction box has high
    coverage but low IoU; a small box inside the target has low coverage.
    """
    gt_area = _bbox_area(gt)
    if gt_area == 0:
        raise ValueError("gt bbox has zero area")
    return _bbox_intersection_area(gt, pred) / gt_area


def bbox_area_ratio(gt: BBox, pred: BBox) -> float:
    """`|pred| / |gt|` -- DINO's box size relative to the target's, catching systematic
    over-/under-boxing (a ratio near 1 with IoU near 1 is ideal; over-prediction inflates the
    ratio while keeping coverage high).
    """
    gt_area = _bbox_area(gt)
    if gt_area == 0:
        raise ValueError("gt bbox has zero area")
    return _bbox_area(pred) / gt_area


@dataclass(frozen=True, slots=True)
class Distribution:
    """Robust summary of a metric's values -- the brief requires median/percentiles because a
    few catastrophic failures can be hidden behind a mean."""

    count: int
    mean: float | None
    median: float | None
    p05: float | None
    p25: float | None
    p75: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None
    std: float | None
    failures: int = 0
    """Number of samples that produced NO valid metric at all (e.g. grounding found nothing,
    or the production segment gate rejected the mask) -- always reported alongside the
    distribution so a zero-count never masquerades as a clean 0.0."""

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "min": self.minimum,
            "max": self.maximum,
            "std": self.std,
            "failures": self.failures,
        }


def compute_distribution(values: list[float], *, failures: int = 0) -> Distribution:
    """Mean/median/percentiles of `values`, or all-None when empty (`failures` still carries the
    no-result count so a report can see how many samples produced nothing at all)."""
    if not values:
        return Distribution(
            count=0,
            mean=None,
            median=None,
            p05=None,
            p25=None,
            p75=None,
            p95=None,
            minimum=None,
            maximum=None,
            std=None,
            failures=failures,
        )
    arr = np.asarray(values, dtype=float)
    return Distribution(
        count=len(arr),
        mean=float(arr.mean()),
        median=float(median(arr)),
        p05=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        std=float(arr.std()),
        failures=failures,
    )
