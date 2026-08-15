"""Phase 18.2A metric aggregation: Qwen direct bbox vs GT, and the SAM downstream masks.

Reuses the Phase 17 metric primitives (`bbox_iou`, `bbox_gt_coverage`, `bbox_area_ratio`,
`mask_metrics`, `compute_distribution`) so the numbers are byte-comparable with Phase 17/18.1.
The phase brief's required aggregates (median/mean/P25/P75, recall at IoU >= 0.25/0.5/0.75,
primary metric Recall@IoU>=0.5) are computed both conditional on the VLM reporting a box AND
over all 64 targets (a not-found/error sample counts as 0 IoU -- the honest all-target rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from manga_animation.benchmarking.phase17.metrics import (
    Distribution,
    compute_distribution,
)

# Recall thresholds to report (primary 0.50; 0.25/0.75 are secondary context).
RECALL_THRESHOLDS = (0.25, 0.50, 0.75)

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class PerTargetMetrics:
    """All Phase 18.2A measurements for ONE target."""

    sample_id: str
    gt_bbox: BBox
    found: bool
    pixel_box: BBox | None
    bbox_iou: float | None
    gt_coverage: float | None
    area_ratio: float | None
    error: str | None
    error_category: str | None
    gs_mask_iou: float | None  # GT bbox -> SAM -> mask vs GT
    qs_mask_iou: float | None  # Qwen bbox -> SAM -> mask vs GT

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "gt_bbox": list(self.gt_bbox),
            "found": self.found,
            "pixel_box": list(self.pixel_box) if self.pixel_box is not None else None,
            "bbox_iou": self.bbox_iou,
            "gt_coverage": self.gt_coverage,
            "area_ratio": self.area_ratio,
            "error": self.error,
            "error_category": self.error_category,
            "gs_mask_iou": self.gs_mask_iou,
            "qs_mask_iou": self.qs_mask_iou,
        }


@dataclass(frozen=True, slots=True)
class Phase18aMetrics:
    n_targets: int
    n_found: int
    found_rate: float
    bbox_iou_found: Distribution
    bbox_iou_all: Distribution
    gt_coverage_found: Distribution
    area_ratio_found: Distribution
    recall_all: dict[float, float]  # found AND iou >= t, over all targets
    recall_found: dict[float, float]  # conditional on found
    n_conversion_failures: int
    gs_mask_iou: Distribution  # reference: GT bbox -> SAM
    qs_mask_iou_found: Distribution  # Qwen bbox -> SAM, samples with a usable box
    qs_mask_iou_all: Distribution  # over all targets (not-found -> 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_targets": self.n_targets,
            "n_found": self.n_found,
            "found_rate": self.found_rate,
            "bbox_iou_found": self.bbox_iou_found.as_dict(),
            "bbox_iou_all": self.bbox_iou_all.as_dict(),
            "gt_coverage_found": self.gt_coverage_found.as_dict(),
            "area_ratio_found": self.area_ratio_found.as_dict(),
            "recall_all": {str(t): v for t, v in self.recall_all.items()},
            "recall_found": {str(t): v for t, v in self.recall_found.items()},
            "n_conversion_failures": self.n_conversion_failures,
            "gs_mask_iou": self.gs_mask_iou.as_dict(),
            "qs_mask_iou_found": self.qs_mask_iou_found.as_dict(),
            "qs_mask_iou_all": self.qs_mask_iou_all.as_dict(),
        }


def compute_metrics(targets: list[PerTargetMetrics]) -> Phase18aMetrics:
    """Aggregate per-target metrics into the phase report's distributions."""
    n = len(targets)
    found = [t for t in targets if t.found and t.pixel_box is not None]
    iou_found = [t.bbox_iou for t in found if t.bbox_iou is not None]
    iou_all = [t.bbox_iou if t.bbox_iou is not None else 0.0 for t in targets]
    coverage = [t.gt_coverage for t in found if t.gt_coverage is not None]
    area_ratio = [t.area_ratio for t in found if t.area_ratio is not None]

    n_conversion = sum(1 for t in targets if t.error is not None)

    recall_all: dict[float, float] = {}
    recall_found: dict[float, float] = {}
    for threshold in RECALL_THRESHOLDS:
        hits_all = sum(1 for t in targets if t.bbox_iou is not None and t.bbox_iou >= threshold)
        recall_all[threshold] = hits_all / n if n else 0.0
        hits_found = sum(1 for t in found if t.bbox_iou is not None and t.bbox_iou >= threshold)
        recall_found[threshold] = hits_found / len(found) if found else 0.0

    gs_iou = [t.gs_mask_iou for t in targets if t.gs_mask_iou is not None]
    qs_iou_found = [t.qs_mask_iou for t in found if t.qs_mask_iou is not None]
    qs_iou_all = [
        t.qs_mask_iou if t.qs_mask_iou is not None else 0.0 for t in targets
    ]

    return Phase18aMetrics(
        n_targets=n,
        n_found=len(found),
        found_rate=len(found) / n if n else 0.0,
        bbox_iou_found=compute_distribution(iou_found),
        bbox_iou_all=compute_distribution(iou_all),
        gt_coverage_found=compute_distribution(coverage),
        area_ratio_found=compute_distribution(area_ratio),
        recall_all=recall_all,
        recall_found=recall_found,
        n_conversion_failures=n_conversion,
        gs_mask_iou=compute_distribution(gs_iou),
        qs_mask_iou_found=compute_distribution(qs_iou_found),
        qs_mask_iou_all=compute_distribution(qs_iou_all),
    )
