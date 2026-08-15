"""Phase 19 controlled-experiment metrics: instance correctness, mask quality, recall, and the
primary END_TO_END_SUCCESS metric -- all pure numpy functions, independently testable.

Definitions (phase brief section 9):

- mask quality: IoU / Dice / precision / recall vs the GT mask (`mask_metrics` from phase 17).
- instance correctness: the predicted mask selected the CORRECT INSTANCE. With only the one GT
  `body` instance per phase-17 sample this is measured as a meaningful overlap with that GT:
  `IoU(pred, GT) >= INSTANCE_IOU_THRESHOLD`. A perfect mask of a different character scores ~0
  and is NOT instance-correct.
- END_TO_END_SUCCESS: `instance_correct AND IoU >= 0.50` -- the primary metric.
- Recall@IoU >= t: fraction of targets whose mask IoU >= t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from manga_animation.benchmarking.phase17.metrics import (
    Distribution,
    MaskMetrics,
    compute_distribution,
    mask_metrics,
)

INSTANCE_IOU_THRESHOLD = 0.25
END_TO_END_IOU = 0.50
RECALL_THRESHOLDS = (0.25, 0.50, 0.75)


def instance_correct(pred_mask: np.ndarray, gt_mask: np.ndarray) -> bool:
    """Did the predicted mask land on the correct instance? IoU with the GT `body` instance >=
    `INSTANCE_IOU_THRESHOLD`. Both masks must share the page geometry (shape mismatch is a
    caller bug)."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"mask shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}"
        )
    return mask_metrics(gt_mask, pred_mask).iou >= INSTANCE_IOU_THRESHOLD


@dataclass(frozen=True, slots=True)
class TargetMetrics:
    """One target's controlled-experiment result."""

    sample_id: str
    metrics: MaskMetrics
    instance_correct: bool
    end_to_end_success: bool
    recall_hits: dict[float, bool]  # IoU >= threshold, per threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "iou": self.metrics.iou,
            "dice": self.metrics.dice,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "instance_correct": self.instance_correct,
            "end_to_end_success": self.end_to_end_success,
            "recall_hits": {str(t): hit for t, hit in self.recall_hits.items()},
        }


def measure_target_metrics(
    sample_id: str, pred_mask: np.ndarray, gt_mask: np.ndarray
) -> TargetMetrics:
    """All controlled metrics for one target. `pred_mask` may be empty (the model emitted no
    mask or a zero mask) -- that is scored as all-zero overlap, not an exception."""
    m = mask_metrics(gt_mask, pred_mask)
    return TargetMetrics(
        sample_id=sample_id,
        metrics=m,
        instance_correct=instance_correct(pred_mask, gt_mask),
        end_to_end_success=instance_correct(pred_mask, gt_mask) and m.iou >= END_TO_END_IOU,
        recall_hits={t: m.iou >= t for t in RECALL_THRESHOLDS},
    )


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Aggregate over targets: IoU/Dice distributions plus the primary success rates."""

    n_targets: int
    iou: Distribution
    dice: Distribution
    recall_at: dict[float, float]  # fraction of targets with IoU >= t
    end_to_end_success_rate: float
    instance_correct_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "n_targets": self.n_targets,
            "iou": self.iou.as_dict(),
            "dice": self.dice.as_dict(),
            "recall_at": {str(t): v for t, v in self.recall_at.items()},
            "end_to_end_success_rate": self.end_to_end_success_rate,
            "instance_correct_rate": self.instance_correct_rate,
        }


def aggregate_metrics(targets: list[TargetMetrics]) -> AggregateMetrics:
    """Aggregate per-target metrics. Targets with no valid metric are counted as failures
    (an empty prediction contributes IoU 0.0 -- never silently dropped)."""
    n = len(targets)
    ious = [t.metrics.iou for t in targets]
    dices = [t.metrics.dice for t in targets]
    failures = sum(1 for t in targets if t.metrics.iou == 0.0)
    recall_at = {
        t: (sum(1 for x in targets if x.metrics.iou >= t) / n if n else 0.0)
        for t in RECALL_THRESHOLDS
    }
    e2e = sum(1 for t in targets if t.end_to_end_success) / n if n else 0.0
    inst = sum(1 for t in targets if t.instance_correct) / n if n else 0.0
    return AggregateMetrics(
        n_targets=n,
        iou=compute_distribution(ious, failures=failures),
        dice=compute_distribution(dices, failures=failures),
        recall_at=recall_at,
        end_to_end_success_rate=e2e,
        instance_correct_rate=inst,
    )
