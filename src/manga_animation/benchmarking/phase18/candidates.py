"""Phase 18.1: DINO candidate-recall measurement -- pure, GPU-free logic.

The phase-18.1 question (docs/phase18.1-results.md): *does a correct candidate exist among all
Grounding DINO detections, and how high is it ranked by DINO's own confidence score?* This
separates Case A ("candidate exists, just not top-1" -> reranker is the fix) from Case B
("candidate rarely exists" -> grounding/candidate-generation must change first).

Everything in this module is deterministic numpy/dataclass logic, independently unit-tested
(tests/test_phase18_candidates.py). The DINO detections themselves are produced by the real
production client (`run.py`); this module only measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from manga_animation.benchmarking.phase17.metrics import bbox_iou

BBox = tuple[int, int, int, int]

# The recall thresholds to report (primary 0.50; 0.25/0.75 are secondary context). A candidate
# is a "correct match" when its bbox IoU with the GT bbox is at least the threshold.
RECALL_THRESHOLDS = (0.25, 0.50, 0.75)
# The K values for Recall@K (None means "all candidates" -- the entire detection set).
RECALL_K_VALUES: tuple[int | None, ...] = (1, 3, 5, 10, 20, None)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One DINO detection, ranked by the model's own confidence score (1 = highest score)."""

    rank: int  # 1-based position after sorting by score descending
    score: float
    box: BBox


@dataclass(frozen=True, slots=True)
class TargetRecall:
    """All phase-18.1 measurements for one GT target against its page's DINO detections."""

    sample_id: str
    page_key: str  # "<BOOK>_<within-page>"
    gt_bbox: BBox
    n_candidates: int  # total detections above DINO's threshold on the page
    top1_iou: float  # IoU of the top-1 (highest-score) detection with the GT bbox
    best_iou_overall: float  # max IoU over ALL detections
    # per-threshold: best (highest-ranked) correct candidate
    per_threshold: dict[float, "ThresholdRecall"] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "page_key": self.page_key,
            "gt_bbox": list(self.gt_bbox),
            "n_candidates": self.n_candidates,
            "top1_iou": self.top1_iou,
            "best_iou_overall": self.best_iou_overall,
            "per_threshold": {
                str(t): tr.as_dict() for t, tr in sorted(self.per_threshold.items())
            },
        }


@dataclass(frozen=True, slots=True)
class ThresholdRecall:
    """Recall measurements for ONE IoU threshold."""

    threshold: float
    correct_exists: bool  # any candidate with IoU >= threshold
    best_rank: int | None  # rank of the best (highest-scored) correct candidate, 1-based
    best_score: float | None  # DINO confidence of that candidate
    category: str  # "A" (exists and top-1) / "B" (exists below top-1) / "C" (does not exist)
    # best IoU achievable within the top-K highest-scored candidates
    topk_best_iou: dict[int | None, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "correct_exists": self.correct_exists,
            "best_rank": self.best_rank,
            "best_score": self.best_score,
            "category": self.category,
            "topk_best_iou": {str(k): v for k, v in self.topk_best_iou.items()},
        }


def rank_candidates(detections) -> list[RankedCandidate]:
    """Sort the raw DINO detections by the model's own score, descending, assigning 1-based
    ranks exactly as production candidate selection would consume them."""
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    return [
        RankedCandidate(rank=i + 1, score=d.score, box=tuple(int(v) for v in d.box))
        for i, d in enumerate(ordered)
    ]


def _topk_best_ious(
    ranked: list[RankedCandidate], gt_bbox: BBox, k_values: tuple[int | None, ...]
) -> dict[int | None, float]:
    """Best IoU with `gt_bbox` achievable within the top-K highest-scored candidates."""
    result: dict[int | None, float] = {}
    for k in k_values:
        subset = ranked if k is None else ranked[:k]
        result[k] = max((bbox_iou(gt_bbox, c.box) for c in subset), default=0.0)
    return result


def measure_target(
    sample_id: str,
    page_key: str,
    gt_bbox: BBox,
    detections,
    *,
    thresholds: tuple[float, ...] = RECALL_THRESHOLDS,
    k_values: tuple[int | None, ...] = RECALL_K_VALUES,
) -> TargetRecall:
    """Compute the recall measurements for one GT target against its page's detection set."""
    ranked = rank_candidates(detections)
    top1_iou = bbox_iou(gt_bbox, ranked[0].box) if ranked else 0.0
    best_iou_overall = max((bbox_iou(gt_bbox, c.box) for c in ranked), default=0.0)
    per_threshold: dict[float, ThresholdRecall] = {}
    for t in thresholds:
        correct = [c for c in ranked if bbox_iou(gt_bbox, c.box) >= t]
        if correct:
            best = correct[0]  # ranked ascending by rank -> highest-score correct candidate
            category = "A" if best.rank == 1 else "B"
            tr = ThresholdRecall(
                threshold=t,
                correct_exists=True,
                best_rank=best.rank,
                best_score=best.score,
                category=category,
                topk_best_iou=_topk_best_ious(ranked, gt_bbox, k_values),
            )
        else:
            tr = ThresholdRecall(
                threshold=t,
                correct_exists=False,
                best_rank=None,
                best_score=None,
                category="C",
                topk_best_iou=_topk_best_ious(ranked, gt_bbox, k_values),
            )
        per_threshold[t] = tr
    return TargetRecall(
        sample_id=sample_id,
        page_key=page_key,
        gt_bbox=gt_bbox,
        n_candidates=len(ranked),
        top1_iou=top1_iou,
        best_iou_overall=best_iou_overall,
        per_threshold=per_threshold,
    )


@dataclass(frozen=True, slots=True)
class RecallCurve:
    """Recall@K for one IoU threshold across all targets."""

    threshold: float
    n_targets: int
    recall_at_k: dict[int | None, float]  # None -> Recall@All
    category_counts: dict[str, int]  # A / B / C

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_targets": self.n_targets,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "category_counts": self.category_counts,
        }


def recall_curves(
    targets: list[TargetRecall],
    *,
    k_values: tuple[int | None, ...] = RECALL_K_VALUES,
) -> dict[float, RecallCurve]:
    """Aggregate per-target measurements into Recall@K curves per IoU threshold."""
    curves: dict[float, RecallCurve] = {}
    for t in sorted({rec.threshold for rec in targets}):
        subset = [rec for rec in targets if t in rec.per_threshold]
        n = len(subset)
        recall_at_k: dict[int | None, float] = {}
        for k in k_values:
            if n == 0:
                recall_at_k[k] = 0.0
                continue
            hit = sum(
                1
                for rec in subset
                if rec.per_threshold[t].correct_exists
                and (k is None or rec.per_threshold[t].best_rank <= k)
            )
            recall_at_k[k] = hit / n
        category_counts = {
            cat: sum(1 for rec in subset if rec.per_threshold[t].category == cat)
            for cat in ("A", "B", "C")
        }
        curves[t] = RecallCurve(
            threshold=t, n_targets=n, recall_at_k=recall_at_k, category_counts=category_counts
        )
    return curves
