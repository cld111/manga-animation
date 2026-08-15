"""Phase 18.1 pure-logic tests: ranking, best-match, Recall@K curves, and the A/B/C split.

Uses synthetic DINO detection lists (no GPU): the recall logic must be correct before the
GPU-collected detections are trusted.
"""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase18.candidates import (
    measure_target,
    rank_candidates,
    recall_curves,
)
from manga_animation.grounding.client import Detection

# GT target bbox: a 20x20 box at (10, 10)-(30, 30).
GT = (10, 10, 30, 30)


def _det(score: float, box) -> Detection:
    return Detection(label="character body", score=score, box=box)


def test_rank_candidates_sorts_by_score_descending():
    dets = [_det(0.3, (0, 0, 10, 10)), _det(0.9, (10, 10, 30, 30)), _det(0.6, (5, 5, 20, 20))]
    ranked = rank_candidates(dets)
    assert [c.score for c in ranked] == pytest.approx([0.9, 0.6, 0.3])
    assert [c.rank for c in ranked] == [1, 2, 3]


def test_measure_target_category_a_correct_top1():
    # Top-1 detection IS the target (IoU ~1.0); a wrong low-score detection also exists.
    dets = [_det(0.9, (10, 10, 30, 30)), _det(0.2, (100, 100, 120, 120))]
    rec = measure_target("s1", "BOOK_000", GT, dets)
    tr = rec.per_threshold[0.5]
    assert tr.correct_exists and tr.category == "A"
    assert tr.best_rank == 1
    assert tr.best_score == pytest.approx(0.9)
    assert rec.best_iou_overall > 0.99
    assert rec.n_candidates == 2


def test_measure_target_category_b_correct_below_top1():
    # A wrong high-score detection is top-1; the correct target is rank 2.
    dets = [_det(0.95, (100, 100, 120, 120)), _det(0.6, (10, 10, 30, 30))]
    rec = measure_target("s1", "BOOK_000", GT, dets)
    tr = rec.per_threshold[0.5]
    assert tr.correct_exists and tr.category == "B"
    assert tr.best_rank == 2
    assert rec.top1_iou == 0.0  # top-1 is the wrong box


def test_measure_target_category_c_no_candidate():
    dets = [_det(0.9, (100, 100, 120, 120)), _det(0.8, (200, 200, 220, 220))]
    rec = measure_target("s1", "BOOK_000", GT, dets)
    tr = rec.per_threshold[0.5]
    assert not tr.correct_exists and tr.category == "C"
    assert tr.best_rank is None and tr.best_score is None
    assert rec.best_iou_overall == 0.0


def test_threshold_dependence():
    # Detection (10,10,40,40) contains GT (20x20) fully: IoU = 400/900 = 0.444.
    dets = [_det(0.8, (10, 10, 40, 40))]
    rec = measure_target("s1", "BOOK_000", GT, dets)
    assert rec.per_threshold[0.25].correct_exists
    assert not rec.per_threshold[0.5].correct_exists  # 0.444 < 0.5
    assert not rec.per_threshold[0.75].correct_exists
    # A slightly tighter box (10,10,36,36): IoU = 400/676 = 0.592 -> now passes 0.5.
    dets2 = [_det(0.8, (10, 10, 36, 36))]
    rec2 = measure_target("s1", "BOOK_000", GT, dets2)
    assert rec2.per_threshold[0.5].correct_exists
    assert not rec2.per_threshold[0.75].correct_exists


def test_topk_best_iou_and_empty_detections():
    rec = measure_target("s1", "BOOK_000", GT, [])
    assert rec.n_candidates == 0
    assert rec.per_threshold[0.5].category == "C"
    assert rec.per_threshold[0.5].topk_best_iou[1] == 0.0
    assert rec.per_threshold[0.5].topk_best_iou[None] == 0.0


def test_topk_best_iou_increases_with_k():
    # Wrong top-1, correct target at rank 2.
    dets = [_det(0.9, (100, 100, 120, 120)), _det(0.5, (10, 10, 30, 30))]
    rec = measure_target("s1", "BOOK_000", GT, dets)
    tr = rec.per_threshold[0.5]
    assert tr.topk_best_iou[1] == pytest.approx(0.0)
    assert tr.topk_best_iou[3] > 0.99
    assert tr.topk_best_iou[None] > 0.99


def test_recall_curves_aggregation():
    # 3 targets: A, B(rank 2), C(no candidate).
    recs = [
        measure_target("a", "P1", GT, [_det(0.9, (10, 10, 30, 30))]),
        measure_target(
            "b", "P2", GT,
            [_det(0.9, (100, 100, 120, 120)), _det(0.5, (10, 10, 30, 30))],
        ),
        measure_target("c", "P3", GT, [_det(0.9, (200, 200, 220, 220))]),
    ]
    curves = recall_curves(recs)
    c = curves[0.5]
    assert c.n_targets == 3
    assert c.recall_at_k[1] == pytest.approx(1 / 3)  # only target A within top-1
    assert c.recall_at_k[3] == pytest.approx(2 / 3)  # A + B(rank 2)
    assert c.recall_at_k[None] == pytest.approx(2 / 3)  # C never recallable
    assert c.category_counts == {"A": 1, "B": 1, "C": 1}
    assert c.recall_at_k[20] == pytest.approx(2 / 3)
