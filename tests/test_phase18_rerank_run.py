"""Phase 18.2 runner/report tests: end-to-end reranking over synthetic per-page detections with
a fake VLM (no GPU), plus report aggregation and error classification."""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase17.dataset import CandidateInstance
from manga_animation.benchmarking.phase17.manifest import build_manifest
from manga_animation.benchmarking.phase18.report_rerank import build_report
from manga_animation.benchmarking.phase18.rerank import (
    VlmCandidateScore,
)
from manga_animation.benchmarking.phase18.run_rerank import rerank_targets


def _candidate(sample_id: str, page: int, bbox, instance_id: int = 0) -> CandidateInstance:
    return CandidateInstance(
        sample_id=sample_id,
        book="BOOK",
        page_index=page,
        instance_id=instance_id,
        category="body",
        semantic_label="character_body",
        gt_bbox=bbox,
        gt_area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        page_size=(200, 200),
    )


def _manifest():
    candidates = [
        _candidate("BOOK_000_0", 0, (10, 10, 30, 30), instance_id=0),
        _candidate("BOOK_001_1", 1, (10, 10, 30, 30), instance_id=1),
        _candidate("BOOK_002_2", 2, (10, 10, 30, 30), instance_id=2),
    ]
    return build_manifest(candidates, seed=1, target_size=3)


def test_rerank_targets_measures_selection_accuracy():
    m = _manifest()
    detections = {
        "BOOK_000": [
            {"box": [100, 100, 120, 120], "score": 0.9},
            {"box": [10, 10, 30, 30], "score": 0.5},
        ],
        "BOOK_001": [
            {"box": [10, 10, 30, 30], "score": 0.9},
            {"box": [100, 100, 120, 120], "score": 0.5},
        ],
        "BOOK_002": [{"box": [150, 150, 170, 170], "score": 0.8}],
    }
    # VLM: for BOOK_000 the wrong box matches (rank1), correct at rank2; BOOK_001 correct
    # matches (rank1); BOOK_002 nothing matches (category C).
    def key(box):
        return f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"

    def score(box, dino, matches, conf):
        return VlmCandidateScore(box=tuple(box), dino_score=dino, matches=matches,
                                 confidence=conf, reason="r")

    scores = {
        "BOOK_000": {
            key([100, 100, 120, 120]): score([100, 100, 120, 120], 0.9, True, 0.95),
            key([10, 10, 30, 30]): score([10, 10, 30, 30], 0.5, False, 0.5),
        },
        "BOOK_001": {
            key([10, 10, 30, 30]): score([10, 10, 30, 30], 0.9, True, 0.8),
            key([100, 100, 120, 120]): score([100, 100, 120, 120], 0.5, False, 0.6),
        },
        "BOOK_002": {
            key([150, 150, 170, 170]): score([150, 150, 170, 170], 0.8, False, 0.7),
        },
    }
    image_shapes = {k: (200, 200) for k in detections}
    per_target = rerank_targets(m, detections, scores, image_shapes)
    by_id = {e["sample_id"]: e for e in per_target}
    # BOOK_000: VLM picks the wrong box -> not correct.
    assert by_id["BOOK_000_0"]["strategies"]["A"]["selected_correct"] is False
    assert by_id["BOOK_000_0"]["best_available_iou"] >= 0.5  # eligible (category B)
    # BOOK_001: VLM picks the correct box.
    assert by_id["BOOK_001_1"]["strategies"]["A"]["selected_correct"] is True
    # BOOK_002: no correct candidate at all (category C).
    assert by_id["BOOK_002_2"]["best_available_iou"] == 0.0


def test_build_report_splits_eligible_and_category_c():
    m = _manifest()
    detections = {
        "BOOK_000": [{"box": [10, 10, 30, 30], "score": 0.9}],
        "BOOK_001": [
            {"box": [100, 100, 120, 120], "score": 0.9},
            {"box": [10, 10, 30, 30], "score": 0.5},
        ],
        "BOOK_002": [{"box": [150, 150, 170, 170], "score": 0.8}],
    }

    def score(box, matches, conf):
        return VlmCandidateScore(box=tuple(box), dino_score=0.9, matches=matches,
                                 confidence=conf, reason="r")

    def key(box):
        return f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"

    scores = {
        "BOOK_000": {key([10, 10, 30, 30]): score([10, 10, 30, 30], True, 0.9)},
        # VLM prefers the WRONG box (matches=True, high confidence) over the correct one.
        "BOOK_001": {
            key([100, 100, 120, 120]): score([100, 100, 120, 120], True, 0.9),
            key([10, 10, 30, 30]): score([10, 10, 30, 30], True, 0.4),
        },
        "BOOK_002": {key([150, 150, 170, 170]): score([150, 150, 170, 170], False, 0.6)},
    }
    per_target = rerank_targets(m, detections, scores, {k: (200, 200) for k in detections})
    report = build_report(per_target, {"vlm_calls": 3})
    assert report.n_targets == 3
    assert report.n_category_c == 1  # BOOK_002
    # eligible = 2; strategy A selects correctly on BOOK_000 only.
    assert report.strategies["A"].n_eligible == 2
    assert report.strategies["A"].sel_acc_eligible == 0.5
    assert report.strategies["A"].sel_acc_all == pytest.approx(1 / 3)
    assert report.strategies["A"].recall_at_k[1] == pytest.approx(0.5)


def test_classify_error_candidate_absent():
    m = _manifest()
    detections = {"BOOK_002": [{"box": [150, 150, 170, 170], "score": 0.8}]}

    def score(box, matches, conf):
        return VlmCandidateScore(box=tuple(box), dino_score=0.8, matches=matches,
                                 confidence=conf, reason="r")

    def key(box):
        return f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"

    scores = {"BOOK_002": {key([150, 150, 170, 170]): score([150, 150, 170, 170], False, 0.6)}}
    per_target = rerank_targets(m, detections, scores, {"BOOK_002": (200, 200)})
    report = build_report(per_target, {})
    assert report.error_classes.get("A:8_candidate_absent", 0) >= 1
