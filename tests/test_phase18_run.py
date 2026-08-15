"""Phase 18.1 runner/report tests: target-recall computation and report aggregation with
synthetic per-page detections (no GPU)."""

from __future__ import annotations

import json

from manga_animation.benchmarking.phase17.dataset import CandidateInstance
from manga_animation.benchmarking.phase17.manifest import build_manifest
from manga_animation.benchmarking.phase18.report import build_report
from manga_animation.benchmarking.phase18.run import (
    _unique_pages,
    compute_target_recall,
)


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
        _candidate("BOOK_000_1", 0, (50, 50, 70, 70), instance_id=1),
        _candidate("BOOK_001_2", 1, (10, 10, 30, 30), instance_id=2),
    ]
    return build_manifest(candidates, seed=1, target_size=3)


def test_unique_pages_groups_by_page():
    m = _manifest()
    pages = _unique_pages(m)
    assert set(pages) == {"BOOK_000", "BOOK_001"}
    assert len(pages["BOOK_000"]) == 2  # both targets on page 0 share one detection set


def test_compute_target_recall_uses_page_detections():
    m = _manifest()
    # Page 0: a correct candidate exists (rank 2); page 1: only a wrong box.
    detections = {
        "BOOK_000": [
            {"box": [100, 100, 120, 120], "score": 0.9},
            {"box": [10, 10, 30, 30], "score": 0.5},  # matches BOOK_000_0's GT exactly
            {"box": [50, 50, 70, 70], "score": 0.4},  # matches BOOK_000_1's GT exactly
        ],
        "BOOK_001": [{"box": [150, 150, 170, 170], "score": 0.8}],
    }
    targets = compute_target_recall(m, detections)
    by_id = {t.sample_id: t for t in targets}
    assert by_id["BOOK_000_0"].per_threshold[0.5].category == "B"  # correct at rank 2
    assert by_id["BOOK_000_1"].per_threshold[0.5].category == "B"  # correct at rank 3
    assert by_id["BOOK_001_2"].per_threshold[0.5].category == "C"  # no correct candidate
    assert by_id["BOOK_000_0"].n_candidates == 3
    assert by_id["BOOK_001_2"].n_candidates == 1


def test_build_report_aggregates_recall():
    m = _manifest()
    detections = {
        "BOOK_000": [
            {"box": [10, 10, 30, 30], "score": 0.9},  # correct target 0 at rank 1
            {"box": [50, 50, 70, 70], "score": 0.8},  # correct target 1 at rank 2
        ],
        "BOOK_001": [{"box": [150, 150, 170, 170], "score": 0.8}],
    }
    targets = compute_target_recall(m, detections)
    report = build_report(targets)
    assert report.n_targets == 3
    assert report.n_pages == 2
    curve = report.curves[0.5]
    assert curve.recall_at_k[1] == 1 / 3  # only target 0 within top-1
    assert curve.recall_at_k[5] == 2 / 3  # + target 1
    assert curve.recall_at_k[None] == 2 / 3
    assert curve.category_counts == {"A": 1, "B": 1, "C": 1}
    assert report.candidate_below_top1[0.5] == 1  # one B case


def test_report_serializes_to_json(tmp_path):
    m = _manifest()
    targets = compute_target_recall(
        m, {"BOOK_000": [{"box": [10, 10, 30, 30], "score": 0.9}], "BOOK_001": []}
    )
    report = build_report(targets)
    as_dict = report.as_dict()
    json.dumps(as_dict)  # must be JSON-serializable
    assert as_dict["n_targets"] == 3
    assert "0.5" in as_dict["curves"]
