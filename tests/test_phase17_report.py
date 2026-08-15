"""Phase 17 report tests: aggregation correctness, forbidden-overlap safety track, and the
visual failure package builder (CPU-side, after the GPU experiments)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manga_animation.benchmarking.phase17 import manifest as man_mod
from manga_animation.benchmarking.phase17.dataset import CandidateInstance
from manga_animation.benchmarking.phase17.report import (
    aggregate,
    build_visual_failures,
    compute_forbidden_overlap,
    write_report,
)
from manga_animation.benchmarking.phase17.run import SampleResult

H, W = 100, 120


def _result(
    sample_id: str,
    *,
    a_iou: float,
    b_iou: float,
    c_outcome: str = "accepted",
    c_iou: float = 0.9,
) -> SampleResult:
    exp_c: dict = {
        "grounding_outcome": "detected",
        "outcome": c_outcome,
        "selected_bbox": [30, 30, 70, 70],
    }
    if c_outcome == "accepted":
        exp_c.update({"iou": c_iou, "dice": c_iou, "precision": 0.9, "recall": 0.9})
    else:
        exp_c["failure_detail"] = "rejected by gate"
        exp_c["rejected_raw_iou"] = 0.8
    return SampleResult(
        sample_id=sample_id,
        category="body",
        gt_bbox=(30, 30, 70, 70),
        exp_a={"iou": a_iou, "dice": a_iou, "precision": 0.9, "recall": 0.9,
               "mask_path": "a.npz"},
        exp_b={"status": "detected", "n_detections": 1, "bbox_iou": b_iou,
               "gt_coverage": 0.9, "area_ratio": 1.1},
        exp_c=exp_c,
    )


def test_aggregate_distributions_and_outcomes():
    results = [
        _result("a", a_iou=0.9, b_iou=0.9),
        _result("b", a_iou=0.5, b_iou=0.2),
        _result("c", a_iou=0.7, b_iou=0.95, c_outcome="segment_gate_rejected"),
    ]
    report = aggregate(results)
    assert report.n_samples == 3
    assert report.exp_a["iou"].median == pytest.approx(0.7)
    # Exp C only counts accepted samples.
    assert report.exp_c["iou"].count == 2
    assert report.exp_c_outcomes == {"accepted": 2, "segment_gate_rejected": 1}
    # bbox IoU < 0.3 counts as a wrong-object selection.
    assert report.wrong_object_count == 1
    per = {m.sample_id: m for m in report.per_sample}
    assert per["c"].c_iou is None
    assert per["c"].c_outcome == "segment_gate_rejected"


def test_write_report_writes_json_and_markdown(tmp_path):
    results = [_result("a", a_iou=0.9, b_iou=0.9)]
    report = aggregate(results)
    json_path, md_path = write_report(report, results, tmp_path)
    assert json_path.exists() and md_path.exists()
    import json

    data = json.loads(json_path.read_text())
    assert data["n_samples"] == 1
    assert "exp_a" in data and "exp_b" in data and "exp_c" in data
    md = md_path.read_text()
    assert "## Experiment A" in md
    assert "## Experiment B" in md
    assert "## Experiment C" in md


def _materialize_manifest(tmp_path: Path) -> man_mod.BenchmarkManifest:
    candidates = [
        CandidateInstance(
            sample_id="BOOK_000_0",
            book="BOOK",
            page_index=0,
            instance_id=0,
            category="body",
            semantic_label="character_body",
            gt_bbox=(30, 30, 70, 70),
            gt_area=1600,
            page_size=(H, W),
        )
    ]
    manifest = man_mod.build_manifest(candidates, seed=1, target_size=1)
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[30:70, 30:70] = 120
    from PIL import Image

    Image.fromarray(img).save(dataset / "BOOK_000_0.png")
    gt = np.zeros((H, W), dtype=np.uint8)
    gt[30:70, 30:70] = 255
    np.savez_compressed(dataset / "BOOK_000_0.mask.npz", mask=gt)
    return manifest


def test_build_visual_failures_writes_montage(tmp_path):
    manifest = _materialize_manifest(tmp_path)
    result = _result("BOOK_000_0", a_iou=0.4, b_iou=0.2)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    # Give the failure case real on-disk masks.
    for key in ("exp_a", "exp_c.raw"):
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        np.savez_compressed(run_dir / f"BOOK_000_0.{key}.mask.npz", mask=mask)
    result.exp_a["mask_path"] = str(run_dir / "BOOK_000_0.exp_a.mask.npz")
    result.exp_c["raw_mask_path"] = str(run_dir / "BOOK_000_0.exp_c.raw.mask.npz")
    written = build_visual_failures(
        [result], manifest, tmp_path / "dataset", run_dir / "visual", max_cases=5
    )
    assert len(written) >= 1
    from PIL import Image

    montage = Image.open(written[0])
    assert montage.size[0] > 0 and montage.size[1] > 0


def test_compute_forbidden_overlap(tmp_path, monkeypatch):
    import manga_animation.benchmarking.phase17.report as report_mod

    manifest = _materialize_manifest(tmp_path)
    # A tiny MS92-style book JSON with a balloon mask overlapping the body region.
    from pycocotools import mask as coco_mask

    balloon = np.zeros((H, W), dtype=bool)
    balloon[40:60, 50:65] = True  # half inside the GT body box (30..70)
    book_json = {
        "images": [{"id": 0, "width": W, "height": H, "file_name": "BOOK/000.jpg"}],
        "annotations": [
            {
                "id": 100,
                "category_id": 5,  # balloon
                "iscrowd": 0,
                "image_id": 0,
                "segmentation": coco_mask.encode(np.asfortranarray(balloon.astype(np.uint8))),
            }
        ],
    }
    monkeypatch.setattr(report_mod, "ms92_book_annotations", lambda tok, book, cache: book_json)
    result = _result("BOOK_000_0", a_iou=0.9, b_iou=0.9)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    final_mask = np.zeros((H, W), dtype=np.uint8)
    final_mask[30:70, 30:70] = 255
    final_path = run_dir / "BOOK_000_0.exp_c.final.mask.npz"
    np.savez_compressed(final_path, mask=final_mask)
    result.exp_c["final_mask_path"] = str(final_path)

    overlap = compute_forbidden_overlap([result], manifest, tmp_path / "cache", "tok")
    assert "BOOK_000_0" in overlap
    # balloon covers (10x15=150 px inside the 40x40=1600 px final mask region overlap).
    inter = int(np.count_nonzero(balloon[30:70, 30:70]))  # fully inside here
    assert overlap["BOOK_000_0"]["balloon"] == pytest.approx(inter / 1600)
    assert overlap["BOOK_000_0"]["forbidden_total"] == pytest.approx(inter / 1600)
    assert overlap["BOOK_000_0"]["text"] == 0.0
