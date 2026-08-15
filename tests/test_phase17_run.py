"""Phase 17 runner tests: the three experiments against fake Grounding/SAM clients.

These exercise the real production functions the runner calls (`ground_object_candidates`,
`segment_object`, `_bbox_plausibility`, the real mask gates) with controlled fake model
outputs, so the Exp A / Exp B / Exp C outcomes -- including production gate rejections and
candidate-selection failures -- are verified without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17 import manifest as man_mod
from manga_animation.benchmarking.phase17.dataset import CandidateInstance
from manga_animation.benchmarking.phase17.run import run_benchmark_experiments
from manga_animation.grounding.client import Detection
from manga_animation.pipeline.types import BBoxPx
from manga_animation.segmentation.client import MaskCandidate

H, W = 100, 120


class FakeGrounding:
    model_id = "fake-dino"

    def __init__(self, detections_by_prompt: dict[str, list[Detection]]):
        self._detections = detections_by_prompt
        self.loaded = self.unloaded = 0

    def load(self) -> None:
        self.loaded += 1

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return list(self._detections.get(text_prompt, []))

    def unload(self) -> None:
        self.unloaded += 1


class FakeSegmentation:
    model_id = "fake-sam"

    def __init__(self, mask_for_box):
        self.mask_for_box = mask_for_box
        self.loaded = self.unloaded = 0

    def load(self) -> None:
        self.loaded += 1

    def segment(self, image, box: BBoxPx) -> list[MaskCandidate]:
        mask = self.mask_for_box(box)
        full = np.zeros((H, W), dtype=np.uint8)
        full[mask > 0] = 255
        return [MaskCandidate(mask=full, iou_score=0.9)]

    def unload(self) -> None:
        self.unloaded += 1


def box_filled(box: BBoxPx) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[box.y0 : box.y1, box.x0 : box.x1] = True
    return m


def asymmetric_mask(box: BBoxPx) -> np.ndarray:
    """A mask that trips production's one-sided edge-touch gate: left edge hugged for the box's
    full height, right edge touched only by a thin top strip."""
    m = np.zeros((H, W), dtype=bool)
    w = box.x1 - box.x0
    m[box.y0 : box.y1, box.x0 : box.x0 + w // 2] = True
    m[box.y0 : box.y0 + 3, box.x0 + w // 2 : box.x1] = True
    return m


def make_sample(tmp_path: Path, *, sample_id: str = "BOOK_000_0") -> man_mod.BenchmarkManifest:
    gt_bbox = (30, 30, 70, 70)
    candidates = [
        CandidateInstance(
            sample_id=sample_id,
            book="BOOK",
            page_index=0,
            instance_id=0,
            category="body",
            semantic_label="character_body",
            gt_bbox=gt_bbox,
            gt_area=1600,
            page_size=(H, W),
        )
    ]
    manifest = man_mod.build_manifest(candidates, seed=1, target_size=1)
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[30:70, 30:70] = 120
    Image.fromarray(img).save(dataset / f"{sample_id}.png")
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[30:70, 30:70] = 255
    np.savez_compressed(dataset / f"{sample_id}.mask.npz", mask=gt_mask)
    return manifest


def test_exp_a_pure_sam_on_gt_box_reaches_high_iou(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({"character body.": [Detection("character body", 0.8, (30, 30, 70, 70))]})
    sam = FakeSegmentation(box_filled)
    results = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)
    r = results[0]
    assert r.exp_a["iou"] > 0.99
    assert r.exp_a["mask_path"]  # intermediate preserved
    assert dino.loaded >= 1 and dino.unloaded >= 1
    assert sam.loaded >= 1 and sam.unloaded >= 1


def test_exp_b_records_detection_and_bbox_metrics(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({"character body.": [Detection("character body", 0.9, (30, 30, 70, 70))]})
    sam = FakeSegmentation(box_filled)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_b["status"] == "detected"
    assert r.exp_b["n_detections"] == 1
    assert r.exp_b["bbox_iou"] > 0.99
    assert r.exp_b["gt_coverage"] > 0.99
    assert abs(r.exp_b["area_ratio"] - 1.0) < 0.05


def test_exp_b_no_detection_is_recorded_not_raised(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({})
    sam = FakeSegmentation(box_filled)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_b["status"] == "no_detection"
    assert "bbox_iou" not in r.exp_b
    assert r.exp_c["outcome"] == "candidate_selection_rejected"


def test_exp_c_happy_path_production_accepts(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({"character body.": [Detection("character body", 0.8, (30, 30, 70, 70))]})
    sam = FakeSegmentation(box_filled)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_c["outcome"] == "accepted"
    assert r.exp_c["iou"] > 0.99
    assert r.exp_c["final_mask_path"] and r.exp_c["raw_mask_path"]


def test_exp_c_segment_gate_rejection_preserves_raw_mask(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({"character body.": [Detection("character body", 0.8, (30, 30, 70, 70))]})
    sam = FakeSegmentation(asymmetric_mask)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_c["outcome"] == "segment_gate_rejected"
    assert "edge" in r.exp_c["failure_detail"].lower() or "hugs" in r.exp_c["failure_detail"]
    # The raw pre-gate mask is preserved for forensics (brief section 8).
    assert r.exp_c["raw_mask_path"] and Path(r.exp_c["raw_mask_path"]).exists()
    assert "rejected_raw_iou" in r.exp_c


def test_exp_c_implausible_dino_box_rejected_by_candidate_selection(tmp_path):
    # A DINO box covering >90% of the image fails the deterministic bbox-plausibility check.
    manifest = make_sample(tmp_path)
    dino = FakeGrounding(
        {"character body.": [Detection("character body", 0.8, (0, 0, W, H))]}
    )
    sam = FakeSegmentation(box_filled)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_c["outcome"] == "candidate_selection_rejected"
    assert "plausibility" in r.exp_c["failure_detail"]


def test_wrong_object_selection_low_bbox_iou_recorded(tmp_path):
    manifest = make_sample(tmp_path)
    # DINO finds a different object far from the GT instance.
    dino = FakeGrounding({"character body.": [Detection("character body", 0.9, (5, 5, 20, 20))]})
    sam = FakeSegmentation(box_filled)
    r = run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)[0]
    assert r.exp_b["bbox_iou"] < 0.3


def test_runner_does_not_co_reside_models(tmp_path):
    manifest = make_sample(tmp_path)
    dino = FakeGrounding({"character body.": [Detection("character body", 0.8, (30, 30, 70, 70))]})
    sam = FakeSegmentation(box_filled)
    run_benchmark_experiments(manifest, tmp_path / "dataset", tmp_path / "run", dino, sam)
    # Two stages: grounding first, then segmentation -- each loaded and released.
    assert dino.loaded == 1 and dino.unloaded == 1
    assert sam.loaded == 1 and sam.unloaded == 1
