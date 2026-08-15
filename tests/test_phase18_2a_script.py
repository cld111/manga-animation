"""Phase 18.2A script/runner tests: pure logic only (no GPU, no model load).

Covers the CPU-side pieces the CLI and runner depend on: `_resize_for_vlm` (aspect-preserving,
production-faithful), `build_per_target_metrics` (VLM records + SAM results + classification),
and the `--report-only` rebuild path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase18a.coords import QwenBboxPrediction
from manga_animation.benchmarking.phase18a.run import (
    DirectLocalizationRecord,
    _resize_for_vlm,
    build_per_target_metrics,
)

# A tiny canonical manifest (2 samples) matching phase-17 field shapes.
_SAMPLES = [
    {
        "sample_id": "AkkeraKanjinchou_084_17245",
        "book": "AkkeraKanjinchou",
        "page_index": 84,
        "instance_id": 17245,
        "category": "body",
        "semantic_label": "character_body",
        "prompt": "character body.",
        "gt_bbox": [100, 100, 300, 300],
        "gt_area": 20000,
        "page_size": [1000, 1000],
        "features": {"area_fraction": 0.04},
    },
    {
        "sample_id": "ARMS_072_3763",
        "book": "ARMS",
        "page_index": 72,
        "instance_id": 3763,
        "category": "body",
        "semantic_label": "character_body",
        "prompt": "character body.",
        "gt_bbox": [500, 500, 700, 700],
        "gt_area": 20000,
        "page_size": [1000, 1000],
        "features": {"area_fraction": 0.04},
    },
]


def _manifest() -> BenchmarkManifest:
    from manga_animation.benchmarking.phase17.manifest import ManifestSample

    return BenchmarkManifest(
        version=1,
        seed=17,
        main_category="body",
        samples=[ManifestSample(**s) for s in _SAMPLES],
    )


def _record(
    sample_id: str, found: bool, pixel_box, error: str | None = None
) -> DirectLocalizationRecord:
    from manga_animation.benchmarking.phase17.metrics import bbox_iou

    gt = (100, 100, 300, 300) if sample_id == _SAMPLES[0]["sample_id"] else (500, 500, 700, 700)
    pred = QwenBboxPrediction(
        sample_id=sample_id,
        found=found,
        box_1000=None,
        pixel_box=pixel_box,
        raw_text='{"found": true}' if found else '{"found": false}',
        error=error,
        convention_ok=error is None,
        width=1000,
        height=1000,
    )
    return DirectLocalizationRecord(
        prediction=pred,
        target_description="character body.",
        gt_bbox=gt,
        bbox_iou=bbox_iou(gt, pixel_box) if pixel_box is not None else None,
        gt_coverage=None,
        area_ratio=None,
        page_w=1000,
        page_h=1000,
    )


def test_resize_for_vlm_preserves_aspect_and_bounds():
    tall = Image.fromarray(np.zeros((2000, 1000, 3), dtype=np.uint8))
    resized = _resize_for_vlm(tall, 1000)
    assert resized.size == (500, 1000)  # long edge (h=2000) -> 1000, w scaled
    small = Image.fromarray(np.zeros((200, 100, 3), dtype=np.uint8))
    assert _resize_for_vlm(small, 1000) is small  # never upscale


def test_build_per_target_metrics_classifies_each():
    records = [
        _record(_SAMPLES[0]["sample_id"], True, (100, 100, 300, 300)),
        _record(_SAMPLES[1]["sample_id"], True, (700, 700, 900, 900)),
    ]
    sam = {r.prediction.sample_id: {"gs_iou": 0.9, "qs_iou": 0.8} for r in records}
    targets = build_per_target_metrics(_manifest(), records, sam)
    by_id = {t.sample_id: t for t in targets}
    assert by_id[_SAMPLES[0]["sample_id"]].error_category == "good"
    assert by_id[_SAMPLES[1]["sample_id"]].error_category == "wrong_instance"
    assert by_id[_SAMPLES[0]["sample_id"]].qs_mask_iou == pytest.approx(0.8)


def test_build_per_target_metrics_not_found_record():
    records = [_record(_SAMPLES[1]["sample_id"], False, None)]
    targets = build_per_target_metrics(_manifest(), records, {})
    assert targets[0].error_category == "not_found"
    assert targets[0].bbox_iou is None


def test_report_only_rebuild(tmp_path):
    """The --report-only path must reproduce report.json from saved per-sample JSON."""
    from manga_animation.benchmarking.phase18a.report import build_report, write_report

    records = [
        _record(_SAMPLES[0]["sample_id"], True, (100, 100, 300, 300)),
        _record(_SAMPLES[1]["sample_id"], True, (700, 700, 900, 900)),
    ]
    targets = build_per_target_metrics(_manifest(), records, {})
    report = build_report(targets)
    json_path, _md_path = write_report(report, tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_targets"] == 2
    assert data["metrics"]["recall_all"]["0.5"] == pytest.approx(0.5)
    assert data["category_counts"]["good"] == 1
    assert data["category_counts"]["wrong_instance"] == 1
