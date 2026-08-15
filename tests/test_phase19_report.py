"""Phase 19 report tests: taxonomy assignment, zero-overlap fallback for failures, aggregation,
and the report writer (report.json / report.md). No HF token needed -- the forbidden-overlap
step is optional and not exercised here."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from manga_animation.benchmarking.phase19.masks import SquarePad
from manga_animation.benchmarking.phase19.report import (
    assign_failure_categories,
    build_report,
    metrics_list,
    write_report,
)
from manga_animation.benchmarking.phase19.run import assemble_controlled_record
from tests.phase19_fixtures import make_phase19_sample

_IMAGE = np.zeros((60, 100, 3), dtype=np.uint8)
_GT = np.zeros((60, 100), dtype=bool)
_GT[20:60, 30:90] = True


def _padded(mask) -> np.ndarray:
    pad = SquarePad.from_page_size((60, 100))
    canvas = np.zeros((100, 100), dtype=bool)
    canvas[pad.sy : pad.sy + 60, pad.sx : pad.sx + 100] = mask
    return canvas


def _records(tmp_path) -> list:
    ok = assemble_controlled_record(
        make_phase19_sample("OK_001_1"), _IMAGE, _GT,
        SimpleNamespace(text="[SEG]", masks=[_padded(_GT)]),
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
    )
    poor = assemble_controlled_record(
        make_phase19_sample("POOR_002_1"), _IMAGE, _GT,
        SimpleNamespace(text="[SEG]", masks=[_padded(np.zeros((60, 100), dtype=bool))]),
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
    )
    err = assemble_controlled_record(
        make_phase19_sample("ERR_003_1"), _IMAGE, _GT, None,
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
        error=RuntimeError("boom"),
    )
    return [ok, poor, err]


def test_assign_failure_categories(tmp_path):
    records = assign_failure_categories(_records(tmp_path))
    by_id = {r.sample_id: r.failure_category for r in records}
    assert by_id["OK_001_1"] == "A"  # the perfect mask
    assert by_id["POOR_002_1"] == "C"  # empty prediction -> wrong instance
    assert by_id["ERR_003_1"] == "K"  # inference error


def test_metrics_list_zero_fallback(tmp_path):
    records = _records(tmp_path)
    metrics = metrics_list(records)
    assert len(metrics) == 3
    assert metrics[0].metrics.iou == 1.0
    assert metrics[1].metrics.iou == 0.0
    assert metrics[2].metrics.iou == 0.0  # inference error -> honest zero, never dropped
    assert metrics[2].instance_correct is False


def test_build_report_aggregates(tmp_path):
    records = _records(tmp_path)
    report = build_report(records, condition="D", provenance="PRODUCTION_AVAILABLE")
    assert report.n_targets == 3
    assert report.metrics.n_targets == 3
    assert report.metrics.iou.median == 0.0  # [1.0, 0.0, 0.0]
    assert report.failure_counts["K"] == 1
    assert "C" in report.failure_counts  # the empty-mask record -> wrong instance
    assert report.latency_median is None or report.latency_median >= 0.0


def test_write_report(tmp_path):
    records = _records(tmp_path)
    report = build_report(records, condition="D", provenance="PRODUCTION_AVAILABLE")
    json_path, md_path = write_report(report, records, tmp_path)
    assert json_path.exists() and md_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["condition"] == "D"
    assert "end_to_end_success_rate" in data["metrics"]
    assert "failure_counts" in data
    assert "correct target + good mask" in data["failure_labels"].values()
