"""Phase 18.2A metrics + prompt tests (pure logic, no GPU)."""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase18a.metrics import (
    PerTargetMetrics,
    compute_metrics,
)
from manga_animation.benchmarking.phase18a.prompt import build_direct_prompt

GT = (10, 10, 30, 30)


def _target(
    sample_id: str,
    *,
    found: bool = True,
    pixel_box=None,
    iou: float | None = None,
    error: str | None = None,
    gs: float | None = 0.9,
    qs: float | None = None,
) -> PerTargetMetrics:
    return PerTargetMetrics(
        sample_id=sample_id,
        gt_bbox=GT,
        found=found,
        pixel_box=pixel_box,
        bbox_iou=iou,
        gt_coverage=None,
        area_ratio=None,
        error=error,
        error_category=None,
        gs_mask_iou=gs,
        qs_mask_iou=qs,
    )


def test_metrics_recall_over_all_targets():
    # 3 targets: perfect (IoU 1.0), wrong (0.0), not found (None -> 0).
    targets = [
        _target("a", pixel_box=GT, iou=1.0, qs=0.8),
        _target("b", pixel_box=(100, 100, 120, 120), iou=0.0, qs=0.1),
        _target("c", found=False, iou=None, error=None, qs=None),
    ]
    m = compute_metrics(targets)
    assert m.n_targets == 3
    assert m.n_found == 2
    assert m.found_rate == pytest.approx(2 / 3)
    assert m.recall_all[0.5] == pytest.approx(1 / 3)  # only the perfect one over all 3
    assert m.recall_found[0.5] == pytest.approx(1 / 2)  # conditional on found
    assert m.bbox_iou_found.median == pytest.approx(0.5)
    # not-found counts as 0 in the all-target IoU distribution
    assert m.bbox_iou_all.minimum == 0.0


def test_metrics_conversion_failure_count():
    targets = [
        _target("a", pixel_box=GT, iou=1.0, error=None),
        _target("b", iou=None, error="unparseable response"),
    ]
    m = compute_metrics(targets)
    assert m.n_conversion_failures == 1
    assert m.recall_all[0.5] == pytest.approx(0.5)


def test_metrics_empty_targets():
    m = compute_metrics([])
    assert m.n_targets == 0 and m.found_rate == 0.0
    assert m.recall_all[0.5] == 0.0


def test_metrics_mask_aggregation():
    targets = [
        _target("a", pixel_box=GT, iou=1.0, gs=0.88, qs=0.82),
        _target("b", pixel_box=(5, 5, 20, 20), iou=0.3, gs=0.9, qs=0.4),
        _target("c", found=False, iou=None, gs=0.85, qs=None),
    ]
    m = compute_metrics(targets)
    assert m.gs_mask_iou.median == pytest.approx(0.88)
    assert m.qs_mask_iou_found.median == pytest.approx(0.61)
    # not-found has no downstream mask -> 0 in the all-target aggregation
    assert m.qs_mask_iou_all.minimum == 0.0
    assert m.qs_mask_iou_all.count == 3


def test_prompt_contains_target_description_and_convention():
    prompt = build_direct_prompt("character body.")
    assert "character body." in prompt
    assert "0..1000" in prompt
    assert "found" in prompt and "bbox" in prompt
    assert "x1, y1, x2, y2" in prompt
