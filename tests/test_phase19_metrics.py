"""Phase 19 metrics tests: instance correctness, END_TO_END_SUCCESS, recall, and aggregation --
all verified on synthetic masks with closed-form expectations (the phase-17 metric-verification
discipline)."""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.benchmarking.phase19.metrics import (
    aggregate_metrics,
    instance_correct,
    measure_target_metrics,
)

# A 60x100 canvas; the GT body occupies x in [30,90), y in [20,60)  (40x60 = 2400 px).
_GT = np.zeros((60, 100), dtype=bool)
_GT[20:60, 30:90] = True


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((60, 100), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_instance_correct_requires_overlap():
    assert instance_correct(_GT.copy(), _GT.copy())
    assert instance_correct(_GT, _mask(30, 20, 90, 60))
    assert not instance_correct(_GT, _mask(0, 0, 20, 20))  # disjoint region


def test_instance_correct_shape_mismatch_raises():
    with pytest.raises(ValueError):
        instance_correct(_GT, np.zeros((30, 50), dtype=bool))


def test_end_to_end_success_requires_correct_instance_and_iou_ge_50():
    perfect = measure_target_metrics("s", _GT.copy(), _GT)
    assert perfect.end_to_end_success
    assert perfect.metrics.iou == pytest.approx(1.0)

    # Same size, half-height overlap vertically: IoU = 0.6 (> 0.50) -> success.
    good = measure_target_metrics("s", _mask(30, 10, 90, 50), _GT)
    assert good.metrics.iou == pytest.approx(0.6)
    assert good.end_to_end_success

    # Same instance, contained mask (IoU = 1/3): correct instance, poor mask -> not success.
    poor = measure_target_metrics("s", _mask(30, 30, 70, 50), _GT)
    assert poor.metrics.iou == pytest.approx(1 / 3)
    assert poor.instance_correct
    assert not poor.end_to_end_success

    # A different character (disjoint): not instance-correct, not success.
    wrong = measure_target_metrics("s", _mask(0, 0, 20, 20), _GT)
    assert not wrong.instance_correct
    assert not wrong.end_to_end_success


def test_recall_hits():
    good = measure_target_metrics("s", _GT.copy(), _GT)
    assert good.recall_hits[0.25] and good.recall_hits[0.50] and good.recall_hits[0.75]
    none = measure_target_metrics("s", np.zeros((60, 100), dtype=bool), _GT)
    assert not any(none.recall_hits.values())
    assert none.metrics.iou == 0.0 and none.metrics.dice == 0.0


def test_aggregate_recall_e2e():
    perfect = measure_target_metrics("a", _GT.copy(), _GT)  # 1.0
    poor = measure_target_metrics("b", _mask(30, 30, 70, 50), _GT)  # 1/3
    wrong = measure_target_metrics("c", _mask(0, 0, 20, 20), _GT)  # 0.0
    empty = measure_target_metrics("d", np.zeros((60, 100), dtype=bool), _GT)  # 0.0
    aggr = aggregate_metrics([perfect, poor, wrong, empty])
    assert aggr.n_targets == 4
    assert aggr.recall_at[0.25] == pytest.approx(2 / 4)
    assert aggr.recall_at[0.50] == pytest.approx(1 / 4)
    assert aggr.recall_at[0.75] == pytest.approx(1 / 4)
    assert aggr.instance_correct_rate == pytest.approx(2 / 4)
    assert aggr.end_to_end_success_rate == pytest.approx(1 / 4)
    assert aggr.iou.median == pytest.approx((1 / 3) / 2)  # median of [1.0, 1/3, 0, 0]
    assert aggr.iou.failures == 2  # wrong + empty


def test_aggregate_empty():
    aggr = aggregate_metrics([])
    assert aggr.n_targets == 0
    assert aggr.end_to_end_success_rate == 0.0
    assert aggr.iou.median is None
