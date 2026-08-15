"""Phase 19 failure-taxonomy tests: every A-K category is reachable deterministically from the
documented priority rules, and the automatic classifier never emits I (no automatic GT signal
in the phase-17 dataset -- it is reserved for the manual review pass)."""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase19.failure_taxonomy import (
    CONTAMINATION_FRACTION,
    SampleSignal,
    classify,
    describe,
    forbidden_total,
)

_OK = {"text": 0.0, "balloon": 0.0, "frame": 0.0, "onomatopoeia": 0.0}


def _sig(**overrides) -> SampleSignal:
    defaults = dict(
        status="ok",
        n_masks=1,
        coord_ok=True,
        instance_correct=True,
        iou=0.8,
        no_target_text=False,
        forbidden=dict(_OK),
        multi_instance=False,
        manual_category=None,
    )
    defaults.update(overrides)
    return SampleSignal(**defaults)


def test_A_correct_target_good_mask():
    assert classify(_sig(instance_correct=True, iou=0.8)) == "A"


def test_B_correct_target_poor_mask():
    assert classify(_sig(instance_correct=True, iou=0.3)) == "B"


def test_C_wrong_instance():
    assert classify(_sig(instance_correct=False, iou=0.0)) == "C"


def test_D_multiple_instances():
    assert classify(_sig(instance_correct=True, iou=0.8, multi_instance=True)) == "D"
    assert classify(_sig(instance_correct=False, multi_instance=True)) == "D"


def test_E_target_not_identified_vs_F_no_mask():
    # no mask + refusal text -> E
    assert classify(_sig(n_masks=0, no_target_text=True)) == "E"
    # no mask + no refusal text -> F
    assert classify(_sig(n_masks=0, no_target_text=False)) == "F"


def test_G_text_balloon_contamination():
    forbidden = dict(_OK, text=CONTAMINATION_FRACTION + 0.1)
    assert classify(_sig(forbidden=forbidden)) == "G"


def test_H_panel_border_contamination():
    forbidden = dict(_OK, frame=CONTAMINATION_FRACTION + 0.1)
    assert classify(_sig(forbidden=forbidden)) == "H"


def test_J_coordinate_failure():
    assert classify(_sig(coord_ok=False)) == "J"


def test_K_inference_error():
    assert classify(_sig(status="inference_error")) == "K"


def test_contamination_priority_over_instance():
    forbidden = dict(_OK, text=CONTAMINATION_FRACTION + 0.2)
    assert classify(_sig(instance_correct=True, iou=0.9, forbidden=forbidden)) == "G"
    assert classify(_sig(instance_correct=False, iou=0.0, forbidden=forbidden)) == "G"


def test_manual_override_wins():
    assert classify(_sig(iou=0.8, manual_category="I")) == "I"


def test_below_threshold_is_not_contamination():
    forbidden = dict(_OK, text=0.01)
    assert classify(_sig(forbidden=forbidden)) == "A"


def test_forbidden_total_and_labels():
    assert forbidden_total({"text": 0.1, "frame": 0.05}) == pytest.approx(0.15)
    assert describe("C") == "wrong instance"
    assert describe("K") == "inference/runtime failure"
