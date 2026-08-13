"""Regression tests for scripts/run_phase9_evaluation.py's pure helper functions -- dataset

loading, resume-state handling, and rate rendering. `scripts/` has no `__init__.py` (a
deliberate, pre-existing convention), so the module is loaded directly by file path, matching
`tests/test_run_phase3_3_evaluation_script.py`'s established approach.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from manga_animation.evaluation import EvaluationReport, Rate, StatusBreakdown
from manga_animation.evaluation.dataset import DEFAULT_DATASET_PATH, REALWORLD_DATASET_PATH

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase9_evaluation.py"
_spec = importlib.util.spec_from_file_location("run_phase9_evaluation", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_load_dataset = _module._load_dataset
_check_missing_images = _module._check_missing_images
_load_resume_state = _module._load_resume_state
_write_json = _module._write_json
_render_rates_in_place = _module._render_rates_in_place


def test_load_dataset_defaults_to_realworld_only():
    samples = _load_dataset(include_golden=False)
    assert len(samples) == 10
    assert all(s.sample_id.startswith("realworld_") for s in samples)


def test_load_dataset_with_include_golden_prepends_the_golden_set():
    realworld_only = _load_dataset(include_golden=False)
    combined = _load_dataset(include_golden=True)
    assert len(combined) == len(realworld_only) + 7
    golden_ids = {s.sample_id for s in combined if not s.sample_id.startswith("realworld_")}
    assert len(golden_ids) == 7


def test_check_missing_images_passes_when_every_image_exists(tmp_path):
    from manga_animation.evaluation.dataset import EvalSample

    image = tmp_path / "present.png"
    image.write_bytes(b"not a real png, just needs to exist")
    sample = EvalSample(
        sample_id="s",
        image_path=str(image),
        source_citation="t",
        diversity_tag="t",
        acceptable_outcome="t",
    )
    _check_missing_images([sample])  # must not raise


def test_check_missing_images_raises_systemexit_listing_the_gap(tmp_path):
    import pytest

    from manga_animation.evaluation.dataset import EvalSample

    sample = EvalSample(
        sample_id="s",
        image_path=str(tmp_path / "missing.png"),
        source_citation="t",
        diversity_tag="t",
        fetch_script="scripts/fetch_phase9_realworld_pages.py",
        acceptable_outcome="t",
    )
    with pytest.raises(SystemExit, match="missing.png"):
        _check_missing_images([sample])


def test_load_resume_state_with_no_path_returns_an_empty_skeleton():
    state = _load_resume_state(None)
    assert state["outcomes"] == {"page": [], "panel": []}
    assert state["nondeterminism"] == []


def test_load_resume_state_with_a_nonexistent_path_returns_an_empty_skeleton(tmp_path):
    state = _load_resume_state(tmp_path / "does_not_exist_yet.json")
    assert state["outcomes"] == {"page": [], "panel": []}


def test_load_resume_state_round_trips_a_previously_written_file(tmp_path):
    path = tmp_path / "partial.json"
    written = {
        "outcomes": {
            "page": [{"sample_id": "a", "analysis_mode": "page", "status": "completed"}],
            "panel": [],
        },
        "nondeterminism": [{"sample_id": "a", "run_count": 2}],
    }
    _write_json(path, written)
    state = _load_resume_state(path)
    assert state["outcomes"]["page"] == written["outcomes"]["page"]
    assert state["nondeterminism"] == written["nondeterminism"]


def test_write_json_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.json"
    _write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}


def _report(sample_count: int, usable_num: int, usable_den: int) -> EvaluationReport:
    return EvaluationReport(
        analysis_mode="page",
        sample_count=sample_count,
        usable_target_rate=Rate(usable_num, usable_den),
        static_rate=Rate(0, sample_count),
        grounding_success_rate=Rate(0, 0),
        validation_acceptance_rate=Rate(0, 0),
        validation_rejection_rate=Rate(0, 0),
        fallback_rate=Rate(0, sample_count),
        end_to_end_completion_rate=Rate(0, sample_count),
        semantic_false_positive_rate=Rate(0, 0),
        semantic_false_negative_rate=Rate(0, 0),
        unresolved_ground_truth_count=0,
        regression_violation_count=0,
        regression_samples_checked=0,
        panel_detection_multi_panel_rate=None,
        secondary_object_render_rate=Rate(0, 0),
        micro_object_render_rate=Rate(0, 0),
        status_breakdown=StatusBreakdown(0, 0, 0, 0),
    )


def test_render_rates_in_place_uses_each_modes_own_report():
    from dataclasses import asdict

    page_report = _report(sample_count=10, usable_num=7, usable_den=10)
    panel_report = _report(sample_count=10, usable_num=8, usable_den=10)
    reports = {"page": page_report, "panel": panel_report}
    serialized = {"page": asdict(page_report), "panel": asdict(panel_report)}

    _render_rates_in_place(reports, serialized)

    assert serialized["page"]["usable_target_rate"]["rendered"] == str(Rate(7, 10))
    assert serialized["panel"]["usable_target_rate"]["rendered"] == str(Rate(8, 10))


def test_realworld_and_golden_dataset_paths_are_distinct_files():
    assert DEFAULT_DATASET_PATH != REALWORLD_DATASET_PATH
    assert DEFAULT_DATASET_PATH.exists()
    assert REALWORLD_DATASET_PATH.exists()
