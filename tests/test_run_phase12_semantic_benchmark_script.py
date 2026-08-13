"""Tests for scripts/run_phase12_semantic_benchmark.py's pure helper functions and

src/manga_animation/evaluation/mask_dataset.py's loader -- `scripts/` has no `__init__.py`
(same established convention as tests/test_run_phase9_evaluation_script.py), so the script
module is loaded directly by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from manga_animation.evaluation.mask_dataset import (
    DEFAULT_MASK_BENCHMARK_PATH,
    MaskSemanticSample,
    load_mask_semantic_benchmark,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_phase12_semantic_benchmark.py"
)
_spec = importlib.util.spec_from_file_location("run_phase12_semantic_benchmark", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_best_threshold = _module._best_threshold
_summarize = _module._summarize
_geometric_signal_method = _module._geometric_signal_method
MethodPrediction = _module.MethodPrediction


# --- the real committed dataset config parses and matches its own documented composition ----


def test_load_mask_semantic_benchmark_parses_the_real_config():
    samples = load_mask_semantic_benchmark(DEFAULT_MASK_BENCHMARK_PATH)
    assert len(samples) == 13
    bad = [s for s in samples if s.ground_truth == "bad"]
    good = [s for s in samples if s.ground_truth == "good"]
    assert len(bad) == 5
    assert len(good) == 8
    assert all(s.evidence and s.provenance for s in samples)  # no unlabeled/undocumented entry


def test_mask_semantic_sample_artifacts_available_false_for_missing_files(tmp_path: Path):
    sample = MaskSemanticSample(
        sample_id="fake",
        source_page=tmp_path / "does_not_exist.png",
        mask_path=tmp_path / "does_not_exist.npy",
        semantic_label="hair",
        bbox_xyxy=(0, 0, 10, 10),
        transform_kind="translate",
        ground_truth="good",
        evidence="test fixture",
        provenance="test fixture",
    )
    assert sample.artifacts_available() is False


def _fake_sample(
    sample_id: str, ground_truth: str, difficulty: str = "typical"
) -> MaskSemanticSample:
    return MaskSemanticSample(
        sample_id=sample_id,
        source_page=Path("unused.png"),
        mask_path=Path("unused.npy"),
        semantic_label="hair",
        bbox_xyxy=(0, 0, 10, 10),
        transform_kind="translate",
        ground_truth=ground_truth,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
        evidence="test fixture",
        provenance="test fixture",
    )


# --- _best_threshold: exhaustive sweep for the best-fit single cut point --------------------


def test_best_threshold_finds_a_perfectly_separating_cut():
    values = [0.1, 0.2, 0.8, 0.9]
    labels = ["good", "good", "bad", "bad"]
    threshold, direction, accuracy = _best_threshold(values, labels)
    assert accuracy == 1.0
    assert direction == "above"
    assert 0.2 < threshold < 0.8


def test_best_threshold_reports_less_than_perfect_accuracy_when_ranges_overlap():
    """Mirrors the real Phase 11 finding this sweep formalizes: overlapping ranges cannot be

    perfectly separated by any single threshold, however it's chosen.
    """
    values = [0.4, 0.5, 0.5, 0.9]
    labels = ["bad", "good", "bad", "good"]
    _, _, accuracy = _best_threshold(values, labels)
    assert accuracy < 1.0


# --- _summarize: confusion matrix and derived rates ------------------------------------------


def test_summarize_computes_confusion_matrix_and_rates():
    predictions = [
        MethodPrediction("a", "bad", "typical", "bad", 0.9, "r"),  # TP
        MethodPrediction("b", "good", "typical", "bad", 0.9, "r"),  # FP
        MethodPrediction("c", "good", "typical", "good", 0.9, "r"),  # TN
        MethodPrediction("d", "bad", "typical", "good", 0.9, "r"),  # FN
        MethodPrediction("e", "bad", "typical", "abstain", 0.5, "r"),  # abstain
    ]
    report = _summarize("test_method", predictions)
    assert report.true_positive == 1
    assert report.false_positive == 1
    assert report.true_negative == 1
    assert report.false_negative == 1
    assert report.abstain == 1
    assert report.n_bad == 3
    assert report.n_good == 2
    assert report.precision == 0.5  # 1 / (1 + 1)
    assert report.recall == 0.5  # 1 / (1 + 1)
    assert report.false_positive_rate == 0.5  # 1 / 2 good
    assert report.abstain_rate == 1 / 5


def test_summarize_handles_zero_positive_predictions_without_dividing_by_zero():
    predictions = [
        MethodPrediction("a", "good", "typical", "good", 0.9, "r"),
        MethodPrediction("b", "good", "typical", "good", 0.9, "r"),
    ]
    report = _summarize("test_method", predictions)
    assert report.precision is None  # no positive predictions at all -- n/a, not 0/0
    assert report.recall is None  # no real positives in this slice either


# --- _geometric_signal_method: uses the best-fit threshold end to end -----------------------


def test_geometric_signal_method_correctly_classifies_a_cleanly_separable_signal():
    samples = [
        _fake_sample("bad_1", "bad"),
        _fake_sample("bad_2", "bad"),
        _fake_sample("good_1", "good"),
        _fake_sample("good_2", "good"),
    ]
    signals_by_sample = {
        "bad_1": {"density": 0.9},
        "bad_2": {"density": 0.95},
        "good_1": {"density": 0.1},
        "good_2": {"density": 0.2},
    }
    report = _geometric_signal_method("density", samples, signals_by_sample)
    assert report.method == "geometric:density"
    assert report.true_positive == 2
    assert report.false_positive == 0
    assert report.false_negative == 0
