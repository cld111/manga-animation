from __future__ import annotations

import time

import pytest
import yaml
from pydantic import ValidationError

from manga_animation.benchmarking.registry import flat_candidates, load_candidates
from manga_animation.benchmarking.report import render_markdown
from manga_animation.benchmarking.runner import run_benchmark, run_sweep
from manga_animation.benchmarking.schemas import BenchmarkResult, ModelCandidate


class FakeAdapter:
    """A `ModelAdapter` double: no real model, just controllable, timeable behavior."""

    def __init__(self, *, infer_seconds: float = 0.0, fail_on: str | None = None):
        self.infer_seconds = infer_seconds
        self.fail_on = fail_on
        self.loaded = False
        self.unloaded = False

    def load(self):
        if self.fail_on == "load":
            raise RuntimeError("simulated load failure")
        self.loaded = True

    def infer(self, sample):
        if self.fail_on == "infer":
            raise RuntimeError("simulated infer failure")
        if self.infer_seconds:
            time.sleep(self.infer_seconds)
        return sample

    def unload(self):
        self.unloaded = True


def make_candidate(**overrides) -> ModelCandidate:
    defaults = dict(id="fake-model", stage="vlm", source="local/fake", license="mit", params="0")
    defaults.update(overrides)
    return ModelCandidate(**defaults)


# --- registry ---------------------------------------------------------------


def test_load_candidates_reads_the_real_repo_manifest():
    by_stage = load_candidates()
    for stage in ("vlm", "grounding", "segmentation", "inpainting"):
        assert stage in by_stage
        assert len(by_stage[stage]) >= 1
        for candidate in by_stage[stage]:
            assert candidate.stage == stage
            assert candidate.id


def test_flat_candidates_matches_sum_of_grouped_candidates():
    by_stage = load_candidates()
    flat = flat_candidates()
    assert len(flat) == sum(len(v) for v in by_stage.values())


def test_candidate_ids_are_unique_within_the_real_manifest():
    flat = flat_candidates()
    ids = [c.id for c in flat]
    assert len(ids) == len(set(ids)), f"duplicate candidate ids: {sorted(ids)}"


def test_load_candidates_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates(tmp_path / "does-not-exist.yaml")


def test_load_candidates_rejects_non_mapping_top_level(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.dump(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="mapping"):
        load_candidates(path)


def test_load_candidates_rejects_non_list_stage_value(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.dump({"vlm": {"id": "oops"}}))
    with pytest.raises(ValueError, match="list of candidates"):
        load_candidates(path)


def test_load_candidates_rejects_entry_missing_required_field(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.dump({"vlm": [{"id": "no-source-or-license"}]}))
    with pytest.raises(ValidationError):
        load_candidates(path)


# --- runner ------------------------------------------------------------------


def test_run_benchmark_records_timing_for_a_successful_adapter():
    candidate = make_candidate()
    adapter = FakeAdapter(infer_seconds=0.001)
    result = run_benchmark(
        candidate, adapter, samples=["a", "b", "c"], device="cpu", dtype="float32"
    )

    assert result.error is None
    assert result.num_samples == 3
    assert result.load_time_s is not None and result.load_time_s >= 0.0
    assert result.latency_ms_mean is not None and result.latency_ms_mean > 0.0
    assert result.latency_ms_p95 is not None
    assert result.latency_ms_p95 >= result.latency_ms_mean
    assert adapter.loaded is True
    assert adapter.unloaded is True


def test_run_benchmark_on_cpu_leaves_peak_memory_unset():
    candidate = make_candidate()
    result = run_benchmark(candidate, FakeAdapter(), samples=[1], device="cpu", dtype="float32")
    assert result.peak_memory_mb is None


def test_run_benchmark_single_sample_p95_equals_mean():
    candidate = make_candidate()
    result = run_benchmark(candidate, FakeAdapter(), samples=[1], device="cpu", dtype="float32")
    assert result.latency_ms_p95 == pytest.approx(result.latency_ms_mean)


@pytest.mark.parametrize("fail_on", ["load", "infer"])
def test_run_benchmark_failure_is_recorded_not_raised(fail_on):
    candidate = make_candidate()
    adapter = FakeAdapter(fail_on=fail_on)

    result = run_benchmark(candidate, adapter, samples=[1], device="cpu", dtype="float32")

    assert result.error is not None
    assert "simulated" in result.error
    assert result.latency_ms_mean is None
    assert result.candidate_id == candidate.id
    assert result.stage == candidate.stage
    assert adapter.unloaded is True


def test_run_sweep_skips_candidates_without_a_registered_adapter():
    candidates = [make_candidate(id="has-adapter"), make_candidate(id="no-adapter")]
    adapters = {"has-adapter": FakeAdapter()}

    results = run_sweep(candidates, adapters, samples=[1], device="cpu", dtype="float32")

    assert [r.candidate_id for r in results] == ["has-adapter"]


def test_run_sweep_with_no_matching_adapters_returns_empty():
    candidates = [make_candidate()]
    assert run_sweep(candidates, {}, samples=[1], device="cpu", dtype="float32") == []


# --- report --------------------------------------------------------------


def make_result(**overrides) -> BenchmarkResult:
    defaults = dict(candidate_id="model-a", stage="vlm", device="cuda", dtype="float16")
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def test_render_markdown_empty_results():
    assert "No benchmark results" in render_markdown([])


def test_render_markdown_groups_by_stage():
    results = [
        make_result(candidate_id="vlm-a", stage="vlm", latency_ms_mean=10.0),
        make_result(candidate_id="seg-a", stage="segmentation", latency_ms_mean=5.0),
    ]
    output = render_markdown(results)
    assert "### segmentation" in output
    assert "### vlm" in output
    assert "vlm-a" in output
    assert "seg-a" in output


def test_render_markdown_sorts_by_latency_ascending_within_a_stage():
    results = [
        make_result(candidate_id="slow", stage="vlm", latency_ms_mean=50.0),
        make_result(candidate_id="fast", stage="vlm", latency_ms_mean=5.0),
    ]
    output = render_markdown(results)
    assert output.index("fast") < output.index("slow")


def test_render_markdown_puts_errored_candidates_last_and_shows_the_error():
    results = [
        make_result(candidate_id="broken", stage="vlm", error="OOM on load"),
        make_result(candidate_id="works", stage="vlm", latency_ms_mean=5.0),
    ]
    output = render_markdown(results)
    assert output.index("works") < output.index("broken")
    assert "OOM on load" in output
