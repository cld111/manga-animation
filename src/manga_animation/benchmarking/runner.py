"""Times a `ModelCandidate` against sample inputs, producing a `BenchmarkResult`.

No model-loading code lives here. `ModelAdapter` is the seam: Phase 2 integration work
implements one adapter per candidate actually worth running on the remote GPU, and this
module stays ignorant of which library (transformers, ultralytics, a custom repo...) that
adapter wraps. See `docs/decisions/0004-phase2-model-candidates.md`.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from typing import Any, Protocol

from manga_animation.benchmarking.schemas import BenchmarkResult, ModelCandidate
from manga_animation.core.config import DType
from manga_animation.core.logging import get_logger

logger = get_logger(__name__)


class ModelAdapter(Protocol):
    """What `run_benchmark` needs from a model wrapper to time it."""

    def load(self) -> None:
        """Load weights and move the model onto its target device."""
        ...

    def infer(self, sample: Any) -> Any:
        """Run one inference call. The return value is not inspected by the runner."""
        ...

    def unload(self) -> None:
        """Release the model (free VRAM/unified memory) before the next candidate loads."""
        ...


def run_benchmark(
    candidate: ModelCandidate,
    adapter: ModelAdapter,
    samples: Sequence[Any],
    *,
    device: str,
    dtype: DType,
) -> BenchmarkResult:
    """Time one candidate's load + per-sample inference.

    Unlike `StageTimer` (which never swallows a pipeline stage's exceptions), a failing
    candidate here is recorded as a `BenchmarkResult` with `error` set rather than raised:
    the point of sweeping N candidates is a comparison table even when one OOMs or fails to
    load, not an all-or-nothing run.
    """
    _reset_peak_memory(device)
    try:
        load_start = time.perf_counter()
        adapter.load()
        load_time_s = time.perf_counter() - load_start

        latencies_ms: list[float] = []
        for sample in samples:
            start = time.perf_counter()
            adapter.infer(sample)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        peak_memory_mb = _peak_memory_mb(device)
        adapter.unload()

        return BenchmarkResult(
            candidate_id=candidate.id,
            stage=candidate.stage,
            device=device,
            dtype=dtype,
            num_samples=len(samples),
            load_time_s=load_time_s,
            latency_ms_mean=statistics.mean(latencies_ms) if latencies_ms else None,
            latency_ms_p95=_percentile(latencies_ms, 0.95) if latencies_ms else None,
            peak_memory_mb=peak_memory_mb,
        )
    except Exception as exc:
        logger.error(
            "benchmark failed candidate_id=%s stage=%s device=%s error=%s",
            candidate.id,
            candidate.stage,
            device,
            exc,
        )
        return BenchmarkResult(
            candidate_id=candidate.id,
            stage=candidate.stage,
            device=device,
            dtype=dtype,
            error=str(exc),
        )


def run_sweep(
    candidates: Sequence[ModelCandidate],
    adapters: dict[str, ModelAdapter],
    samples: Sequence[Any],
    *,
    device: str,
    dtype: DType,
) -> list[BenchmarkResult]:
    """`run_benchmark` for each candidate that has a matching entry in `adapters` (by id).

    Candidates with no adapter registered are skipped, not errored — Phase 2 typically has
    working adapters for only a subset of the shortlist at any given time.
    """
    results = []
    for candidate in candidates:
        adapter = adapters.get(candidate.id)
        if adapter is None:
            logger.info("skipping candidate_id=%s: no adapter registered", candidate.id)
            continue
        results.append(run_benchmark(candidate, adapter, samples, device=device, dtype=dtype))
    return results


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = p * (len(ordered) - 1)
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def _reset_peak_memory(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str) -> float | None:
    if device != "cuda":
        # MPS/CPU have no reliable cross-run peak-tracking API; a point-in-time sample is
        # available via core.logging.get_gpu_memory_mb() but would understate a real peak, so
        # it's left as None here rather than reported misleadingly.
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)
