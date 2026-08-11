"""Formats `BenchmarkResult`s into a human-readable comparison table.

Pure formatting, no I/O — callers decide whether the output goes to a file, stdout, or a
PR description. Kept separate from `runner.py` so results collected across multiple sweep
runs (e.g. T4 today, L4 tomorrow) can be combined and reported together.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from manga_animation.benchmarking.schemas import BenchmarkResult

_HEADERS = (
    "Candidate",
    "Device",
    "Dtype",
    "Load (s)",
    "Latency mean (ms)",
    "Latency p95 (ms)",
    "Peak mem (MB)",
    "Status",
)


def render_markdown(results: Sequence[BenchmarkResult]) -> str:
    """One markdown table per stage, sorted by mean latency (failed candidates listed last)."""
    if not results:
        return "_No benchmark results._\n"

    by_stage: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        by_stage[result.stage].append(result)

    sections = []
    for stage in sorted(by_stage):
        ordered = sorted(by_stage[stage], key=_sort_key)
        sections.append(f"### {stage}\n\n{_table(ordered)}")
    return "\n\n".join(sections) + "\n"


def _sort_key(result: BenchmarkResult) -> tuple[int, float]:
    if result.error is not None or result.latency_ms_mean is None:
        return (1, 0.0)
    return (0, result.latency_ms_mean)


def _table(results: Sequence[BenchmarkResult]) -> str:
    rows = [_HEADERS, tuple("---" for _ in _HEADERS)]
    for result in results:
        rows.append(_row(result))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _row(result: BenchmarkResult) -> tuple[str, ...]:
    status = f"error: {result.error}" if result.error else "ok"
    return (
        result.candidate_id,
        result.device,
        result.dtype,
        _fmt(result.load_time_s, "{:.2f}"),
        _fmt(result.latency_ms_mean, "{:.1f}"),
        _fmt(result.latency_ms_p95, "{:.1f}"),
        _fmt(result.peak_memory_mb, "{:.0f}"),
        status,
    )


def _fmt(value: float | None, spec: str) -> str:
    return spec.format(value) if value is not None else "—"
