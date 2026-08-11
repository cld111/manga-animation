"""Data contracts for Phase 2 model benchmarking.

`ModelCandidate` describes a model worth measuring (loaded from
`configs/benchmark_candidates.yaml`); `BenchmarkResult` records what actually happened when
one candidate was measured on one device. Neither type runs anything — see `runner.py` for
that, and `docs/decisions/0004-phase2-model-candidates.md` for the shortlist and methodology
these feed into.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from manga_animation.core.config import DType

Stage = str


class ModelCandidate(BaseModel):
    """One model worth benchmarking for a given pipeline stage."""

    id: str = Field(min_length=1)
    stage: Stage = Field(min_length=1, description='e.g. "vlm", "grounding", "segmentation".')
    source: str = Field(min_length=1, description="Hugging Face repo id, GitHub repo, or similar.")
    license: str = Field(min_length=1)
    params: str = Field(min_length=1, description='Free-form size, e.g. "7B" or "218M".')
    notes: str = ""


class BenchmarkResult(BaseModel):
    """What happened when one `ModelCandidate` was measured on one device/dtype.

    `error` is set (and the latency/memory fields left `None`) when the candidate failed to
    load or run — a failed candidate is a valid, useful benchmark outcome, not a crash. See
    `runner.run_benchmark`.
    """

    candidate_id: str = Field(min_length=1)
    stage: Stage = Field(min_length=1)
    device: str = Field(min_length=1)
    dtype: DType

    num_samples: int = Field(ge=0, default=0)
    load_time_s: float | None = Field(default=None, ge=0.0)
    latency_ms_mean: float | None = Field(default=None, ge=0.0)
    latency_ms_p95: float | None = Field(default=None, ge=0.0)
    peak_memory_mb: float | None = Field(default=None, ge=0.0)

    quality_score: float | None = Field(
        default=None, description="Optional automated score; manga-domain fit is usually a note."
    )
    quality_notes: str | None = Field(
        default=None, description="Qualitative spot-check observations (see ADR 0004)."
    )

    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
