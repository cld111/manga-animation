"""Phase 2 model benchmarking: candidate manifest, timing harness, and comparison reports.

See docs/decisions/0004-phase2-model-candidates.md. No model-loading dependency (torch,
transformers, ...) is imported by this package itself — `runner.ModelAdapter` is the seam
Phase 2 integration code implements per selected candidate.
"""

from manga_animation.benchmarking.registry import flat_candidates, load_candidates
from manga_animation.benchmarking.report import render_markdown
from manga_animation.benchmarking.runner import ModelAdapter, run_benchmark, run_sweep
from manga_animation.benchmarking.schemas import BenchmarkResult, ModelCandidate

__all__ = [
    "BenchmarkResult",
    "ModelAdapter",
    "ModelCandidate",
    "flat_candidates",
    "load_candidates",
    "render_markdown",
    "run_benchmark",
    "run_sweep",
]
