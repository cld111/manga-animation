"""Phase 3.3 evaluation framework: a real-sample dataset (`dataset.py`), typed run-outcome

records (`schemas.py`), reproducible metrics with denominators (`metrics.py`), and VLM
run-to-run nondeterminism measurement (`nondeterminism.py`). See docs/phase3.3-results.md for
the actual evaluation run this package's output feeds, and
`scripts/run_phase3_3_evaluation.py` for the real (remote-GPU) driver script.

Nothing here imports torch/transformers -- this package computes metrics FROM run outcomes, it
does not run the pipeline itself (that stays `scripts/run_phase3_3_evaluation.py`'s job, per
CLAUDE.md's "pipeline is not run locally" policy) -- so it is fully unit-testable with plain
fixtures, like every other stage in this project.
"""

from manga_animation.evaluation.dataset import DEFAULT_DATASET_PATH, EvalSample, load_eval_dataset
from manga_animation.evaluation.metrics import EvaluationReport, Rate, compute_metrics
from manga_animation.evaluation.nondeterminism import (
    NondeterminismSummary,
    RepeatedRunRecord,
    summarize_repeated_runs,
)
from manga_animation.evaluation.schemas import (
    ObjectAttemptOutcome,
    PageRunOutcome,
    ValidationAttemptOutcome,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "EvalSample",
    "EvaluationReport",
    "NondeterminismSummary",
    "ObjectAttemptOutcome",
    "PageRunOutcome",
    "Rate",
    "RepeatedRunRecord",
    "ValidationAttemptOutcome",
    "compute_metrics",
    "load_eval_dataset",
    "summarize_repeated_runs",
]
