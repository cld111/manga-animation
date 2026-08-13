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

from manga_animation.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    GOLDEN_DATASET_CATEGORIES,
    EvalSample,
    GoldenCategory,
    golden_category_coverage,
    load_eval_dataset,
    uncovered_golden_categories,
)
from manga_animation.evaluation.metrics import (
    E2EStatus,
    EvaluationReport,
    Rate,
    StatusBreakdown,
    classify_outcome,
    compute_metrics,
)
from manga_animation.evaluation.nondeterminism import (
    NondeterminismSummary,
    RepeatedRunRecord,
    summarize_repeated_runs,
)
from manga_animation.evaluation.schemas import (
    LoopMetricsOutcome,
    ObjectAttemptOutcome,
    PageRunOutcome,
    RenderSummary,
    ValidationAttemptOutcome,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "GOLDEN_DATASET_CATEGORIES",
    "E2EStatus",
    "EvalSample",
    "EvaluationReport",
    "GoldenCategory",
    "LoopMetricsOutcome",
    "NondeterminismSummary",
    "ObjectAttemptOutcome",
    "PageRunOutcome",
    "Rate",
    "RenderSummary",
    "RepeatedRunRecord",
    "StatusBreakdown",
    "ValidationAttemptOutcome",
    "classify_outcome",
    "compute_metrics",
    "golden_category_coverage",
    "load_eval_dataset",
    "summarize_repeated_runs",
    "uncovered_golden_categories",
]
