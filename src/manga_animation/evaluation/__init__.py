"""Phase 3.3 evaluation framework: a real-sample dataset (`dataset.py`), typed run-outcome

records (`schemas.py`), reproducible metrics with denominators (`metrics.py`), VLM run-to-run
nondeterminism measurement (`nondeterminism.py`), and (Phase 9) a validated visual-artifact
signal (`artifacts.py`). See docs/phase3.3-results.md/docs/phase9-results.md for the actual
evaluation runs this package's output feeds, and `scripts/run_phase3_3_evaluation.py`/
`scripts/run_phase9_evaluation.py` for the real (remote-GPU) driver scripts.

Nothing here imports torch/transformers -- this package computes metrics FROM run outcomes, it
does not run the pipeline itself (that stays the two scripts above's job, per CLAUDE.md's
"pipeline is not run locally" policy) -- so it is fully unit-testable with plain fixtures, like
every other stage in this project. `artifacts.py` imports `cv2` (this project's other,
already-ubiquitous, non-`ml` optional dependency tier -- same as `rendering`/`compositing`), not
torch, so this promise still holds. The one real exception is `harness.py` (the piece that
actually calls `pipeline.orchestrator.run_pipeline`) -- deliberately NOT imported here; see its
own module docstring for why.
"""

from manga_animation.evaluation.artifacts import (
    ChangedRegionShape,
    SeamArtifactReport,
    detect_changed_region_shapes,
    detect_seam_like_artifacts,
)
from manga_animation.evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    GEOMETRIC_DIFFICULTY_TAGS,
    GOLDEN_DATASET_CATEGORIES,
    MOTION_TYPE_TAGS,
    POTENTIAL_MOTION_TAGS,
    REALWORLD_DATASET_PATH,
    SCENE_COMPLEXITY_TAGS,
    DifficultyLevel,
    EvalSample,
    GeometricDifficultyTag,
    GoldenCategory,
    MotionTypeTag,
    PotentialMotionTag,
    SceneComplexityTag,
    dataset_composition,
    golden_category_coverage,
    load_combined_eval_dataset,
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
from manga_animation.evaluation.visual_qa import (
    CAPABILITY_DIMENSIONS,
    VISUAL_FAILURE_CATEGORIES,
    VISUAL_QA_DIMENSIONS,
    VISUAL_QA_SCALE,
    CapabilityDimension,
    CapabilityMatrixEntry,
    CapabilityVerdict,
    VisualFailureCategory,
    VisualQADimension,
    VisualQAScore,
    VisualQAScoreValue,
    build_capability_matrix,
)

__all__ = [
    "CAPABILITY_DIMENSIONS",
    "ChangedRegionShape",
    "SeamArtifactReport",
    "DEFAULT_DATASET_PATH",
    "GEOMETRIC_DIFFICULTY_TAGS",
    "GOLDEN_DATASET_CATEGORIES",
    "MOTION_TYPE_TAGS",
    "POTENTIAL_MOTION_TAGS",
    "REALWORLD_DATASET_PATH",
    "SCENE_COMPLEXITY_TAGS",
    "VISUAL_FAILURE_CATEGORIES",
    "VISUAL_QA_DIMENSIONS",
    "VISUAL_QA_SCALE",
    "CapabilityDimension",
    "CapabilityMatrixEntry",
    "CapabilityVerdict",
    "DifficultyLevel",
    "E2EStatus",
    "EvalSample",
    "EvaluationReport",
    "GeometricDifficultyTag",
    "GoldenCategory",
    "LoopMetricsOutcome",
    "MotionTypeTag",
    "NondeterminismSummary",
    "ObjectAttemptOutcome",
    "PageRunOutcome",
    "PotentialMotionTag",
    "Rate",
    "RenderSummary",
    "RepeatedRunRecord",
    "SceneComplexityTag",
    "StatusBreakdown",
    "ValidationAttemptOutcome",
    "VisualFailureCategory",
    "VisualQADimension",
    "VisualQAScore",
    "VisualQAScoreValue",
    "build_capability_matrix",
    "classify_outcome",
    "compute_metrics",
    "dataset_composition",
    "detect_changed_region_shapes",
    "detect_seam_like_artifacts",
    "golden_category_coverage",
    "load_combined_eval_dataset",
    "load_eval_dataset",
    "summarize_repeated_runs",
    "uncovered_golden_categories",
]
