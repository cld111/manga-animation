"""Phase 18.1: DINO candidate-recall benchmark.

Measures whether the correct target bbox exists among ALL Grounding DINO detections and how
high it is ranked by DINO confidence, for the 64 phase-17 human-annotated `body` instances.
Diagnostic only: no production code or thresholds are changed. See docs/phase18.1-results.md.
"""

from manga_animation.benchmarking.phase18.candidates import (
    RECALL_K_VALUES,
    RECALL_THRESHOLDS,
    RankedCandidate,
    RecallCurve,
    TargetRecall,
    ThresholdRecall,
    measure_target,
    rank_candidates,
    recall_curves,
)
from manga_animation.benchmarking.phase18.report import Phase18Report, build_report, write_report
from manga_animation.benchmarking.phase18.run import (
    collect_detections,
    compute_target_recall,
)

__all__ = [
    "Phase18Report",
    "RECALL_K_VALUES",
    "RECALL_THRESHOLDS",
    "RankedCandidate",
    "RecallCurve",
    "TargetRecall",
    "ThresholdRecall",
    "build_report",
    "collect_detections",
    "compute_target_recall",
    "measure_target",
    "rank_candidates",
    "recall_curves",
    "write_report",
]
