"""Phase 18.2A: Qwen2.5-VL direct target localization benchmark.

Answers (docs/phase18.2a-qwen-bbox-results.md): can Qwen2.5-VL, given a full manga page and
the production target description, localize the SPECIFIC target instance well enough to either
replace or guide the DINO->candidate-selection path? Measured on the same 64 Phase 17/18.1
human-annotated `body` targets, with three comparable signals: Qwen direct bbox (vs GT bbox),
Qwen bbox -> SAM -> mask (vs GT mask), and GT bbox -> SAM -> mask (the Phase 17 reference).

Diagnostic only: no production code, prompts, thresholds, or gates are changed. See the
`coords.py` docstring for the coordinate contract this phase pins down and unit-tests.
"""

from manga_animation.benchmarking.phase18a.classify import (
    ERROR_CATEGORY_NAMES,
    Classification,
    classify,
)
from manga_animation.benchmarking.phase18a.coords import (
    EDGE_TOLERANCE_FRACTION,
    BBox,
    QwenBboxPrediction,
    bbox_from_response,
    clamp_box,
    convert_prediction,
    extract_json_object,
    parse_direct_response,
)
from manga_animation.benchmarking.phase18a.metrics import (
    PerTargetMetrics,
    Phase18aMetrics,
    compute_metrics,
)
from manga_animation.benchmarking.phase18a.prompt import build_direct_prompt
from manga_animation.benchmarking.phase18a.report import (
    DINO_RALL_RECALL,
    DINO_TOP1_RECALL,
    PHASE17_GT_SAM_MEDIAN_IOU,
    Phase18aReport,
    build_report,
    write_report,
)
from manga_animation.benchmarking.phase18a.run import (
    DirectLocalizationRecord,
    build_per_target_metrics,
    collect_direct_predictions,
    collect_sam_masks,
)
from manga_animation.benchmarking.phase18a.visuals import build_visual_packages

__all__ = [
    "BBox",
    "Classification",
    "DINO_RALL_RECALL",
    "DINO_TOP1_RECALL",
    "DirectLocalizationRecord",
    "EDGE_TOLERANCE_FRACTION",
    "ERROR_CATEGORY_NAMES",
    "PHASE17_GT_SAM_MEDIAN_IOU",
    "PerTargetMetrics",
    "Phase18aMetrics",
    "Phase18aReport",
    "QwenBboxPrediction",
    "bbox_from_response",
    "build_direct_prompt",
    "build_per_target_metrics",
    "build_report",
    "build_visual_packages",
    "classify",
    "clamp_box",
    "collect_direct_predictions",
    "collect_sam_masks",
    "compute_metrics",
    "convert_prediction",
    "extract_json_object",
    "parse_direct_response",
    "write_report",
]
