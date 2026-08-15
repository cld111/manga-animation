"""Phase 17 object-segmentation diagnostic benchmark.

Answers: how well does the production Grounding DINO -> SAM 2.1 pipeline segment ordinary
objects in manga artwork, and where does the observed degradation actually come from
(grounding localization vs. SAM segmentation vs. candidate selection vs. post-processing)?

Diagnostic only: nothing here modifies the production pipeline. It measures the real
production models/clients with the real thresholds, the real candidate ranking, and the real
deterministic mask post-processing, in three independent experiments (see `run.py` and
`docs/phase17-results.md`).
"""

from manga_animation.benchmarking.phase17.dataset import (
    CATEGORY_IDS,
    CATEGORY_NAMES,
    FORBIDDEN_CATEGORIES,
    MAIN_OBJECT_CATEGORY,
    CandidateInstance,
    MangaSegSample,
    candidate_pool_from_ms92,
    decode_rle,
    materialize_samples,
    tight_bbox_from_mask,
    verify_ms92_vs_mirror,
)
from manga_animation.benchmarking.phase17.manifest import (
    BenchmarkManifest,
    ManifestSample,
    build_manifest,
    load_manifest,
    sample_prompt,
    select_samples,
    write_manifest,
)
from manga_animation.benchmarking.phase17.metrics import (
    Distribution,
    MaskMetrics,
    bbox_area_ratio,
    bbox_gt_coverage,
    bbox_iou,
    compute_distribution,
    mask_metrics,
)
from manga_animation.benchmarking.phase17.report import (
    BenchmarkReport,
    PerSampleMetrics,
    aggregate,
    build_visual_failures,
    compute_forbidden_overlap,
    write_report,
)
from manga_animation.benchmarking.phase17.run import (
    SampleResult,
    run_benchmark_experiments,
)

__all__ = [
    "BenchmarkManifest",
    "BenchmarkReport",
    "CATEGORY_IDS",
    "CATEGORY_NAMES",
    "CandidateInstance",
    "Distribution",
    "FORBIDDEN_CATEGORIES",
    "MAIN_OBJECT_CATEGORY",
    "ManifestSample",
    "MaskMetrics",
    "MangaSegSample",
    "PerSampleMetrics",
    "SampleResult",
    "aggregate",
    "bbox_area_ratio",
    "bbox_gt_coverage",
    "bbox_iou",
    "build_manifest",
    "build_visual_failures",
    "candidate_pool_from_ms92",
    "compute_distribution",
    "compute_forbidden_overlap",
    "decode_rle",
    "load_manifest",
    "mask_metrics",
    "materialize_samples",
    "run_benchmark_experiments",
    "sample_prompt",
    "select_samples",
    "tight_bbox_from_mask",
    "verify_ms92_vs_mirror",
    "write_manifest",
    "write_report",
]
