"""Phase 9: the human/AI visual-quality scoring protocol (brief sections 8-13) -- a fixed,

typed rubric so a video's visual quality is scored the same way every time, not judged by a
vague "looks good" impression. Reuses the project's existing pipeline-failure vocabulary
(`pipeline.types.Stage`, `evaluation.metrics.E2EStatus`) for technical/attributed failures --
this module only adds what's genuinely new: a VISUAL failure taxonomy and a 0-5 quality rubric,
both of which only apply to a sample that actually produced a video to look at.

**Reproducibility, not automation**: none of this replaces direct visual inspection of the
actual rendered frames -- these types exist so that inspection produces a structured, comparable
record instead of unstructured prose, and so the resulting scores can be aggregated
(`capability_matrix`) without re-reading every note by hand.

**Inter-rater reliability (brief section 10)**: this project has exactly one available
evaluator for Phase 9 (the Claude Code assistant performing direct visual inspection, same
methodology as the Real-World Evaluation Dataset's own ground-truth labels) -- `evaluator` is
recorded on every score specifically so this limitation is visible in the data itself, not just
asserted in prose. No inter-rater agreement statistic is computed because there is only one
rater; this is a real, disclosed limitation (see `docs/phase9-results.md`), not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from pydantic import BaseModel, Field

VisualQADimension = Literal[
    "target_correctness",
    "motion_correctness",
    "motion_quality",
    "mask_quality",
    "background_preservation",
    "compositing_quality",
    "artwork_preservation",
    "loop_quality",
]
"""The Phase 9 brief's section 8 (A-H) visual-quality dimensions, verbatim."""

VISUAL_QA_DIMENSIONS: tuple[VisualQADimension, ...] = get_args(VisualQADimension)

VisualQAScoreValue = Literal[0, 1, 2, 3, 4, 5]

VISUAL_QA_SCALE: dict[VisualQAScoreValue, str] = {
    0: "unusable -- the dimension being scored is completely broken or absent (e.g. the wrong "
    "object entirely, or the composited frame is visibly corrupted)",
    1: "major problems -- clearly, immediately visible on a first look, would make a viewer "
    "question whether the tool works at all",
    2: "noticeable problems -- visible on direct inspection, would bother an attentive viewer "
    "but the output is still recognizably functional",
    3: "acceptable -- a careful look finds real, minor issues, but a casual viewer would not "
    "be bothered",
    4: "good -- no real issues found on direct inspection; only a very fine-grained comparison "
    "against the source artwork would find anything to note",
    5: "excellent -- indistinguishable from a hand-crafted result on this dimension",
}
"""Fixed score definitions (brief section 9) -- every score in this project must be justified

against this exact text, not a personal/vague sense of "good" vs "bad"."""

VisualFailureCategory = Literal[
    "wrong_target",
    "bad_mask",
    "duplicate_silhouette",
    "seam",
    "ghosting",
    "background_corruption",
    "incorrect_occlusion",
    "excessive_motion",
    "insufficient_motion",
    "unnatural_motion",
    "loop_discontinuity",
]
"""The Phase 9 brief's section 12 VISUAL failure taxonomy, verbatim -- deliberately separate

from `pipeline.types.Stage` (which already, adequately, categorizes PIPELINE failures --
analysis/grounding/segmentation/validation/rendering -- see `PageRunOutcome.failing_stage`;
reused as-is here, not duplicated). This taxonomy only applies to a sample that reached a real
composited video -- a REJECTED/ERROR outcome with no render has nothing visual to categorize
this way."""

VISUAL_FAILURE_CATEGORIES: tuple[VisualFailureCategory, ...] = get_args(VisualFailureCategory)


class VisualQAScore(BaseModel):
    """One evaluator's structured visual-quality judgment of one real rendered sample/mode --

    the Phase 9 brief's required "fixed criteria; consistent interpretation; reproducibility;
    explicit definitions; no vague 'looks good' judgments" (section 9), made into data.
    """

    sample_id: str
    analysis_mode: Literal["page", "panel"]
    evaluator: str = Field(
        description='Who produced this score, e.g. "claude (Phase 9 direct visual '
        'inspection)" -- always recorded, since this project currently has exactly one '
        "evaluator (see this module's own docstring)."
    )
    evaluated_at: str = Field(description="ISO-8601 timestamp of when this score was recorded.")
    has_video: bool = Field(
        description="False for a REJECTED/ERROR outcome with no rendered video at all -- every "
        "score field below is then None (nothing to look at), not a fabricated 0."
    )
    target_correctness: VisualQAScoreValue | None = None
    motion_correctness: VisualQAScoreValue | None = None
    motion_quality: VisualQAScoreValue | None = None
    mask_quality: VisualQAScoreValue | None = None
    background_preservation: VisualQAScoreValue | None = None
    compositing_quality: VisualQAScoreValue | None = None
    artwork_preservation: VisualQAScoreValue | None = None
    loop_quality: VisualQAScoreValue | None = None
    failure_categories: list[VisualFailureCategory] = Field(
        default_factory=list,
        description="Every VisualFailureCategory actually observed on direct inspection -- "
        "empty means none were found, not that inspection didn't happen (see notes for that).",
    )
    notes: str = Field(
        default="",
        description="Free-text justification -- what was actually seen, citing "
        "specific frames/regions, not a restatement of the numeric scores.",
    )

    @property
    def dimension_scores(self) -> dict[VisualQADimension, VisualQAScoreValue | None]:
        return {
            "target_correctness": self.target_correctness,
            "motion_correctness": self.motion_correctness,
            "motion_quality": self.motion_quality,
            "mask_quality": self.mask_quality,
            "background_preservation": self.background_preservation,
            "compositing_quality": self.compositing_quality,
            "artwork_preservation": self.artwork_preservation,
            "loop_quality": self.loop_quality,
        }

    @property
    def mean_score(self) -> float | None:
        """Mean across every dimension that was actually scored -- `None` when `has_video` is

        False (nothing was scored) or, defensively, if every individual score is somehow still
        `None` despite `has_video=True`."""
        scored = [v for v in self.dimension_scores.values() if v is not None]
        return sum(scored) / len(scored) if scored else None


CapabilityDimension = Literal[
    "single_object",
    "multiple_objects",
    "translation",
    "rotation",
    "scale",
    "occlusion",
    "boundary_objects",
    "complex_background",
    "hair_or_clothing",
    "weapons",
    "effects",
    "dense_scenes",
]
"""The Phase 9 brief's section 13 example capability-matrix rows, verbatim."""

CAPABILITY_DIMENSIONS: tuple[CapabilityDimension, ...] = get_args(CapabilityDimension)

CapabilityVerdict = Literal["WORKS_WELL", "PARTIAL", "FAILS", "UNKNOWN"]
"""`UNKNOWN` is the honest default (brief: "Mark unknown capability as UNKNOWN, not PASS") --

every other verdict must be backed by cited real evidence, never assumed."""


@dataclass(frozen=True, slots=True)
class CapabilityMatrixEntry:
    """One row of the Phase 9 capability matrix -- a verdict plus the real evidence

    (sample_ids, not vibes) it rests on. Constructing this by hand (or letting `dimension`
    default to `UNKNOWN` with empty evidence) is the only way a dimension enters the matrix --
    there is no automatic verdict-assignment function, deliberately: turning pipeline stats and
    visual QA scores into a WORKS_WELL/PARTIAL/FAILS judgment is exactly the kind of call the
    brief says must be evidence-driven, not formulaic.
    """

    dimension: CapabilityDimension
    verdict: CapabilityVerdict
    evidence_sample_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.verdict != "UNKNOWN" and not self.evidence_sample_ids:
            raise ValueError(
                f"capability entry {self.dimension!r} has verdict {self.verdict!r} but cites "
                "no evidence_sample_ids -- every non-UNKNOWN verdict must be backed by at "
                "least one real sample_id"
            )


def build_capability_matrix(
    entries: list[CapabilityMatrixEntry],
) -> dict[CapabilityDimension, CapabilityMatrixEntry]:
    """Every `CAPABILITY_DIMENSIONS` value, defaulted to `UNKNOWN` with no evidence unless an

    explicit entry overrides it -- raises `ValueError` on a duplicate dimension (an accidental
    double-assignment silently overwriting the first is worse than failing loudly) or an unknown
    dimension name (a typo must not silently create an uncounted row, same discipline as
    `GoldenCategory`/`golden_category_coverage`).
    """
    matrix: dict[CapabilityDimension, CapabilityMatrixEntry] = {
        d: CapabilityMatrixEntry(dimension=d, verdict="UNKNOWN") for d in CAPABILITY_DIMENSIONS
    }
    seen: set[CapabilityDimension] = set()
    for entry in entries:
        if entry.dimension not in CAPABILITY_DIMENSIONS:
            raise ValueError(f"unknown capability dimension: {entry.dimension!r}")
        if entry.dimension in seen:
            raise ValueError(f"duplicate capability entry for dimension: {entry.dimension!r}")
        seen.add(entry.dimension)
        matrix[entry.dimension] = entry
    return matrix
