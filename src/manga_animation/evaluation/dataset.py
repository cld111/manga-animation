"""The Phase 3.3 evaluation dataset: real manga pages with honestly-establishable ground truth.

See `docs/phase3.3-results.md` for the actual dataset write-up and
`configs/phase3_3_eval_dataset.yaml` for the data itself -- this module only defines the
schema and the loader. Per the Phase 3.3 brief: "Do NOT fabricate ground truth." Every field
below is optional (`None`) except `sample_id`/`image_path`/`diversity_tag`/`source_citation` --
a sample with genuinely uncertain ground truth leaves the uncertain fields `None` rather than
guessing, and `ground_truth_uncertain=True` makes that explicit instead of silently omitting it.
`fetch_script` is also optional (Pre-Phase-3.4 "verified action" integration): some samples are
manually provided by the project owner with no reproducible fetch mechanism at all, which is a
distinct, honestly-recorded state from "has a fetch script that happens not to be
deterministic" (`sample_page_01`/`sample_page_02`'s pre-existing, different gap).

**Ground truth vs. prediction (Phase 3.3.2, see ADR 0009)**: `EvalSample` is this project's
ONLY representation of evaluation ground truth, and it is real evidence, established by direct
human/visual inspection of the source artwork (or, where that evidence doesn't exist or has
since been contradicted, honestly marked `ground_truth_uncertain=True` -- never silently
guessed). It is deliberately `frozen=True`: nothing in this codebase's normal operation --
least of all a VLM inference call -- may mutate a loaded `EvalSample` in place. A real, observed
failure motivated this: `sample_page_02`'s `animation_possible` was originally set to `"yes"` on
the strength of a single Phase 3.2 VLM classification (later confirmed only by a pixel-diff of
that classification's own downstream render, not by an independent re-examination of the source
page's actual drawn motion cues) -- two later, independent real sessions (Phase 3.3, Phase
3.3.1) had the same VLM read the same page as all-STATIC. Ground truth predicated on a single
VLM read, even one a later sanity check found visually plausible, is not independent evidence;
see this module's own `sample_page_02` entry in `configs/phase3_3_eval_dataset.yaml` for how
that specific case was resolved. `PageRunOutcome`/`RepeatedRunRecord` (`schemas.py`/
`nondeterminism.py`) are the separate, mutable-by-construction PREDICTION types real pipeline
runs produce -- `evaluation/metrics.py::compute_metrics` always compares a prediction against
this module's stored ground truth, never the reverse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "phase3_3_eval_dataset.yaml"
)
REALWORLD_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "phase9_realworld_eval_dataset.yaml"
)
"""Phase 9: the Real-World Evaluation Dataset -- a separate, larger, characterization-focused

manifest (docs/decisions/0016-phase9-realworld-evaluation.md), reusing this exact `EvalSample`
schema rather than fabricating a parallel one. Deliberately a different file from
`DEFAULT_DATASET_PATH` (the Phase 8 golden regression set): the brief that motivated this is
explicit that the 7-sample golden set is "primarily designed for regression/safety validation"
and must not be treated as "a statistically meaningful real-world quality benchmark" -- see
`load_combined_eval_dataset` for how the two are used together without duplicating either."""

AnimationPossible = Literal["yes", "no", "uncertain"]
"""Whether a real, visually-justified animation target exists on this page, per human/AI visual

inspection against the `manga-analysis` skill's STATIC vs. ANIMATED checklist -- "uncertain" is
a first-class value, not a missing one, for pages where this project's own real evidence (see
docs/phase3.2-results.md's VLM nondeterminism finding) shows the honest answer varies."""

AnnotationProvenance = Literal["independent_human_verification"]
"""How a sample's ground truth was actually established, when that provenance is structured

enough to be worth recording as data rather than only prose in `notes` (Pre-Phase-3.4 "verified
action" integration, see docs/decisions/0009-evaluation-ground-truth-integrity.md's revision).
`None` (the default) means no structured provenance was recorded -- most pre-existing samples
predate this field and are not retroactively assigned a value here, since doing so would assert
a historical claim about how confidently their labels were established that this project cannot
actually verify after the fact; their real provenance remains in `notes`' free text instead.
The one real value that exists so far: `"independent_human_verification"` -- the project owner
directly confirmed action/animation presence, independent of any VLM inference or of inspecting
a pipeline's own downstream render (the exact failure `sample_page_02`'s original, since-revised
annotation did not avoid -- see this module's top-level docstring)."""


GoldenCategory = Literal[
    "single_animatable_object",
    "multiple_animatable_objects",
    "partially_occluded_object",
    "object_near_boundary",
    "complex_background",
    "weapon_or_effect",
    "rotation",
    "translation",
    "scale_or_deformation",
    "should_not_animate",
]
"""Phase 8: the 10 failure-mode/coverage categories the Phase 8 brief's golden E2E dataset must

represent (its section 6, items 1-10). A closed `Literal`, not a free-form string, so a typo in
`configs/phase3_3_eval_dataset.yaml`'s `golden_categories` fails loudly at load time rather than
silently creating an uncounted category -- see `golden_category_coverage`."""

GOLDEN_DATASET_CATEGORIES: tuple[GoldenCategory, ...] = get_args(GoldenCategory)

SceneComplexityTag = Literal[
    "single_character",
    "multiple_characters",
    "multiple_panels",
    "crowded_scene",
    "sparse_scene",
    "complex_background",
    "simple_background",
]
"""Phase 9 (docs/phase9-results.md): the Real-World Evaluation Dataset's 'scene complexity'

coverage dimension, verbatim from the Phase 9 brief's section 4 -- a closed `Literal`, same
typo-safety rationale as `GoldenCategory`. Descriptive dataset-composition metadata, not ground
truth -- see `EvalSample.scene_complexity_tags`."""

PotentialMotionTag = Literal[
    "character_movement",
    "weapon",
    "hair_or_clothing",
    "facial_feature",
    "environmental_effect",
    "impact_or_action_effect",
    "object_moving_across_scene",
]
"""Phase 9 brief section 4's 'potential motion' coverage dimension."""

GeometricDifficultyTag = Literal[
    "near_boundary",
    "partially_occluded",
    "overlapping_objects",
    "thin_structure",
    "irregular_silhouette",
    "large_rectangular_object",
    "complex_internal_holes",
]
"""Phase 9 brief section 4's 'geometric difficulty' coverage dimension."""

MotionTypeTag = Literal["translation", "rotation", "scale", "deformation", "mixed_motion"]
"""Phase 9 brief section 4's 'motion types' coverage dimension -- deliberately a separate,

finer taxonomy from `GoldenCategory`'s `rotation`/`translation`/`scale_or_deformation` (which
exist to prove the Phase 8 golden regression set's own 10-category coverage, not to
characterize a page's potential motion before the pipeline ever runs on it)."""

DifficultyLevel = Literal["easy", "medium", "hard"]

SCENE_COMPLEXITY_TAGS: tuple[SceneComplexityTag, ...] = get_args(SceneComplexityTag)
POTENTIAL_MOTION_TAGS: tuple[PotentialMotionTag, ...] = get_args(PotentialMotionTag)
GEOMETRIC_DIFFICULTY_TAGS: tuple[GeometricDifficultyTag, ...] = get_args(GeometricDifficultyTag)
MOTION_TYPE_TAGS: tuple[MotionTypeTag, ...] = get_args(MotionTypeTag)


class EvalSample(BaseModel):
    """One evaluation sample: a real page plus only the ground truth that can actually be

    established about it (see the Phase 3.3 brief's explicit list: animation_possible, expected
    target category, expected region if known, expected motion category, acceptable outcome).

    Frozen (Phase 3.3.2, ADR 0009): ground truth must be immutable during normal evaluation.
    Changing an annotation means editing `configs/phase3_3_eval_dataset.yaml` and bumping
    `annotation_version` -- an explicit, reviewed, git-tracked change -- never mutating a loaded
    instance in place.
    """

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1, description="Path relative to the repo root.")
    source_citation: str = Field(
        min_length=1, description="Series/chapter/page and MangaDex ids, for reproducibility."
    )
    diversity_tag: str = Field(
        min_length=1,
        description='e.g. "static", "hair", "weapon_action_effects" -- see the Phase 3.3 '
        "brief's requested diversity categories.",
    )
    fetch_script: str | None = Field(
        default=None,
        description="The scripts/fetch_*.py that reproduces image_path on demand. `None` means "
        "no such mechanism exists at all -- the file was manually provided (see "
        "source_citation/notes for how to obtain it) rather than fetched -- distinct from a "
        "recorded-but-not-actually-reproducible fetch_script value.",
    )

    animation_possible: AnimationPossible = "uncertain"
    expected_target_category: str | None = Field(
        default=None, description='e.g. "hair", "weapon" -- only set when real evidence exists.'
    )
    expected_motion_category: str | None = Field(
        default=None, description='motion_type if known, e.g. "primary"/"secondary".'
    )
    expected_region_note: str | None = Field(
        default=None,
        description="A qualitative description only (e.g. 'the character's hair, "
        "upper-left panel') -- deliberately NOT a pixel bbox, since no independently "
        "measured ground-truth region exists for any sample.",
    )
    acceptable_outcome: str = Field(
        min_length=1, description="What a correct pipeline result looks like for this sample."
    )
    regression_reference: str | None = Field(
        default=None,
        description="A specific, known-bad outcome this sample must never reproduce (e.g. the "
        "Phase 3.1 flag_banner-grounds-to-a-face failure) -- checked by the evaluation harness "
        "as a semantic false-positive when applicable, not merely narrated.",
    )
    ground_truth_uncertain: bool = Field(
        default=False,
        description="True when even the fields above are a best-effort read, not a confident "
        "one -- e.g. sample_page_01.png's real, evidenced VLM nondeterminism.",
    )
    honest_failure_acceptable: bool = Field(
        default=False,
        description="True when this sample's own acceptable_outcome explicitly allows an "
        "honest, attributed grounding/validation failure (no video) as a good result, even "
        "though animation_possible='yes' and ground_truth_uncertain=False -- distinct from "
        "ground_truth_uncertain, which means the *ground truth itself* is unsettled. Here the "
        "ground truth is confident (something real and animatable IS present), but the target "
        "is inherently hard to ground (e.g. an effect-heavy motion cue with no single concrete "
        "object), so a correct pipeline can still legitimately fail to find/validate it. Only "
        "an *attributed* failure (PageRunOutcome.failing_stage is not None) counts as honest -- "
        "see classify_outcome. Phase 8.3: formalizes a distinction phase3_action_page/"
        "eval_weapon_effects's acceptable_outcome prose already made, into a structured, "
        "checkable field (docs/decisions/0014's 'Open questions').",
    )
    notes: str = Field(default="", description="Free-text reasoning behind the labels above.")
    annotation_provenance: AnnotationProvenance | None = Field(
        default=None,
        description="Structured provenance for how ground truth was established, when known -- "
        "see the AnnotationProvenance docstring above. Never set by any code path; only by a "
        "human editing configs/phase3_3_eval_dataset.yaml.",
    )
    annotation_version: int = Field(
        default=1,
        ge=1,
        description="Bumped whenever this sample's ground-truth fields (animation_possible, "
        "expected_target_category, expected_motion_category, expected_region_note, "
        "acceptable_outcome, regression_reference, ground_truth_uncertain, "
        "honest_failure_acceptable, annotation_provenance) change intentionally -- an explicit, "
        "auditable signal that this "
        "specific annotation was reviewed and revised, on top of (not instead of) git's own "
        "commit history. Never bumped by any code path in this project; only by a human editing "
        "configs/phase3_3_eval_dataset.yaml.",
    )
    golden_categories: list[GoldenCategory] = Field(
        default_factory=list,
        description="Phase 8: which of the golden E2E dataset's required coverage categories "
        "(GOLDEN_DATASET_CATEGORIES) this sample demonstrates, based on real evidence already "
        "documented for it (see docs/phase7-results.md section 6.2, docs/decisions/0011, ADR "
        "0010's Phase 5 audit). Descriptive coverage metadata, like diversity_tag -- NOT a "
        "ground-truth claim, so changing it does not require bumping annotation_version (see "
        "that field's own docstring for the exact list of fields that do).",
    )
    scene_complexity_tags: list[SceneComplexityTag] = Field(
        default_factory=list,
        description="Phase 9 (Real-World Evaluation Dataset): which SCENE_COMPLEXITY_TAGS "
        "this sample's real artwork visibly exhibits, assigned by direct visual inspection of "
        "the actual page -- e.g. multiple_panels, crowded_scene, complex_background. "
        "Descriptive dataset-composition metadata, like diversity_tag/golden_categories -- NOT "
        "a ground-truth claim about animation_possible, so changing it does not require "
        "bumping annotation_version.",
    )
    potential_motion_tags: list[PotentialMotionTag] = Field(
        default_factory=list,
        description="Phase 9: which POTENTIAL_MOTION_TAGS this sample's artwork visibly "
        "suggests (drawn motion lines, wind/implied-force cues, a raised weapon, an impact "
        "effect...), by direct visual inspection -- describes what a human sees as plausible, "
        "not a claim that the pipeline will find/select it. Same non-ground-truth status as "
        "scene_complexity_tags.",
    )
    geometric_difficulty_tags: list[GeometricDifficultyTag] = Field(
        default_factory=list,
        description="Phase 9: which GEOMETRIC_DIFFICULTY_TAGS this sample's most plausible "
        "animation target(s) exhibit -- e.g. near_boundary, overlapping_objects, "
        "thin_structure. Assigned from visual inspection of the specific candidate region(s), "
        "not the whole page in the abstract. Same non-ground-truth status as "
        "scene_complexity_tags.",
    )
    motion_type_tags: list[MotionTypeTag] = Field(
        default_factory=list,
        description="Phase 9: which MOTION_TYPE_TAGS the page's drawn motion cues most "
        "plausibly call for (translation/rotation/scale/deformation/mixed_motion), by visual "
        "inspection -- a prediction about what the pipeline *should* attempt, not a report of "
        "what it actually did (that comes from PageRunOutcome). Same non-ground-truth status "
        "as scene_complexity_tags.",
    )
    expected_difficulty: DifficultyLevel | None = Field(
        default=None,
        description="Phase 9: a qualitative, honestly-subjective difficulty estimate "
        "('easy'/'medium'/'hard') for this sample, based on how many of the tags above stack "
        "up and how open-ended the target selection is -- `None` when no considered estimate "
        "was actually made, never guessed to fill the field.",
    )


def load_eval_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvalSample]:
    """Load and validate the evaluation dataset manifest. Raises `pydantic.ValidationError` on

    a malformed entry -- never silently drops or invents a sample. Also raises `ValueError` if
    two samples declare the same `image_path` (Pre-Phase-3.4 "verified action" integration): the
    same image must never carry two conflicting ground-truth identities, and this is a cheap,
    filesystem-independent check (it does not require the images themselves to exist on disk,
    unlike a byte-content check would) that still catches the literal, structural form of that
    mistake -- a copy-pasted manifest entry pointing at an already-used path.
    """
    data = yaml.safe_load(path.read_text()) or {}
    samples = [EvalSample.model_validate(entry) for entry in data.get("samples", [])]

    seen_paths: dict[str, str] = {}
    for sample in samples:
        if sample.image_path in seen_paths:
            raise ValueError(
                f"duplicate image_path {sample.image_path!r} declared under both "
                f"sample_id={seen_paths[sample.image_path]!r} and "
                f"sample_id={sample.sample_id!r} -- the same image must not carry two "
                "conflicting ground-truth identities"
            )
        seen_paths[sample.image_path] = sample.sample_id

    return samples


def load_combined_eval_dataset(paths: list[Path] | None = None) -> list[EvalSample]:
    """Phase 9: load and concatenate multiple dataset manifests -- by default the Phase 8

    golden regression set (`DEFAULT_DATASET_PATH`) plus the Real-World Evaluation Dataset
    (`REALWORLD_DATASET_PATH`) -- into one list, with the same duplicate-`image_path` integrity
    check as `load_eval_dataset` extended ACROSS files (each file's own internal duplicate check
    already runs via the per-file `load_eval_dataset` call; this additionally catches the same
    image accidentally declared in two different manifest files, which no single
    `load_eval_dataset` call can see on its own). Never silently drops or merges a collision.
    """
    if paths is None:
        paths = [DEFAULT_DATASET_PATH, REALWORLD_DATASET_PATH]
    combined: list[EvalSample] = []
    seen_paths: dict[str, str] = {}
    for path in paths:
        for sample in load_eval_dataset(path):
            if sample.image_path in seen_paths:
                raise ValueError(
                    f"duplicate image_path {sample.image_path!r} declared under both "
                    f"sample_id={seen_paths[sample.image_path]!r} and "
                    f"sample_id={sample.sample_id!r} across the combined manifest set "
                    f"{[str(p) for p in paths]!r} -- the same image must not carry two "
                    "conflicting ground-truth identities"
                )
            seen_paths[sample.image_path] = sample.sample_id
            combined.append(sample)
    return combined


def golden_category_coverage(samples: list[EvalSample]) -> dict[GoldenCategory, list[str]]:
    """Phase 8: which `sample_id`s cover each of `GOLDEN_DATASET_CATEGORIES` -- an empty list

    for a category is a real, honest coverage gap, not an error (see
    `uncovered_golden_categories` and configs/phase3_3_eval_dataset.yaml's header note on the
    two categories this dataset genuinely does not cover yet: `partially_occluded_object` and
    `scale_or_deformation`).
    """
    coverage: dict[GoldenCategory, list[str]] = {c: [] for c in GOLDEN_DATASET_CATEGORIES}
    for sample in samples:
        for category in sample.golden_categories:
            coverage[category].append(sample.sample_id)
    return coverage


def uncovered_golden_categories(samples: list[EvalSample]) -> list[GoldenCategory]:
    """Every `GoldenCategory` with zero samples covering it, in `GOLDEN_DATASET_CATEGORIES`

    order -- the Phase 8 brief's own instruction not to hide gaps, made directly checkable
    instead of only prose in a doc.
    """
    coverage = golden_category_coverage(samples)
    return [c for c in GOLDEN_DATASET_CATEGORIES if not coverage[c]]


def dataset_composition(samples: list[EvalSample]) -> dict[str, dict[str, int]]:
    """Phase 9: per-taxonomy-dimension sample counts for the Real-World Evaluation Dataset's

    required "dataset size, composition, categories" report field (brief sections 4/19/23) --
    computed once from the real tag fields instead of hand-tallied prose. Every dimension's
    inner dict is keyed by every value the closed `Literal` allows (0 included, same "a gap is
    real and visible, not silently absent" convention as `golden_category_coverage`), plus a
    `sample_count` top-level entry so a reader never has to cross-reference `len(samples)`
    separately.
    """
    return {
        "sample_count": {"total": len(samples)},
        "scene_complexity_tags": _tag_counts(
            samples, SCENE_COMPLEXITY_TAGS, "scene_complexity_tags"
        ),
        "potential_motion_tags": _tag_counts(
            samples, POTENTIAL_MOTION_TAGS, "potential_motion_tags"
        ),
        "geometric_difficulty_tags": _tag_counts(
            samples, GEOMETRIC_DIFFICULTY_TAGS, "geometric_difficulty_tags"
        ),
        "motion_type_tags": _tag_counts(samples, MOTION_TYPE_TAGS, "motion_type_tags"),
        "expected_difficulty": _scalar_counts(
            samples, ("easy", "medium", "hard"), "expected_difficulty"
        ),
        "animation_possible": _scalar_counts(
            samples, ("yes", "no", "uncertain"), "animation_possible"
        ),
    }


def _tag_counts(
    samples: list[EvalSample], all_values: tuple[str, ...], attr: str
) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(all_values, 0)
    for sample in samples:
        for value in getattr(sample, attr):
            counts[value] += 1
    return counts


def _scalar_counts(
    samples: list[EvalSample], all_values: tuple[str, ...], attr: str
) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(all_values, 0)
    for sample in samples:
        value = getattr(sample, attr)
        if value is not None:
            counts[value] += 1
    return counts
