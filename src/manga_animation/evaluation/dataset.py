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
        "annotation_provenance) change intentionally -- an explicit, auditable signal that this "
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
