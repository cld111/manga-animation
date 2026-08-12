"""The Phase 3.3 evaluation dataset: real manga pages with honestly-establishable ground truth.

See `docs/phase3.3-results.md` for the actual dataset write-up and
`configs/phase3_3_eval_dataset.yaml` for the data itself -- this module only defines the
schema and the loader. Per the Phase 3.3 brief: "Do NOT fabricate ground truth." Every field
below is optional (`None`) except `sample_id`/`image_path`/`diversity_tag`/`source_citation` --
a sample with genuinely uncertain ground truth leaves the uncertain fields `None` rather than
guessing, and `ground_truth_uncertain=True` makes that explicit instead of silently omitting it.

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
from typing import Literal

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
    fetch_script: str = Field(
        min_length=1, description="The scripts/fetch_*.py that reproduces image_path on demand."
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
    annotation_version: int = Field(
        default=1,
        ge=1,
        description="Bumped whenever this sample's ground-truth fields (animation_possible, "
        "expected_target_category, expected_motion_category, expected_region_note, "
        "acceptable_outcome, regression_reference, ground_truth_uncertain) change intentionally "
        "-- an explicit, auditable signal that this specific annotation was reviewed and "
        "revised, on top of (not instead of) git's own commit history. Never bumped by any "
        "code path in this project; only by a human editing configs/phase3_3_eval_dataset.yaml.",
    )


def load_eval_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[EvalSample]:
    """Load and validate the evaluation dataset manifest. Raises `pydantic.ValidationError` on

    a malformed entry -- never silently drops or invents a sample.
    """
    data = yaml.safe_load(path.read_text()) or {}
    samples = data.get("samples", [])
    return [EvalSample.model_validate(entry) for entry in samples]
