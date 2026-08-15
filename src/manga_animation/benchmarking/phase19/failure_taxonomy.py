"""Phase 19 failure taxonomy: every per-target outcome gets exactly one primary category.

The brief's categories (section 11):

    A  correct target + good mask          B  correct target + poor mask
    C  wrong instance                      D  multiple instances
    E  target not identified               F  no mask
    G  text/speech bubble contamination    H  panel-border contamination
    I  unrelated-object contamination      J  coordinate/preprocessing failure
    K  inference/runtime failure

Classification is deterministic from measured signals so the same record always maps to the
same category and the report's counts are reproducible. The signals come from the adapter
(status / masks) and from the metrics + phase-17 GT safety masks (forbidden overlap).

`I` (unrelated-object contamination) has no automatic GT signal in the phase-17 dataset, so the
automatic classifier never emits it; it is reserved for a manual review pass (the autonomous
gallery), and the report documents that explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# The one forbidden-overlap fraction that counts as "contaminated". Phase 17 measured
# text/balloon/onomatopoeia absorption at <=1-3% for healthy masks, so 0.15 is a generous
# safety bar -- a mask crossing that has absorbed a substantial forbidden region.
CONTAMINATION_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class SampleSignal:
    """All measured signals the classifier needs for one target."""

    status: Literal["ok", "inference_error"]
    n_masks: int
    coord_ok: bool  # masks map to the page geometry (no coordinate/preprocessing failure)
    instance_correct: bool
    iou: float
    no_target_text: bool  # the response text indicates the target was not found/refused
    forbidden: dict[str, float] = field(default_factory=dict)  # text/balloon/frame/onomatopoeia
    multi_instance: bool = False  # >1 distinct mask emitted (mutual overlap < 0.5)
    manual_category: str | None = None  # optional human override (autonomous review), e.g. "I"


def forbidden_total(forbidden: dict[str, float]) -> float:
    return sum(v for v in forbidden.values())


def classify(signal: SampleSignal) -> str:
    """Assign one primary failure-taxonomy category to a target's measured outcome.

    Priority order (documented so counts are interpretable):
      1. inference/runtime failure        -> K
      2. coordinate/preprocessing failure -> J
      3. no mask emitted                  -> E (target not identified) / F (no mask)
      4. forbidden contamination          -> G (text/balloon/onomatopoeia) / H (frame)
      5. multiple instances               -> D
      6. instance-correct -> A / B, wrong instance -> C
    A manual override (autonomous review) takes precedence over every automatic rule.
    """
    if signal.manual_category is not None:
        return signal.manual_category
    if signal.status == "inference_error":
        return "K"
    if not signal.coord_ok:
        return "J"
    if signal.n_masks == 0:
        return "E" if signal.no_target_text else "F"
    total = forbidden_total(signal.forbidden)
    if total >= CONTAMINATION_FRACTION:
        if signal.forbidden.get("frame", 0.0) >= CONTAMINATION_FRACTION:
            return "H"
        return "G"
    if signal.multi_instance:
        return "D"
    if signal.instance_correct:
        return "A" if signal.iou >= 0.50 else "B"
    return "C"


CATEGORY_LABELS: dict[str, str] = {
    "A": "correct target + good mask",
    "B": "correct target + poor mask",
    "C": "wrong instance",
    "D": "multiple instances",
    "E": "target not identified",
    "F": "no mask",
    "G": "text/speech-bubble contamination",
    "H": "panel-border contamination",
    "I": "unrelated-object contamination",
    "J": "coordinate/preprocessing failure",
    "K": "inference/runtime failure",
}


def describe(category: str) -> str:
    return CATEGORY_LABELS.get(category, f"unknown category {category!r}")
