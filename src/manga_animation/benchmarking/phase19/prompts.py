"""Phase 19 prompt generation for OMG-LLaVA, adapted to the official chat interface.

OMG-LLaVA is a natural-language referring-segmentation model: you describe the target in a
referring expression and (in grounded-caption mode) it emits `[SEG]` tokens where a pixel mask
is requested. Per the official README/app:

- referring form:   "Can you please segment <target> in the given image"
- autonomous form:  grounded caption asking for interleaved segmentation masks, e.g.
  "Please output with interleaved segmentation masks for the corresponding parts of the answer."

This module builds the exact prompt text for both the autonomous experiment (section 4/18) and
the four controlled conditions A-D (section 8), and is deliberately pure (no model imports).
"""

from __future__ import annotations

from dataclasses import dataclass

from manga_animation.benchmarking.phase17.dataset import (
    FORBIDDEN_CATEGORIES,
    MAIN_OBJECT_CATEGORY,
)

# Official referring-segmentation phrasing (verified from the OMG-LLaVA gradio app).
REFERRING_TEMPLATE = "Can you please segment {expression} in the given image"

CONTROLLED_CONDITIONS = ("A", "B", "C", "D")


@dataclass(frozen=True, slots=True)
class ControlledPrompt:
    """One controlled-condition prompt for a single benchmark target.

    `provenance` records where the description came from. Only `PRODUCTION_AVAILABLE`
    descriptions may feed the primary controlled result (phase brief section 7); the
    GT-derived conditions are diagnostics.
    """

    condition: str  # "A" | "B" | "C" | "D"
    expression: str
    prompt: str  # the exact text sent after the image token
    provenance: str  # "PRODUCTION_AVAILABLE" | "GT_DERIVED"

    def as_dict(self) -> dict[str, str]:
        return {
            "condition": self.condition,
            "expression": self.expression,
            "prompt": self.prompt,
            "provenance": self.provenance,
        }


PRODUCTION_AVAILABLE = "PRODUCTION_AVAILABLE"
GT_DERIVED = "GT_DERIVED"


def referring_prompt(expression: str) -> str:
    """The official OMG-LLaVA referring-segmentation prompt for a target expression."""
    return REFERRING_TEMPLATE.format(expression=expression)


def production_expression(semantic_label: str) -> str:
    """The production-available description for a target: the exact phrase the production
    grounding prompt uses (`_prompt_from_label` -> "character body."), minus the trailing
    period so it reads as a referring expression. This is what the existing pipeline can
    actually produce today -- no GT-derived information is added."""
    phrase = semantic_label.replace("_", " ").strip()
    return phrase.rstrip(".")


# Human-readable word for each dataset category when it is used as a bare category prompt
# (the phase brief's condition-A example is "character"; the MS92 category name is "body").
_CATEGORY_WORDS = {"body": "character"}


def category_expression(category: str = MAIN_OBJECT_CATEGORY) -> str:
    """Condition A: category-only expression (diagnostic)."""
    return _CATEGORY_WORDS.get(category, category)


def spatial_side(gt_bbox: tuple[int, int, int, int], page_size: tuple[int, int]) -> str:
    """A coarse spatial qualifier for the GT instance relative to its page, derived purely from
    the GT bbox center vs the page center. GT-DERIVED -- diagnostic only (condition B/C)."""
    x0, y0, x1, y1 = gt_bbox
    h, w = page_size
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    if abs(cx - w / 2) > abs(cy - h / 2):
        return "on the left" if cx < w / 2 else "on the right"
    return "in the upper part" if cy < h / 2 else "in the lower part"


def spatial_expression(
    category: str, gt_bbox: tuple[int, int, int, int], page_size: tuple[int, int]
) -> str:
    """Condition B: spatial-only referring expression (GT-derived, diagnostic)."""
    return f"the {category_expression(category)} {spatial_side(gt_bbox, page_size)}"


def semantic_spatial_expression(
    category: str, gt_bbox: tuple[int, int, int, int], page_size: tuple[int, int]
) -> str:
    """Condition C: semantic + spatial referring expression. The only semantic available for a
    phase-17 target is its category (all targets are `character_body`); combining it with the
    GT-derived spatial qualifier is a diagnostic, explicitly GT-derived."""
    return f"the {category_expression(category)} {spatial_side(gt_bbox, page_size)}"


def controlled_prompt(
    condition: str,
    *,
    semantic_label: str,
    category: str = MAIN_OBJECT_CATEGORY,
    gt_bbox: tuple[int, int, int, int] | None = None,
    page_size: tuple[int, int] | None = None,
) -> ControlledPrompt:
    """Build the prompt for one controlled condition of one target.

    - A (diagnostic): category only, e.g. "character".
    - B (diagnostic, GT-derived): spatial, e.g. "the character on the right".
    - C (diagnostic, GT-derived): semantic + spatial.
    - D (primary): the production-available description, i.e. the exact phrase production
      grounding would build for this target.
    """
    if condition == "A":
        expression = category_expression(category)
        return ControlledPrompt(condition, expression, referring_prompt(expression), GT_DERIVED)
    if condition == "B":
        if gt_bbox is None or page_size is None:
            raise ValueError("condition B requires gt_bbox and page_size")
        expression = spatial_expression(category, gt_bbox, page_size)
        return ControlledPrompt(condition, expression, referring_prompt(expression), GT_DERIVED)
    if condition == "C":
        if gt_bbox is None or page_size is None:
            raise ValueError("condition C requires gt_bbox and page_size")
        expression = semantic_spatial_expression(category, gt_bbox, page_size)
        return ControlledPrompt(condition, expression, referring_prompt(expression), GT_DERIVED)
    if condition == "D":
        expression = production_expression(semantic_label)
        return ControlledPrompt(
            condition, expression, referring_prompt(expression), PRODUCTION_AVAILABLE
        )
    raise ValueError(
        f"unknown condition {condition!r}; expected one of {CONTROLLED_CONDITIONS}"
    )


# --- Autonomous mode ----------------------------------------------------------------------
# Phrased in OMG-LLaVA's grounded-caption style (asks for the segmentation mask to be
# output), so the model can emit a [SEG] token for the chosen element. Deliberately generic:
# no GT box, no GT mask, no target crop, no GT-derived description (phase brief section 4).
AUTONOMOUS_INSTRUCTION = (
    "Analyze this manga page and identify one visual element whose depicted motion could be "
    "meaningfully animated while preserving the original artwork: for example a character in "
    "action, flowing hair, moving clothing, a waving flag, flowing water, smoke, dust, "
    "particles, or an object involved in an action. Do not select text, speech bubbles, "
    "captions, panel borders, logos, or unrelated static background elements. Briefly state "
    "which element you chose and why it depicts motion, and output the segmentation mask "
    "for it."
)

# NOTE: the instruction must NOT contain the literal "[SEG]" token. "[SEG]" is a special token
# in OMG-LLaVA's vocabulary; putting it in the prompt would place it in the prompt's token ids
# and corrupt the official mask-extraction alignment (`get_seg_hidden_states` indexes the
# generated hidden states against the full output ids). The model emits [SEG] by itself in
# grounded-caption mode when asked to output a segmentation mask.


def autonomous_prompt() -> str:
    """The full-page autonomous instruction (no oracle information)."""
    return AUTONOMOUS_INSTRUCTION


# --- Annotation helpers --------------------------------------------------------------------
def condition_provenance(condition: str) -> str:
    """The provenance label for a condition, for the report's oracle-classification table."""
    if condition == "D":
        return PRODUCTION_AVAILABLE
    return GT_DERIVED


def forbidden_terms_doc() -> str:
    """Human-readable list of what the autonomous prompt forbids (for the report)."""
    return ", ".join(FORBIDDEN_CATEGORIES)
