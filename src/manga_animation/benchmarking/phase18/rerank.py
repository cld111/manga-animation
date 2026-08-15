"""Phase 18.2: VLM-guided candidate reranking -- the ranking logic and the VLM scoring.

Reuses the EXISTING production-compatible VLM verification mechanism end to end (the brief:
"использовать существующий production-compatible механизм настолько, насколько это возможно"):
the exact same prompt builder, crop, and response parser that `validation.validate_target` uses
(`_build_verification_prompt`, `_crop_with_margin`, `_parse_verification`), plus the exact same
deterministic bbox-plausibility pre-filter (`_bbox_plausibility`). Nothing here modifies
production; it applies the production mechanism to MANY candidates instead of one, which is the
benchmark-only extension.

The VLM semantic score per candidate is `(matches: bool, confidence: float)` (unparseable ->
fail-closed None). Ranking strategies:
  A -- semantic only:  (matches desc, confidence desc)
  B -- semantic + DINO: (matches desc, (confidence or 0) + dino_score desc) -- DINO is NOT the
      main signal (fixed 1:1 blend, deterministic, no ML)
  C -- semantic + geometry gate: drop candidates failing the production `_bbox_plausibility`
      check first, then rank as A.

GT is never an input here: the selector sees only DINO candidates, page image, and the
production prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.benchmarking.phase17.metrics import bbox_iou
from manga_animation.pipeline.types import BBoxPx, ImageArray
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
    Vector2,
)
from manga_animation.validation.validate import (
    _bbox_plausibility,
    _build_verification_prompt,
    _crop_with_margin,
    _parse_verification,
)

BBox = tuple[int, int, int, int]


def production_object_plan(semantic_label: str = "character_body") -> ObjectPlan:
    """The `ObjectPlan` the production verification prompt is built from (TRANSLATE motion, as
    a `body` target would carry). `validate_target` never reads motion geometry here -- only
    the prompt text uses it as motion context."""
    return ObjectPlan(
        object_id="p18.2",
        panel_id="p18.2",
        semantic_label=semantic_label,
        confidence=0.9,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.02,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
        ),
    )


@dataclass(frozen=True, slots=True)
class VlmCandidateScore:
    """The VLM's semantic read on one DINO candidate, via the production verification prompt."""

    box: BBox
    dino_score: float
    matches: bool | None  # None -> VLM response unparseable (fail closed)
    confidence: float | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "dino_score": self.dino_score,
            "matches": self.matches,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def vlm_score_candidate(
    vlm_client: VLMClient,
    image: ImageArray,
    object_plan: ObjectPlan,
    box: BBox,
) -> VlmCandidateScore:
    """One production-compatible VLM semantic check on one candidate's margin crop.

    Reuses `validate_target`'s exact prompt (`_build_verification_prompt`), crop
    (`_crop_with_margin`, 15% margin) and parser (`_parse_verification`); unparseable responses
    are fail-closed (`matches=None`), exactly as production treats them.
    """
    bbox = BBoxPx(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    crop = _crop_with_margin(image, bbox)
    prompt = _build_verification_prompt(object_plan)
    raw_text = vlm_client.generate(Image.fromarray(crop), prompt)
    parsed = _parse_verification(raw_text)
    if parsed is None:
        return VlmCandidateScore(
            box=box, dino_score=0.0, matches=None, confidence=None, reason=None
        )
    return VlmCandidateScore(
        box=box,
        dino_score=0.0,  # set by the caller from the DINO detection
        matches=parsed.matches,
        confidence=parsed.confidence,
        reason=parsed.reason,
    )


def _attach_dino_score(score: VlmCandidateScore, dino_score: float) -> VlmCandidateScore:
    return VlmCandidateScore(
        box=score.box,
        dino_score=dino_score,
        matches=score.matches,
        confidence=score.confidence,
        reason=score.reason,
    )


# --- Instance-specific contrastive prompt (benchmark-only, Phase 18.2.1) --------------------
#
# The production verification prompt is a PRESENCE check ("does the crop show character
# body?") and is satisfied by every character on a page -- the measured reason the VLM
# reranker falls into semantic confusion. This benchmark-only prompt instead forces the VLM to
# discriminate ONE specific, cleanly-isolated instance: it must reject crops that contain
# multiple characters, partial figures cut by the crop edge, or content dominated by text /
# bubbles / panel borders. It deliberately does NOT see the GT box and does NOT use DINO score.
# This is a research variant, NOT the production prompt; nothing here changes production.

SPECIFIC_INSTANCE_PROMPT_TEMPLATE = """You are selecting the ONE candidate region that \
isolates a single, complete, distinct character body on a manga page, so that region can be \
treated as one specific character.

Target object: "character body"

Look at the image above. Does it show EXACTLY ONE character body as its clear, prominent \
subject: a single complete figure (head, torso, clothing) that is NOT cut off by the crop \
edge, does NOT contain a second character or a stray limb of another character, and is NOT \
dominated by a speech bubble, dialogue text, or panel border?

Only answer true when the crop isolates exactly one such specific character and nothing else \
claims the frame. Otherwise answer false -- a crop with several characters, a partial figure, \
or a dominating bubble/text/frame is NOT a specific single-instance crop.

Answer with ONLY one JSON object, no prose: {"is_specific": true or false, "confidence": a \
float 0-1, "reason": "one short sentence"}"""


@dataclass(frozen=True, slots=True)
class SpecificCandidateScore:
    """The VLM's instance-specific read on one candidate crop (benchmark-only prompt)."""

    box: BBox
    is_specific: bool | None  # None -> unparseable response (fail closed)
    confidence: float | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": list(self.box),
            "is_specific": self.is_specific,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class _SpecificResponse(BaseModel):
    is_specific: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def _parse_specific_response(raw_text: str) -> _SpecificResponse | None:
    """Parse the specific-prompt JSON; None (fail closed) on anything unparseable. Defensively
    tolerates the VLM echoing the prompt's JSON placeholder as doubled braces (`{{...}}`)."""
    from manga_animation.validation.validate import _extract_json_object

    candidates = [raw_text]
    if "{{" in raw_text and "}}" in raw_text:
        candidates.append(raw_text.replace("{{", "{").replace("}}", "}"))
    for text in candidates:
        try:
            object_text = _extract_json_object(text)
            data = json.loads(object_text)
            return _SpecificResponse.model_validate(data)
        except (json.JSONDecodeError, ValueError, ValidationError):
            continue
    return None


def specific_score_candidate(
    vlm_client: VLMClient, image: ImageArray, box: BBox
) -> SpecificCandidateScore:
    """One instance-specific contrastive VLM read on one candidate's margin crop (the same
    production crop, a benchmark-only prompt -- see SPECIFIC_INSTANCE_PROMPT_TEMPLATE)."""
    bbox = BBoxPx(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    crop = _crop_with_margin(image, bbox)
    raw_text = vlm_client.generate(Image.fromarray(crop), SPECIFIC_INSTANCE_PROMPT_TEMPLATE)
    parsed = _parse_specific_response(raw_text)
    if parsed is None:
        return SpecificCandidateScore(box=box, is_specific=None, confidence=None, reason=None)
    return SpecificCandidateScore(
        box=box, is_specific=parsed.is_specific, confidence=parsed.confidence, reason=parsed.reason
    )


def rank_specific(scores: list[SpecificCandidateScore]) -> list[SpecificCandidateScore]:
    """Strategy S ranking: instance-specific signal only (no DINO score): is_specific=True
    first (confidence desc), then False, then unparseable (fail closed) last."""
    def key(s: SpecificCandidateScore) -> tuple[int, float]:
        if s.is_specific is None:
            return (0, -1.0)
        if s.is_specific:
            return (2, s.confidence if s.confidence is not None else -1.0)
        return (1, s.confidence if s.confidence is not None else -1.0)

    return sorted(scores, key=key, reverse=True)


def specific_is_correct(
    gt_bbox: BBox, ranked: list[SpecificCandidateScore], threshold: float = 0.5
) -> bool:
    """Whether the top-ranked instance-specific candidate matches the GT (evaluation only)."""
    return bool(ranked) and bbox_iou(gt_bbox, ranked[0].box) >= threshold


def rank_of_best_specific(
    gt_bbox: BBox, ranked: list[SpecificCandidateScore], threshold: float = 0.5
) -> int | None:
    """1-based rank of the highest-positioned specific candidate with IoU >= threshold."""
    for i, cand in enumerate(ranked):
        if bbox_iou(gt_bbox, cand.box) >= threshold:
            return i + 1
    return None


def _semantic_rank_key(score: VlmCandidateScore) -> tuple[int, float]:
    # Fail-closed candidates (matches=None) rank last; matches=True before matches=False;
    # higher confidence first.
    if score.matches is None:
        return (0, -1.0)
    if score.matches:
        return (2, score.confidence if score.confidence is not None else -1.0)
    return (1, score.confidence if score.confidence is not None else -1.0)


def _blend_rank_key(score: VlmCandidateScore) -> tuple[int, float]:
    # matches dominates; within a matches group, DINO score is a fixed 1:1 additive secondary
    # signal (DINO is deliberately NOT the primary signal).
    if score.matches is None:
        return (0, -1.0)
    confidence = score.confidence if score.confidence is not None else 0.0
    blended = confidence + score.dino_score
    if score.matches:
        return (2, blended)
    return (1, blended)


def rank_scores(
    scores: list[VlmCandidateScore],
    strategy: str,
    *,
    image_shape: tuple[int, int] | None = None,
) -> list[VlmCandidateScore]:
    """Rank candidates by the given strategy (A/B/C); returns best-first. Unparseable VLM
    responses always sort last. `image_shape` is required for strategy C (geometry gate)."""
    if strategy not in ("A", "B", "C"):
        raise ValueError(f"unknown ranking strategy {strategy!r}")
    if strategy == "C":
        if image_shape is None:
            # No page geometry -> the geometry gate cannot run -> no candidate survives it.
            return []
        filtered = [
            s
            for s in scores
            if _bbox_plausibility(
                BBoxPx(x0=s.box[0], y0=s.box[1], x1=s.box[2], y1=s.box[3]), image_shape
            )[0]
        ]
    else:
        filtered = list(scores)
    key = _semantic_rank_key if strategy in ("A", "C") else _blend_rank_key
    return sorted(filtered, key=key, reverse=True)


def selected_is_correct(
    gt_bbox: BBox, ranked: list[VlmCandidateScore], threshold: float = 0.5
) -> bool:
    """Whether the top-ranked candidate after reranking matches the GT (evaluation only)."""
    return bool(ranked) and bbox_iou(gt_bbox, ranked[0].box) >= threshold


def rank_of_best_correct(
    gt_bbox: BBox, ranked: list[VlmCandidateScore], threshold: float = 0.5
) -> int | None:
    """1-based rank of the highest-positioned candidate with IoU >= threshold in the RERANKED
    order; None if no candidate matches. The R@K comparison metric for phase 18.2."""
    for i, cand in enumerate(ranked):
        if bbox_iou(gt_bbox, cand.box) >= threshold:
            return i + 1
    return None
