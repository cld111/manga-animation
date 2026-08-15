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

from dataclasses import dataclass
from typing import Any

from PIL import Image

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
