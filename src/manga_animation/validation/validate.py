"""Explicit target validation: does a grounded region actually depict what it claims to?

New Phase 3.2 stage, sitting between grounding and segmentation (see docs/pipeline.md and
`docs/decisions/0006-grounding-target-validation.md`). Grounding DINO's own detection score is
a "does this box respond to this text prompt" signal, not a "is this the right object" one — a
technically valid, in-bounds, above-threshold detection can still be semantically wrong (the
real Phase 3.1 finding this stage exists to catch: a `flag_banner` prompt scored 0.269 and
landed on a face/speech-bubble region with no banner anywhere near it — SAM 2.1 then produced a
perfectly clean mask for entirely the wrong object). Only `validate_target` is the public entry
point; everything else here is an implementation detail of that decision.

Nothing here imports `torch`/`transformers` directly — the (potentially model-backed) VLM call
happens entirely behind the `VLMClient` protocol already defined in `analysis/client.py`, same
seam the analysis stage uses (see that module's docstring for why).
"""

from __future__ import annotations

import json
import re

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import (
    MAX_OBJECT_COVERAGE_FRACTION,
    MIN_OBJECT_COVERAGE_FRACTION,
    BBoxPx,
    GroundingResult,
    ImageArray,
    ValidationResult,
)
from manga_animation.schemas.animation_plan import ObjectPlan
from manga_animation.validation.transform_geometry import check_transform_geometry

logger = get_logger(__name__)

_MARGIN_FRACTION = 0.15
"""Padding added around a grounding bbox before cropping for the VLM check, as a fraction of
the box's own width/height. Not a pass/fail threshold (nothing is accepted or rejected based
on this number) — purely an image-prep choice so the VLM sees a little surrounding context
instead of an overly tight crop, a standard practice for crop-based verification, not a
calibrated value."""

_VERIFICATION_PROMPT_TEMPLATE = """You are checking whether a cropped region of a manga/comic \
page actually shows a specific object, before that region is animated. Be strict: manga art is \
dense and a crop can easily contain the wrong nearby element (a face, dialogue text, another \
character's clothing) instead of the intended object.

Target object: "{semantic_label}"
{motion_context}

Does the image above show "{semantic_label}"? Answer with ONLY one JSON object, no prose, no \
markdown fences, in exactly this shape:
{{"matches": true or false, "confidence": a float 0-1, "reason": "one short sentence"}}

If the crop shows something else entirely rather than the target object, answer false. Only \
answer true if the crop is a plausible, identifiable instance of the target object itself."""


class _VerificationResponse(BaseModel):
    """The VLM's structured read on one cropped candidate region."""

    matches: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def _build_verification_prompt(object_plan: ObjectPlan) -> str:
    label = object_plan.semantic_label.replace("_", " ")
    motion_context = ""
    if object_plan.motion is not None:
        motion_context = (
            f'The animation plan intends to apply "{object_plan.motion.transform_kind.value}" '
            f'motion to it. A genuine "{label}" should be physically plausible to move that '
            "way -- if the crop instead shows something rigid or unrelated (a face, dialogue "
            "text, a flat background) that would not move like that, answer false even if it "
            "loosely resembles the target."
        )
    return _VERIFICATION_PROMPT_TEMPLATE.format(semantic_label=label, motion_context=motion_context)


def _extract_json_object(text: str) -> str:
    """Best-effort recovery of a JSON object from a VLM's raw text response.

    Mirrors `analysis.plan_builder._extract_json_array`'s fence-stripping/balanced-scan
    approach, applied to a single `{...}` object instead of a `[...]` array.
    """
    stripped = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found in VLM verification output")

    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                json.loads(candidate)  # raises json.JSONDecodeError if still malformed
                return candidate

    raise ValueError("no balanced JSON object found in VLM verification output")


def _parse_verification(raw_text: str) -> _VerificationResponse | None:
    """`None` (never an exception) on failure -- an unparseable verification response fails

    closed (rejected), it never blocks the whole pipeline run the way an unparseable analysis
    response does. Unlike `plan_builder`, there is deliberately no recovery re-prompt here:
    validation may run several times per object (once per grounding candidate), and "the model
    couldn't answer a simple yes/no about this specific crop" is itself informative enough to
    reject on, without doubling the call count for every candidate.
    """
    try:
        object_text = _extract_json_object(raw_text)
        data = json.loads(object_text)
        return _VerificationResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logger.info(
            "validation stage: VLM verification response could not be parsed (%s); "
            "rejecting this candidate",
            exc,
        )
        return None


def _crop_with_margin(
    image: ImageArray, bbox: BBoxPx, margin_frac: float = _MARGIN_FRACTION
) -> ImageArray:
    h, w = image.shape[0], image.shape[1]
    mx = max(1, round(bbox.width * margin_frac))
    my = max(1, round(bbox.height * margin_frac))
    x0 = max(0, bbox.x0 - mx)
    y0 = max(0, bbox.y0 - my)
    x1 = min(w, bbox.x1 + mx)
    y1 = min(h, bbox.y1 + my)
    return image[y0:y1, x0:x1]


def _bbox_plausibility(bbox: BBoxPx, image_shape: tuple[int, int]) -> tuple[bool, float]:
    """Cheap, deterministic pre-filter, no model call. Reuses the same coverage-fraction

    bounds `segmentation/segment.py` already applies to its tight mask (see
    `pipeline/types.py`'s `MIN_OBJECT_COVERAGE_FRACTION`/`MAX_OBJECT_COVERAGE_FRACTION`) —
    applied here to the looser grounding bbox instead of a tight mask, so it is, if anything,
    more permissive than the segmentation check that already uses the same numbers. This is
    reuse of an existing, already-documented calibration rationale, not a new invented number.
    """
    h, w = image_shape
    area_fraction = (bbox.width * bbox.height) / (h * w)
    plausible = MIN_OBJECT_COVERAGE_FRACTION <= area_fraction <= MAX_OBJECT_COVERAGE_FRACTION
    return plausible, area_fraction


def _client_model_id(vlm_client: VLMClient) -> str:
    return str(getattr(vlm_client, "source", type(vlm_client).__name__))


def validate_target(
    image: ImageArray,
    object_plan: ObjectPlan,
    grounding: GroundingResult,
    vlm_client: VLMClient,
    *,
    candidate_rank: int = 0,
    panel_bbox_px: BBoxPx | None = None,
) -> ValidationResult:
    """ACCEPT/REJECT one grounding candidate for `object_plan`, with structured diagnostics.

    Three independent checks, cheapest first:

    1. **Bbox plausibility** (deterministic, no model call) — see `_bbox_plausibility`. A
       candidate that fails this is rejected without spending a VLM call on it.
    2. **Semantic agreement** (`vlm_client`, one cheap call on the cropped region) — real
       calibration evidence (see `docs/decisions/0006-grounding-target-validation.md`) shows
       grounding's own confidence score alone cannot separate a genuinely correct
       low-confidence match (`hair`, scored 0.32) from an incorrect one at a similar score
       (`flag_banner`, scored 0.269) — the two real, observed scores are closer to each other
       than either is to an obvious cutoff. An independent signal is required, not a tighter
       number on the same score. This reuses the existing `VLMClient` protocol rather than
       adding a new model dependency, and is the same "cheap VLM-based visual sanity check on
       the grounded crop" `docs/phase3-results.md` already identified as the fix for this
       exact failure.
    3. **Transform-aware geometric compatibility** (deterministic, no model call) — see
       `validation/transform_geometry.py::check_transform_geometry` and
       `docs/decisions/0008-transform-aware-target-validation.md`. "Is this a plausible X" and
       "is this specific box safe to animate with the plan's transform_kind" are independent
       questions — a real, observed failure (`eval_weapon_effects.png`, Phase 3.3's real E2E
       run) had a candidate pass check 2 (a real weapon was visible in the crop) while its bbox
       was far too large to safely `rotate`, visibly swinging the whole panel instead of the
       object. Runs only after check 2 accepts (skipped entirely on a semantic mismatch — no
       point scoring geometry for a candidate that's already the wrong object).

    Never raises — a REJECT is a normal, expected outcome (see the Phase 3.2 acceptance
    criterion: "a correct REJECT is a successful result"), always returned as a
    `ValidationResult`, never silently promoted to an accept. An unparseable VLM response is
    treated as a REJECT (fail closed), never swallowed into a false accept.

    `panel_bbox_px`: the object's real panel region in page-pixel space, if known — used as
    check 3's reference region for area-fraction/edge-margin bounds. Falls back to the full
    image when `None` (page-level analysis, or a caller that hasn't threaded panel geometry
    through yet), identical to how a page-level `AnimationPlan`'s single `(0, 0, 1, 1)`
    `PanelPlan` already behaves everywhere else in this pipeline.
    """
    h, w = image.shape[0], image.shape[1]
    plausible, area_fraction = _bbox_plausibility(grounding.bbox, (h, w))
    if not plausible:
        reason = (
            f"bbox covers {area_fraction:.4%} of the image, outside the plausible "
            f"[{MIN_OBJECT_COVERAGE_FRACTION:.4%}, {MAX_OBJECT_COVERAGE_FRACTION:.0%}] range "
            "for a specific object -- rejected before spending a VLM call"
        )
        logger.info(
            "validation stage: object_id=%s candidate_rank=%d REJECT (bbox implausible): %s",
            object_plan.object_id,
            candidate_rank,
            reason,
        )
        return ValidationResult(
            object_id=object_plan.object_id,
            candidate_rank=candidate_rank,
            accepted=False,
            grounding_score=grounding.bbox.score,
            bbox_area_fraction=area_fraction,
            bbox_plausible=False,
            semantic_match=None,
            semantic_confidence=None,
            reason=reason,
            model_id="none",
            transform_compatible=None,
        )

    crop = _crop_with_margin(image, grounding.bbox)
    prompt = _build_verification_prompt(object_plan)
    raw_text = vlm_client.generate(Image.fromarray(crop), prompt)
    verification = _parse_verification(raw_text)
    model_id = _client_model_id(vlm_client)

    if verification is None:
        reason = "VLM verification response could not be parsed -- rejected (fail closed)"
        logger.info(
            "validation stage: object_id=%s candidate_rank=%d REJECT: %s",
            object_plan.object_id,
            candidate_rank,
            reason,
        )
        return ValidationResult(
            object_id=object_plan.object_id,
            candidate_rank=candidate_rank,
            accepted=False,
            grounding_score=grounding.bbox.score,
            bbox_area_fraction=area_fraction,
            bbox_plausible=True,
            semantic_match=None,
            semantic_confidence=None,
            reason=reason,
            model_id=model_id,
            transform_compatible=None,
        )

    logger.info(
        "validation stage: object_id=%s candidate_rank=%d semantic %s (semantic_match=%s "
        "confidence=%.2f): %s",
        object_plan.object_id,
        candidate_rank,
        "ACCEPT" if verification.matches else "REJECT",
        verification.matches,
        verification.confidence,
        verification.reason,
    )

    if not verification.matches:
        return ValidationResult(
            object_id=object_plan.object_id,
            candidate_rank=candidate_rank,
            accepted=False,
            grounding_score=grounding.bbox.score,
            bbox_area_fraction=area_fraction,
            bbox_plausible=True,
            semantic_match=False,
            semantic_confidence=verification.confidence,
            reason=verification.reason,
            model_id=model_id,
            transform_compatible=None,
        )

    # Semantic check passed -- check 3: is this specific box geometrically safe for the plan's
    # transform_kind? (see check_transform_geometry's docstring for the real defect this
    # catches). Objects with no motion spec (not expected for a PRIMARY object the schema
    # guarantees a MotionSpec for, but validate_target is a generic function -- defensive) skip
    # this check entirely, matching _build_verification_prompt's own defensive handling above.
    if object_plan.motion is not None:
        h_img, w_img = image.shape[0], image.shape[1]
        transform_compatible, geometry_reason = check_transform_geometry(
            grounding.bbox,
            object_plan.motion.transform_kind,
            panel_bbox_px=panel_bbox_px,
            image_shape=(h_img, w_img),
        )
        logger.info(
            "validation stage: object_id=%s candidate_rank=%d geometry %s: %s",
            object_plan.object_id,
            candidate_rank,
            "ACCEPT" if transform_compatible else "REJECT",
            geometry_reason,
        )
        if not transform_compatible:
            return ValidationResult(
                object_id=object_plan.object_id,
                candidate_rank=candidate_rank,
                accepted=False,
                grounding_score=grounding.bbox.score,
                bbox_area_fraction=area_fraction,
                bbox_plausible=True,
                semantic_match=True,
                semantic_confidence=verification.confidence,
                reason=geometry_reason,
                model_id=model_id,
                transform_compatible=False,
            )
    else:
        transform_compatible = None

    return ValidationResult(
        object_id=object_plan.object_id,
        candidate_rank=candidate_rank,
        accepted=True,
        grounding_score=grounding.bbox.score,
        bbox_area_fraction=area_fraction,
        bbox_plausible=True,
        semantic_match=True,
        semantic_confidence=verification.confidence,
        reason=verification.reason,
        model_id=model_id,
        transform_compatible=transform_compatible,
    )
