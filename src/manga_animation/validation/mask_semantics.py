"""Post-segmentation semantic mask validation: does this exact SAM mask correspond to the

intended semantic object -- not "is this a plausible box for it" (`validate.py::validate_target`,
pre-segmentation, bbox-level) but "does the mask's own silhouette/content match the label, in
full" (Phase 12, post-segmentation, mask-level).

Directly motivated by Phase 11's confirmed, unresolved finding (`docs/phase11-results.md`
section 6.4): SAM 2.1 can produce a mask that is geometrically unremarkable -- it passes every
existing check (bbox coverage bounds, `segmentation/segment.py`'s edge-asymmetry test,
`pipeline/orchestrator.py`'s cross-object overlap guard) -- but semantically wrong, capturing
substantially more or different real content than its assigned label. Four of twelve real
objects investigated: a "cloth" mask that also covered a full speech bubble and a hand; a
"bicycle wheel" mask that was an incoherent stripe through a face, spokes, and a jersey; two
"character_hair" masks that each covered most of a face, or a creature's head plus a large
background region. Phase 11 tested three purely-geometric candidate signals (fragmentation,
density, aspect ratio, convex-hull solidity) against this exact real data and none separated
the confirmed-defective masks from the rest (`docs/phase11-results.md` section 7) -- the
failure is semantic, not geometric, so this module's primary method is a second, cheap VLM
crop-verification call (reusing the same `VLMClient` protocol `validate.py` already uses,
Method B in `docs/phase12-results.md`'s candidate-method comparison), not a new threshold.

See `docs/decisions/0018-semantic-mask-validation.md` for the full design rationale, including
why this sits AFTER segmentation rather than folded into `validate.py` (no mask exists at that
point in the pipeline -- see that module's own docstring) and the calibration limitations of
the ABSTAIN confidence band below (12 real labeled objects is not enough to statistically tune
a threshold; the band is evidenced-but-not-calibrated, same status as this codebase's other
deterministic thresholds).

Only `verify_mask_semantics` is the public entry point.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import (
    BBoxPx,
    ImageArray,
    MaskArray,
    MaskSemanticResult,
    MaskSemanticVerdict,
)
from manga_animation.schemas.animation_plan import ObjectPlan
from manga_animation.validation.validate import _client_model_id, _extract_json_object

logger = get_logger(__name__)

METHOD_ID = "vlm_mask_crop_v1"

_MARGIN_FRACTION = 0.15
"""Same convention as `validate.py`'s own crop margin -- surrounding context for the VLM to
judge against, not a pass/fail threshold."""

_OUTSIDE_MASK_DIM_FACTOR = 0.35
"""How much to darken pixels OUTSIDE the mask within the crop, so the VLM can see exactly which
region is being asked about while the surrounding context (for spotting accidentally-included
neighboring content, e.g. a speech bubble just outside the object) stays faintly visible rather
than being cropped away entirely. An image-prep choice, not a calibrated number -- same status
as `validate.py`'s `_MARGIN_FRACTION`."""

_ABSTAIN_CONFIDENCE_BAND = (0.4, 0.6)
"""Real evidence limitation (see `docs/phase12-results.md`'s calibration-study section,
Workstream 5): only 12 real labeled objects exist for this method (4 confirmed-bad, 8
presumed-good) -- far too few to statistically calibrate a numeric confidence threshold. This
band is a documented, evidenced-but-NOT-statistically-calibrated placeholder (same status as
`transform_geometry.py`'s bounds) marking only VLM reads close to a coin flip as ABSTAIN rather
than forcing a binary call on a near-50/50 read. Widening/narrowing this band responsibly needs
materially more real labeled data than currently exists."""

_VERIFICATION_PROMPT_TEMPLATE = """You are checking whether a SEGMENTATION MASK for a manga/\
comic page actually corresponds to one specific intended object, before that exact mask is \
animated. The image shows a cropped region of the page; the BRIGHT area is the mask being \
checked, the DARKENED surrounding area is context only (not part of what you are judging).

Manga segmentation can sometimes produce a mask that is geometrically fine but semantically \
wrong -- covering the intended object plus something else it should not include. This is NOT \
always the case: most masks are correct. Look carefully at what is ACTUALLY bright in THIS \
specific image before answering -- do not assume a defect is present.

Target object: "{semantic_label}"

Does the bright region show ONLY "{semantic_label}", with nothing else of note included? \
Answer with ONLY one JSON object, no prose, no markdown fences, in exactly this shape:
{{"mask_matches_object": true or false, "confidence": a float 0-1, "unexpected_content": \
["short label", ...] (empty list if none), "reason": "one short sentence describing what you \
actually see in the bright region"}}

Only answer false if the bright region visibly, concretely includes something beyond the \
target object itself -- name exactly what you see in unexpected_content, based on this \
specific image, not a generic guess."""


class _MaskVerificationResponse(BaseModel):
    """The VLM's structured read on one masked candidate region."""

    mask_matches_object: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unexpected_content: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


def _build_mask_verification_prompt(object_plan: ObjectPlan) -> str:
    label = object_plan.semantic_label.replace("_", " ")
    return _VERIFICATION_PROMPT_TEMPLATE.format(semantic_label=label)


def _parse_mask_verification(raw_text: str) -> _MaskVerificationResponse | None:
    """`None` (never an exception) on failure -- an unparseable verification response fails

    closed (rejected), the same convention `validate.py::_parse_verification` already
    established for the pre-segmentation semantic check.
    """
    try:
        object_text = _extract_json_object(raw_text)
        data = json.loads(object_text)
        return _MaskVerificationResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logger.info(
            "mask_semantics stage: VLM verification response could not be parsed (%s); "
            "rejecting this mask (fail closed)",
            exc,
        )
        return None


def _crop_with_mask_overlay(
    image: ImageArray, mask: MaskArray, bbox: BBoxPx, margin_frac: float = _MARGIN_FRACTION
) -> ImageArray:
    """A margin-padded crop around `bbox` with everything OUTSIDE `mask` dimmed -- see

    `_OUTSIDE_MASK_DIM_FACTOR`. Highlights exactly what the mask covers without discarding
    surrounding context entirely, so the VLM can judge both "is the bright region the target"
    and "did the mask reach into something nearby it shouldn't have".
    """
    h, w = image.shape[0], image.shape[1]
    mx = max(1, round(bbox.width * margin_frac))
    my = max(1, round(bbox.height * margin_frac))
    x0 = max(0, bbox.x0 - mx)
    y0 = max(0, bbox.y0 - my)
    x1 = min(w, bbox.x1 + mx)
    y1 = min(h, bbox.y1 + my)

    crop = image[y0:y1, x0:x1].astype(np.float32)
    mask_crop = (mask[y0:y1, x0:x1] > 0)[..., None]
    dimmed = crop * _OUTSIDE_MASK_DIM_FACTOR
    highlighted = np.where(mask_crop, crop, dimmed)
    return np.clip(highlighted, 0, 255).astype(np.uint8)


def _compute_geometric_signals(mask: MaskArray, bbox: BBoxPx) -> dict[str, float]:
    """Phase 11's own candidate geometric signals, recomputed here for every result as

    non-gating diagnostic/forensic evidence (see `MaskSemanticResult.geometric_signals`'s
    docstring for why these are recorded but never decide the verdict).
    """
    binary = (mask > 0).astype(np.uint8)
    tight = binary[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    total_area = int(binary.sum())
    if total_area == 0 or tight.size == 0:
        return {
            "second_component_area_fraction": 0.0,
            "bbox_density": 0.0,
            "aspect_ratio": 1.0,
            "convex_hull_solidity": 1.0,
        }

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = sorted(
        (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)), reverse=True
    )
    second_component_fraction = (
        (component_areas[1] / total_area) if len(component_areas) > 1 else 0.0
    )

    bbox_density = float(tight.sum()) / tight.size
    aspect_ratio = max(bbox.width, bbox.height) / max(1, min(bbox.width, bbox.height))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        hull_area = float(cv2.contourArea(hull))
        solidity = (total_area / hull_area) if hull_area > 0 else 1.0
    else:
        solidity = 1.0

    return {
        "second_component_area_fraction": float(second_component_fraction),
        "bbox_density": bbox_density,
        "aspect_ratio": float(aspect_ratio),
        "convex_hull_solidity": float(min(solidity, 1.0)),
    }


def _classify(verification: _MaskVerificationResponse) -> tuple[MaskSemanticVerdict, str]:
    lo, hi = _ABSTAIN_CONFIDENCE_BAND
    if lo <= verification.confidence <= hi:
        return "abstain", (
            f"VLM confidence {verification.confidence:.2f} falls inside the near-coin-flip "
            f"[{lo:.1f}, {hi:.1f}] band -- insufficient evidence either way (VLM's own reason: "
            f"{verification.reason})"
        )
    if verification.mask_matches_object:
        return "accept", verification.reason
    return "reject", verification.reason


def verify_mask_semantics(
    image: ImageArray,
    object_plan: ObjectPlan,
    mask: MaskArray,
    bbox: BBoxPx,
    vlm_client: VLMClient,
) -> MaskSemanticResult:
    """ACCEPT/REJECT/ABSTAIN one segmented mask against `object_plan.semantic_label`.

    Two independent pieces of evidence are gathered, only one of which decides the verdict:

    1. **Geometric signals** (`_compute_geometric_signals`, deterministic, no model call) --
       computed and attached to every result for forensic/explainability value, but never
       gating (Phase 11 found none of them reliably separates real defective masks from
       legitimate ones; see this module's docstring).
    2. **VLM mask-crop verification** (`vlm_client`, one cheap call on a crop that highlights
       exactly the masked region) -- the actual decision signal. Fail-closed on an unparseable
       response (never a silent accept), and ABSTAIN on a near-50/50 confidence read rather than
       forcing a binary call the evidence doesn't support (see `_ABSTAIN_CONFIDENCE_BAND`).

    Never raises -- REJECT/ABSTAIN are normal, expected outcomes, always returned as a
    `MaskSemanticResult`.
    """
    geometric_signals = _compute_geometric_signals(mask, bbox)
    crop = _crop_with_mask_overlay(image, mask, bbox)
    prompt = _build_mask_verification_prompt(object_plan)
    raw_text = vlm_client.generate(Image.fromarray(crop), prompt)
    verification = _parse_mask_verification(raw_text)
    model_id = _client_model_id(vlm_client)

    if verification is None:
        result = MaskSemanticResult(
            object_id=object_plan.object_id,
            verdict="reject",
            vlm_matches=None,
            vlm_confidence=None,
            reason="VLM mask verification response could not be parsed -- rejected (fail closed)",
            model_id=model_id,
            method=METHOD_ID,
            geometric_signals=geometric_signals,
        )
    else:
        verdict, reason = _classify(verification)
        result = MaskSemanticResult(
            object_id=object_plan.object_id,
            verdict=verdict,
            vlm_matches=verification.mask_matches_object,
            vlm_confidence=verification.confidence,
            reason=reason,
            model_id=model_id,
            method=METHOD_ID,
            unexpected_content=tuple(verification.unexpected_content),
            geometric_signals=geometric_signals,
        )

    logger.info(
        "mask_semantics stage: object_id=%s semantic_label=%s verdict=%s (vlm_matches=%s "
        "confidence=%s): %s",
        object_plan.object_id,
        object_plan.semantic_label,
        result.verdict,
        result.vlm_matches,
        result.vlm_confidence,
        result.reason,
    )
    return result
