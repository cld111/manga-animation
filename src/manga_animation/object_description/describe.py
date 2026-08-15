"""The Phase 18.3 per-candidate VLM stage: full image + bbox coordinates -> structured
animation description (with strict bbox validation and fail-closed behavior).

The stage's place in the pipeline (see `docs/pipeline.md`): after semantic mask validation,
before animation. For every object that already passed grounding -> validation -> segmentation
-> mask_semantics, the VLM looks at the FULL pipeline image and the candidate bounding box
given as pixel coordinates, judges the candidate itself (`BBoxAssessment`: pass/ambiguous/
partial/reject/not_animatable), and produces a structured animation description
(`ObjectDescriptionResponse`). The deterministic mapping layer converts an accepted
description to a schema-valid `MotionSpec`, and the animation stage actually applies it
(orchestrator `_describe_objects` -> `_animate_objects`). The SAM mask is NOT an input here:
it stays for the later stages that need it.

Fail-closed discipline (mirrors `validation.validate_target`/`mask_semantics`):

- An unparseable or schema-invalid response gets exactly one recovery re-prompt (the same
  one-attempt recovery `analysis.plan_builder._decisions_from_vlm` uses); a second failure is
  a REJECT with a machine-readable `rejection_reason`.
- An accepted result requires `bbox_assessment == "pass"` AND `matches_semantic_label` AND
  `animatable` -- every other combination is a REJECT with the specific reason recorded.
- Every VLM raw response (initial and recovery) is logged and kept in
  `ObjectDescriptionResult.raw_responses` for the audit trail.
"""

from __future__ import annotations

import json

from PIL import Image
from pydantic import ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.core.logging import get_logger
from manga_animation.object_description.mapping import motion_spec_from_description
from manga_animation.object_description.prompt import (
    PROMPT_MARKER,
    build_prompt,
    prepare_image_and_bbox,
)
from manga_animation.object_description.schema import (
    BBoxAssessment,
    ObjectDescriptionResponse,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    ImageArray,
    ObjectDescriptionResult,
)
from manga_animation.schemas.animation_plan import MotionSpec, ObjectPlan
from manga_animation.validation.validate import _client_model_id, _extract_json_object

logger = get_logger(__name__)

METHOD_ID = "vlm_full_image_bbox_v1"

_RECOVERY_PROMPT_TEMPLATE = """Your previous response could not be parsed/validated as the \
required JSON object. Error: {error}

Return ONLY the corrected JSON object with the exact same fields as before ("bbox_assessment", \
"object_identity", "matches_semantic_label", "animatable", "movable_parts", "static_parts", \
"motion_kind", "direction", "amplitude_band", "speed_band", "pivot_hint", "constraints", \
"neighbor_conflicts", "confidence", "reason"). No prose, no markdown fences."""


def _parse_response(raw_text: str) -> ObjectDescriptionResponse | None:
    """`None` (never an exception) on failure -- the caller decides the fail-closed outcome
    and records the reason. Mirrors `validation.validate.py::_parse_verification`."""
    try:
        object_text = _extract_json_object(raw_text)
        data = json.loads(object_text)
        return ObjectDescriptionResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logger.info(
            "object_description stage: VLM response could not be parsed (%s); rejecting this "
            "candidate (fail closed)",
            exc,
        )
        return None


def _accepted(parsed: ObjectDescriptionResponse) -> tuple[bool, str | None]:
    """The fail-closed acceptance rule, with the exact reason for a rejection. `rejection_reason`
    is machine-readable ("bbox_assessment=...", "semantic_label_mismatch", "not_animatable")."""
    if parsed.bbox_assessment != BBoxAssessment.PASS:
        return False, f"bbox_assessment={parsed.bbox_assessment.value} (must be 'pass')"
    if not parsed.matches_semantic_label:
        return False, "semantic_label_mismatch"
    if not parsed.animatable:
        return False, "not_animatable"
    return True, None


def describe_object(
    image: ImageArray,
    bbox: BBoxPx,
    object_plan: ObjectPlan,
    vlm_client: VLMClient,
    *,
    max_long_edge: int,
) -> ObjectDescriptionResult:
    """One full-image + bbox VLM call for one grounded candidate, fail-closed.

    `image` is the pipeline image the bbox lives in (page-level: the page; panel-mode: the
    scene crop) -- the VLM sees exactly this image plus the bbox as pixel coordinates, never
    a crop of the candidate (see `object_description.prompt` for the coordinate contract).

    Returns an `ObjectDescriptionResult`; never raises for a non-accept (a REJECT is the
    normal, expected outcome of this semantic validation layer). Exceptions from the client
    itself propagate to the orchestrator's stage policy (PRIMARY fails / SECONDARY drops).
    """
    prepared = prepare_image_and_bbox(
        Image.fromarray(image), bbox, max_long_edge=max_long_edge
    )
    prompt = build_prompt(prepared=prepared, semantic_label=object_plan.semantic_label)

    raw_text = vlm_client.generate(prepared.image, prompt)
    logger.info(
        "object_description stage: object_id=%s bbox=%s -> raw VLM response: %s",
        object_plan.object_id,
        bbox.as_xyxy(),
        raw_text,
    )
    parsed = _parse_response(raw_text)
    raw_responses: tuple[str, ...] = (raw_text,)
    if parsed is None:
        recovery_prompt = _RECOVERY_PROMPT_TEMPLATE.format(error="malformed JSON or schema")
        recovery_text = vlm_client.generate(prepared.image, recovery_prompt)
        logger.info(
            "object_description stage: object_id=%s recovery VLM response: %s",
            object_plan.object_id,
            recovery_text,
        )
        parsed = _parse_response(recovery_text)
        raw_responses = (raw_text, recovery_text)
        if parsed is None:
            return ObjectDescriptionResult(
                object_id=object_plan.object_id,
                accepted=False,
                assessment=None,
                matches_semantic_label=None,
                animatable=None,
                object_identity=None,
                motion_spec=None,
                reason="VLM response remained unparseable after one recovery attempt",
                rejection_reason="unparseable",
                model_id=_client_model_id(vlm_client),
                raw_responses=raw_responses,
                method=METHOD_ID,
            )

    accepted, rejection_reason = _accepted(parsed)
    motion_spec: MotionSpec | None = None
    if accepted and parsed.motion_kind is not None:
        try:
            motion_spec = motion_spec_from_description(
                motion_kind=parsed.motion_kind,
                direction=parsed.direction,
                amplitude_band=parsed.amplitude_band,
                speed_band=parsed.speed_band.value,
                pivot_hint=parsed.pivot_hint,
            )
        except ValueError as exc:
            accepted = False
            rejection_reason = f"motion_mapping_error: {exc}"
            logger.warning(
                "object_description stage: object_id=%s description mapped to an invalid "
                "MotionSpec (%s) -- rejecting this candidate (fail closed)",
                object_plan.object_id,
                exc,
            )

    logger.info(
        "object_description stage: object_id=%s %s (assessment=%s matches=%s animatable=%s "
        "confidence=%s): %s",
        object_plan.object_id,
        "ACCEPT" if accepted else "REJECT",
        parsed.bbox_assessment.value,
        parsed.matches_semantic_label,
        parsed.animatable,
        parsed.confidence,
        parsed.reason,
    )

    return ObjectDescriptionResult(
        object_id=object_plan.object_id,
        accepted=accepted,
        assessment=parsed.bbox_assessment.value,
        matches_semantic_label=parsed.matches_semantic_label,
        animatable=parsed.animatable,
        object_identity=parsed.object_identity,
        motion_spec=motion_spec,
        movable_parts=tuple(parsed.movable_parts),
        static_parts=tuple(parsed.static_parts),
        constraints=tuple(parsed.constraints),
        neighbor_conflicts=tuple(parsed.neighbor_conflicts),
        confidence=parsed.confidence,
        reason=parsed.reason,
        rejection_reason=rejection_reason,
        model_id=_client_model_id(vlm_client),
        raw_responses=raw_responses,
        method=METHOD_ID,
    )


__all__ = ["PROMPT_MARKER", "METHOD_ID", "describe_object"]
