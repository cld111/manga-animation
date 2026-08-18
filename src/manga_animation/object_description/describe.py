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
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
from pydantic import ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.analysis.plan_builder import _extract_json_array
from manga_animation.core.logging import get_logger
from manga_animation.object_description.mapping import motion_spec_from_description
from manga_animation.object_description.prompt import (
    PROMPT_MARKER,
    PromptCandidate,
    build_multi_prompt,
    prepare_image_and_bbox,
)
from manga_animation.object_description.schema import (
    AmplitudeBand,
    BBoxAssessment,
    ObjectDescriptionResponse,
    PivotHint,
    SpeedBand,
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
required JSON array. Error: {error}

Return ONLY the corrected JSON ARRAY of the same candidate objects (one per candidate box, \
each with "box_index" matching the [index] from the prompt). Each object has exactly these \
fields: "box_index" (the candidate index), "bbox_assessment" (one of "pass" | "ambiguous" | \
"partial" | "reject" | "not_animatable"), "object_identity" ("short snake_case name"), \
"matches_semantic_label" (true or false), "animatable" (true or false), "movable_parts" \
([...]), "static_parts" ([...]), "motion_kind" (null or one of "sway"|"flow"|"drift"|"rotate"|\
"pulse"|"breathe"|"flicker"), "direction" (null or one of "up"|"down"|"left"|"right"|"up_left"|\
"up_right"|"down_left"|"down_right"), "amplitude_band" ("subtle"|"moderate"|"pronounced"), \
"speed_band" ("slow"|"normal"|"fast"), "pivot_hint" ("top"|"center"|"bottom"), "constraints" \
([...]), "neighbor_conflicts" ([...]), "confidence" (a float 0-1), "reason" ("one short \
sentence")).

Rules: "motion_kind" is required iff "animatable" is true; "direction" is required iff \
"motion_kind" is "drift", otherwise "direction" is null. Use ONLY the listed enum values -- \
never anything else. Cover EVERY candidate exactly once. No prose, no markdown fences."""


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


def _parse_batch(raw_text: str, n_expected: int) -> dict[int, ObjectDescriptionResponse] | None:
    """Parse a batch answer: a JSON ARRAY with one entry per candidate, mapped back by
    `box_index`. `None` on failure; entries with missing/duplicate box indices are rejected
    by the caller per-candidate (fail closed), not silently merged."""
    try:
        array_text = _extract_json_array(raw_text)
        data = json.loads(array_text)
        if not isinstance(data, list):
            raise ValueError("batch answer is not a JSON array")
        parsed: dict[int, ObjectDescriptionResponse] = {}
        for item in data:
            entry = ObjectDescriptionResponse.model_validate(item)
            if entry.box_index in parsed:
                raise ValueError(f"duplicate box_index={entry.box_index}")
            parsed[entry.box_index] = entry
        if len(parsed) < n_expected:
            raise ValueError(
                f"batch answer covers {len(parsed)} of {n_expected} candidate boxes"
            )
        return parsed
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        logger.info(
            "object_description stage: batch VLM response could not be parsed (%s); failing "
            "closed",
            exc,
        )
        return None


def _accepted(parsed: ObjectDescriptionResponse) -> tuple[bool, str | None]:
    """The fail-closed acceptance rule, with the exact reason for a rejection. `rejection_reason`
    is machine-readable ("bbox_assessment=...", "semantic_label_mismatch", "not_animatable",
    "identity_conflict").

    Since the 2026 architecture change (generative per-object animation from a DINO bbox crop)
    both `pass` and `partial` are accepted, provided the object is still animatable and matches
    its semantic label: a `partial` box captures only part of the object, which is fine for a
    generative crop animation (the crop shows the object; no exact mask is needed). `ambiguous`,
    `reject` and `not_animatable` remain rejections (fail closed)."""
    if parsed.bbox_assessment not in (BBoxAssessment.PASS, BBoxAssessment.PARTIAL):
        return (
            False,
            f"bbox_assessment={parsed.bbox_assessment.value} (must be 'pass' or 'partial')",
        )
    if not parsed.matches_semantic_label:
        return False, "semantic_label_mismatch"
    if not parsed.animatable:
        return False, "not_animatable"
    identity = (parsed.object_identity or "").lower()
    if any(kw in identity for kw in _NON_ANIMATABLE_IDENTITY_KEYWORDS):
        return False, f"identity_conflict={parsed.object_identity}"
    return True, None


# Deterministic backstop for the observed real false-accept (phase18_3_final, villainess):
# a recovery response can say `object_identity: "speech_bubble"` while still claiming
# `matches_semantic_label: true` and `animatable: true` -- the model contradicts itself and
# every soft signal says "accept". Content categories that can NEVER be a valid animation
# target are rejected on the identity string alone, mirroring
# `analysis.plan_builder._is_text_label`'s label-keyed guard. "bubble" matches both speech
# bubbles and thought bubbles; "panel"/"background"/"lettering"/"text" are the other rigid
# or non-object categories.
_NON_ANIMATABLE_IDENTITY_KEYWORDS: tuple[str, ...] = (
    "speech_bubble",
    "speech bubble",
    "thought_bubble",
    "thought bubble",
    "bubble",
    "text",
    "lettering",
    "caption",
    "dialogue",
    "narration",
    "sound_effect",
    "sfx",
    "panel_border",
    "panel",
    "background",
    "border",
)


def _result_from_parsed(
    parsed: ObjectDescriptionResponse | None,
    *,
    object_id: str,
    model_id: str,
    raw_responses: tuple[str, ...],
) -> ObjectDescriptionResult:
    """Assemble the fail-closed `ObjectDescriptionResult` for one candidate from its parsed
    response (or `None` when the batch never produced a usable entry for it). Shared by the
    single-candidate and the batch paths."""
    if parsed is None:
        return ObjectDescriptionResult(
            object_id=object_id,
            accepted=False,
            assessment=None,
            matches_semantic_label=None,
            animatable=None,
            object_identity=None,
            motion_spec=None,
            reason="VLM response remained unparseable after one recovery attempt",
            rejection_reason="unparseable",
            model_id=model_id,
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
                amplitude_band=parsed.amplitude_band or AmplitudeBand.MODERATE,
                speed_band=(parsed.speed_band or SpeedBand.NORMAL).value,
                pivot_hint=parsed.pivot_hint or PivotHint.CENTER,
            )
        except ValueError as exc:
            accepted = False
            rejection_reason = f"motion_mapping_error: {exc}"
            logger.warning(
                "object_description stage: object_id=%s description mapped to an invalid "
                "MotionSpec (%s) -- rejecting this candidate (fail closed)",
                object_id,
                exc,
            )

    logger.info(
        "object_description stage: object_id=%s %s (assessment=%s matches=%s animatable=%s "
        "confidence=%s): %s",
        object_id,
        "ACCEPT" if accepted else "REJECT",
        parsed.bbox_assessment.value,
        parsed.matches_semantic_label,
        parsed.animatable,
        parsed.confidence,
        parsed.reason,
    )

    return ObjectDescriptionResult(
        object_id=object_id,
        accepted=accepted,
        assessment=parsed.bbox_assessment.value,
        matches_semantic_label=parsed.matches_semantic_label,
        animatable=parsed.animatable,
        object_identity=parsed.object_identity,
        motion_spec=motion_spec,
        movable_parts=tuple(parsed.movable_parts or ()),
        static_parts=tuple(parsed.static_parts or ()),
        constraints=tuple(parsed.constraints or ()),
        neighbor_conflicts=tuple(parsed.neighbor_conflicts or ()),
        confidence=parsed.confidence,
        reason=parsed.reason,
        rejection_reason=rejection_reason,
        model_id=model_id,
        raw_responses=raw_responses,
        method=METHOD_ID,
    )


@dataclass(frozen=True, slots=True)
class CandidateBox:
    """One candidate for the batch description call: its plan identity plus the bbox on the
    image it belongs to."""

    object_id: str
    semantic_label: str
    bbox: BBoxPx


def describe_objects(
    image: ImageArray,
    candidates: Sequence[CandidateBox],
    vlm_client: VLMClient,
    *,
    max_long_edge: int,
) -> list[ObjectDescriptionResult]:
    """ONE image + ALL of its candidate bboxes in a single VLM call, fail-closed per
    candidate (Phase 18.3: the model sees the full image and every candidate's pixel
    coordinates at once, never one crop per candidate).

    Returns one `ObjectDescriptionResult` per input candidate (same order). A candidate
    whose box the model's batch answer omits or mis-indexes is failed closed.
    """
    prepared = prepare_image_and_bbox(
        Image.fromarray(image), candidates[0].bbox, max_long_edge=max_long_edge
    )
    # All candidates share the same image; the prepared bboxes are the same scaling.
    scaled_bboxes = [
        BBoxPx(
            x0=round(c.bbox.x0 * prepared.scale_x),
            y0=round(c.bbox.y0 * prepared.scale_y),
            x1=round(c.bbox.x1 * prepared.scale_x),
            y1=round(c.bbox.y1 * prepared.scale_y),
        )
        for c in candidates
    ]
    prompt_candidates = [
        PromptCandidate(
            index=i,
            image_index=0,
            semantic_label=c.semantic_label,
            bbox_px=scaled_bboxes[i],
            image_size=(prepared.image.width, prepared.image.height),
        )
        for i, c in enumerate(candidates)
    ]
    raw_text = vlm_client.generate(
        prepared.image, build_multi_prompt(prompt_candidates)
    )
    logger.info(
        "object_description stage: batch of %d candidate(s) -> raw VLM response: %s",
        len(candidates),
        raw_text,
    )
    parsed = _parse_batch(raw_text, len(candidates))
    raw_responses: tuple[str, ...] = (raw_text,)
    if parsed is None:
        recovery_text = vlm_client.generate(
            prepared.image,
            _RECOVERY_PROMPT_TEMPLATE.format(error="malformed JSON array or schema"),
        )
        logger.info(
            "object_description stage: batch recovery VLM response: %s", recovery_text
        )
        parsed = _parse_batch(recovery_text, len(candidates))
        raw_responses = (raw_text, recovery_text)

    model_id = _client_model_id(vlm_client)
    results: list[ObjectDescriptionResult] = []
    for i, candidate in enumerate(candidates):
        results.append(
            _result_from_parsed(
                parsed.get(i) if parsed is not None else None,
                object_id=candidate.object_id,
                model_id=model_id,
                raw_responses=raw_responses,
            )
        )
    return results


def describe_object(
    image: ImageArray,
    bbox: BBoxPx,
    object_plan: ObjectPlan,
    vlm_client: VLMClient,
    *,
    max_long_edge: int,
) -> ObjectDescriptionResult:
    """Single-candidate convenience wrapper over the batch path (used by tests and the
    benchmark; the pipeline itself uses `describe_objects` so the model sees ALL of one
    image's candidate bboxes in a single call). See `describe_objects` for the fail-closed
    contract.
    """
    return describe_objects(
        image,
        [CandidateBox(object_id=object_plan.object_id,
                      semantic_label=object_plan.semantic_label, bbox=bbox)],
        vlm_client,
        max_long_edge=max_long_edge,
    )[0]


__all__ = ["PROMPT_MARKER", "METHOD_ID", "CandidateBox", "describe_object", "describe_objects"]
