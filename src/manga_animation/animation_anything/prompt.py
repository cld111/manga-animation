"""Deterministic prompt construction: Qwen object description -> AnimateAnything prompt.

The generative animation engine consumes a single natural-language prompt (the upstream model
is a text-conditioned video diffusion model). The pipeline's semantic signal is the accepted
per-object `ObjectDescriptionResult`, so this module deterministically composes one prompt
from a single accepted object's read: `object_identity` (short snake_case name) plus the
model's own plain-language `reason` sentence, plus a motion phrase mapped from the accepted
description's `motion_spec.transform_kind`. Nothing is invented here -- every token comes from
fields the VLM already produced, which keeps the prompt reproducible from the persisted
`descriptions.json` checkpoint.
"""

from __future__ import annotations

from manga_animation.pipeline.types import ObjectDescriptionResult
from manga_animation.schemas.animation_plan import ObjectPlan, TransformKind

# transform_kind -> plain-language motion phrase. These read like instructions a video-diffusion
# model can follow ("flowing", "swaying") rather than kinematic jargon.
_MOTION_PHRASES: dict[TransformKind, str] = {
    TransformKind.TRANSLATE: "drifting",
    TransformKind.ROTATE: "rotating",
    TransformKind.SCALE: "breathing",
    TransformKind.SHEAR: "shearing",
    TransformKind.MESH_WARP: "flowing",
    TransformKind.OPACITY: "flickering",
    TransformKind.RADIAL_EXPAND: "pulsing outward",
}


def motion_phrase(description: ObjectDescriptionResult) -> str:
    """The plain-language motion phrase for one accepted description's mapped transform kind."""
    if description.motion_spec is None:
        return "moving"
    return _MOTION_PHRASES.get(description.motion_spec.transform_kind, "moving")


# Detected from the VLM's own text (identity/reason), never invented here: a bicycle context
# deserves an explicit pedaling/wheel-spinning instruction that the generic motion phrase
# would otherwise leave implicit. Matched case-insensitively on the joined identity+reason.
_BICYCLE_HINTS: tuple[str, ...] = ("bicycle", "bike", "cycling", "cyclist")


def _bicycle_clause(text: str) -> str | None:
    lowered = text.lower()
    if any(hint in lowered for hint in _BICYCLE_HINTS):
        return "the character is pedaling, the bicycle wheels are spinning"
    return None


def build_animation_prompt(obj: ObjectPlan, description: ObjectDescriptionResult) -> str:
    """Compose the AnimateAnything text prompt for ONE accepted object (its DINO bbox crop).

    The prompt is a direct MOTION instruction: `"<identity> <motion>, <reason>. static
    camera, no camera movement"`. The identity is the VLM's short object name (underscores
    replaced by spaces), the motion phrase is mapped from the accepted description's
    transform kind, and the reason is the VLM's own one-sentence read of the action. For a
    bicycle context the prompt additionally states that the character is pedaling and the
    wheels are spinning (deterministic, matched on the VLM's own text). The trailing
    "static camera" clause tells the model to move the OBJECT, not the camera. An empty
    identity falls back to the semantic label (never invented here).
    """
    identity = description.object_identity or obj.semantic_label
    identity = identity.replace("_", " ")
    phrase = f"{identity} {motion_phrase(description)}"
    if description.reason:
        reason = description.reason.strip().rstrip(".")
        if reason and reason.lower() not in phrase.lower():
            phrase = f"{phrase}, {reason}"
    bicycle_clause = _bicycle_clause(f"{identity} {description.reason or ''}")
    if bicycle_clause:
        phrase = f"{phrase}, {bicycle_clause}"
    return f"{phrase}. static camera, no camera movement"
