"""Deterministic prompt construction: Qwen object descriptions -> AnimateAnything prompt.

The generative animation engine consumes a single natural-language prompt (the upstream model
is a text-conditioned video diffusion model). The pipeline's semantic signal is the accepted
per-object `ObjectDescriptionResult`s, so this module deterministically composes one prompt
from the PRIMARY object's read plus the accepted SECONDARY reads -- PRIMARY first (the
readable action), then the others, joined in plan order. Everything is derived from fields the
VLM already produced (`object_identity`, `motion_spec.transform_kind`); nothing is invented
here, which keeps the prompt reproducible from the persisted `descriptions.json` checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

from manga_animation.pipeline.types import ObjectDescriptionResult
from manga_animation.schemas.animation_plan import MotionType, ObjectPlan, TransformKind

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


def _object_phrase(obj: ObjectPlan, description: ObjectDescriptionResult) -> str:
    identity = description.object_identity or obj.semantic_label
    identity = identity.replace("_", " ")
    return f"{identity} {motion_phrase(description)}"


def build_animation_prompt(
    objects: Sequence[tuple[ObjectPlan, ObjectDescriptionResult]],
) -> str:
    """Compose the AnimateAnything text prompt from accepted objects, PRIMARY first.

    `objects` must already be the accepted, ranked list (plan order, PRIMARY first). The prompt
    is `"<primary phrase>, <secondary phrase>, ..."` -- a flat comma-joined instruction whose
    first element is the readable PRIMARY action. Empty input raises `ValueError` (fail closed;
    the caller must never feed an empty acceptance into the generative engine).
    """
    if not objects:
        raise ValueError("cannot build an AnimateAnything prompt from an empty object list")
    ordered = sorted(
        objects,
        key=lambda item: (
            0 if item[0].motion_type == MotionType.PRIMARY else 1,
            item[0].object_id,
        ),
    )
    return ", ".join(_object_phrase(obj, description) for obj, description in ordered)
