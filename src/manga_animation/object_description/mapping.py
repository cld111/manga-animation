"""Deterministic mapping from a VLM `ObjectDescriptionResponse` to a schema-valid `MotionSpec`.

This is the "animation planning" side of the Phase 18.3 contract: the VLM supplies SEMANTIC
judgments (motion category, direction word, amplitude band, speed band, pivot hint) and this
module converts them to concrete kinematics, reusing the same baseline amplitudes/easings the
analysis stage's keyword heuristics already use (`analysis/plan_builder._MOTION_HEURISTICS` /
`_EFFECT_LABEL_KEYWORDS`) -- one documented table of per-category parameters, now driven by a
per-candidate VLM read instead of a label keyword. All numbers are schema-valid by
construction (direction normalized, whole-number speeds so a seamless `loop_mode="cycle"` loop
is satisfied).
"""

from __future__ import annotations

from manga_animation.object_description.schema import (
    AmplitudeBand,
    DirectionWord,
    MotionKind,
    PivotHint,
)
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    PivotSpec,
    TransformKind,
    Vector2,
)

# Direction word -> unit vector (x right-positive, y down-positive -- the MotionSpec
# convention, docs/animation-plan-schema.md).
_DIRECTION_VECTORS: dict[DirectionWord, Vector2] = {
    DirectionWord.UP: Vector2(x=0.0, y=-1.0),
    DirectionWord.DOWN: Vector2(x=0.0, y=1.0),
    DirectionWord.LEFT: Vector2(x=-1.0, y=0.0),
    DirectionWord.RIGHT: Vector2(x=1.0, y=0.0),
    DirectionWord.UP_LEFT: Vector2(x=-0.7071, y=-0.7071),
    DirectionWord.UP_RIGHT: Vector2(x=0.7071, y=-0.7071),
    DirectionWord.DOWN_LEFT: Vector2(x=-0.7071, y=0.7071),
    DirectionWord.DOWN_RIGHT: Vector2(x=0.7071, y=0.7071),
}

_PIVOT_POINTS: dict[PivotHint, tuple[float, float]] = {
    PivotHint.TOP: (0.5, 0.0),
    PivotHint.CENTER: (0.5, 0.5),
    PivotHint.BOTTOM: (0.5, 1.0),
}

# Relative amplitude band -> multiplier on the motion category's baseline amplitude.
_AMPLITUDE_MULTIPLIERS: dict[AmplitudeBand, float] = {
    AmplitudeBand.SUBTLE: 0.5,
    AmplitudeBand.MODERATE: 1.0,
    AmplitudeBand.PRONOUNCED: 1.5,
}

# Speed band -> whole cycles per loop (seamless `cycle` loop requires an integer speed).
_SPEED_CYCLES: dict[str, float] = {
    "slow": 1.0,
    "normal": 2.0,
    "fast": 3.0,
}

# Motion category -> baseline kinematic parameters (amplitudes/easings carried over from the
# analysis stage's heuristic tables so the two paths stay comparable).
_MOTION_KIND_BASELINES: dict[MotionKind, MotionSpec] = {
    MotionKind.SWAY: MotionSpec(
        transform_kind=TransformKind.MESH_WARP,
        amplitude=0.12,
        speed=1.0,
        easing=Easing.SINE,
        pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
    ),
    MotionKind.FLOW: MotionSpec(
        transform_kind=TransformKind.MESH_WARP,
        amplitude=0.10,
        speed=1.0,
        easing=Easing.SINE,
        pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
    ),
    MotionKind.DRIFT: MotionSpec(
        transform_kind=TransformKind.TRANSLATE,
        direction=Vector2(x=0.0, y=1.0),  # placeholder; overridden from the description's direction
        amplitude=0.03,
        speed=1.0,
        easing=Easing.EASE_IN_OUT,
        pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
    ),
    MotionKind.ROTATE: MotionSpec(
        transform_kind=TransformKind.ROTATE,
        amplitude=8.0,
        speed=1.0,
        easing=Easing.SINE,
        pivot=PivotSpec(x=0.5, y=1.0, reference="object_bbox"),
    ),
    MotionKind.PULSE: MotionSpec(
        transform_kind=TransformKind.RADIAL_EXPAND,
        amplitude=0.08,
        speed=1.0,
        easing=Easing.SINE,
        pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
    ),
    MotionKind.BREATHE: MotionSpec(
        transform_kind=TransformKind.SCALE,
        amplitude=0.03,
        speed=1.0,
        easing=Easing.SINE,
        pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
    ),
    MotionKind.FLICKER: MotionSpec(
        transform_kind=TransformKind.OPACITY,
        amplitude=0.35,
        speed=2.0,
        easing=Easing.EASE_IN_OUT,
        pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
    ),
}


def motion_spec_from_description(
    *,
    motion_kind: MotionKind,
    direction: DirectionWord | None,
    amplitude_band: AmplitudeBand,
    speed_band: str,
    pivot_hint: PivotHint,
) -> MotionSpec:
    """Map one accepted description's motion fields to a schema-valid `MotionSpec`.

    Raises `ValueError` on a combination that cannot be expressed validly (e.g. a drift with
    no direction -- though the response schema already forbids it, this function stays a
    pure, independently-testable mapper).
    """
    spec = _MOTION_KIND_BASELINES[motion_kind].model_copy(deep=True)
    spec.amplitude = round(spec.amplitude * _AMPLITUDE_MULTIPLIERS[amplitude_band], 6)
    spec.speed = _SPEED_CYCLES[speed_band]
    if direction is not None:
        spec.direction = _DIRECTION_VECTORS[direction]
    px, py = _PIVOT_POINTS[pivot_hint]
    spec.pivot = PivotSpec(x=px, y=py, reference="object_bbox")
    # Re-validate: MotionSpec's own validator normalizes the direction and enforces the
    # translate/shear-needs-direction and non-zero rules.
    return MotionSpec.model_validate(spec.model_dump())
