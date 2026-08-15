"""Structured response contract for the per-candidate VLM object-description call.

Phase 18.3: Qwen2.5-VL sees the FULL pipeline image plus a candidate bounding box given as
pixel coordinates and must (a) judge the candidate region itself (one coherent object?
several objects? part of an object? background? unsafe to animate?), and (b) produce a
structured animation description that the deterministic mapping layer
(`object_description.mapping`) turns into a schema-valid `MotionSpec`.

This module is deliberately pure pydantic/numpy-free: it is the model-facing contract and is
unit-tested without any ML dependency. The strict fail-closed behavior lives in
`object_description.describe`; this module only defines *what a valid answer is*.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from manga_animation.schemas.animation_plan import TransformKind


class BBoxAssessment(StrEnum):
    """The VLM's judgment of the candidate bounding box itself.

    Deliberately finer than a single boolean -- the task brief explicitly requires the VLM to
    act as a semantic validation layer between grounding and animation, and these five states
    are the distinct, mutually exclusive outcomes that can be expressed (see
    `describe.py::_accepted` for how each maps to pipeline behavior):

    - `PASS`: the box contains exactly one coherent instance of the intended object, with
      enough surrounding-context understanding to claim it is a good single-object candidate.
    - `AMBIGUOUS`: the box contains several objects, or the intended target is unclear (e.g. a
      character next to a weapon, several visually similar instances).
    - `PARTIAL`: the box captures only part of the object (a limb, a fragment), not the whole.
    - `REJECT`: the box is mostly background or otherwise does not contain a coherent object.
    - `NOT_ANIMATABLE`: the box does contain an identifiable object, but animating it is not
      safe (rigid scene element, text-like content, occlusion that makes motion ambiguous).
    """

    PASS = "pass"
    AMBIGUOUS = "ambiguous"
    PARTIAL = "partial"
    REJECT = "reject"
    NOT_ANIMATABLE = "not_animatable"


class MotionKind(StrEnum):
    """The semantic motion category, in plain language the VLM is asked to choose from.

    The deterministic mapping layer (`mapping.py`) converts each category to a concrete
    `TransformKind` + baseline kinematic parameters -- the VLM never emits raw kinematics
    numbers (consistent with the project's "deterministic CV, semantic VLM" principle; see
    `docs/architecture.md`). The categories are named in terms a reader of the artwork would
    use, not in terms of the implementation:

    - `SWAY`: gentle back-and-forth oscillation around a point (hair, cloth, flags).
    - `FLOW`: continuous deforming drift (smoke, water, speed lines, energy wisps).
    - `DRIFT`: steady movement in one direction (rain, particles, a drifting object).
    - `ROTATE`: rotation around a pivot (a raised weapon, an arm swing).
    - `PULSE`: radially symmetric expansion/contraction from the object's own center (impact
      bursts, glow, energy fields -- the drawn-effect motion model).
    - `BREATHE`: slow uniform scale oscillation (an energy aura, a balloon).
    - `FLICKER`: rapid on/off or intensity variation (blink, sparks).
    """

    SWAY = "sway"
    FLOW = "flow"
    DRIFT = "drift"
    ROTATE = "rotate"
    PULSE = "pulse"
    BREATHE = "breathe"
    FLICKER = "flicker"


class DirectionWord(StrEnum):
    """Direction of a `DRIFT` motion, as a word -- mapped deterministically to a unit vector."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"


class AmplitudeBand(StrEnum):
    """Relative motion amplitude, as a coarse band -- mapped to a multiplier on the motion
    kind's baseline amplitude."""

    SUBTLE = "subtle"
    MODERATE = "moderate"
    PRONOUNCED = "pronounced"


class SpeedBand(StrEnum):
    """Relative motion speed, as a coarse band -- mapped to a whole number of cycles per loop
    (a seamless loop with `loop_mode="cycle"` requires an integer speed)."""

    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


class PivotHint(StrEnum):
    """Where the motion anchors, relative to the object's own bbox (used by ROTATE and SWAY)."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


class ObjectDescriptionResponse(BaseModel):
    """The VLM's full structured read on one candidate region.

    Field semantics (each maps to a required task-brief question):

    - `bbox_assessment` -- is the box a good single-object candidate (task: bbox validation).
    - `object_identity` -- what object is actually inside the box (task Q1).
    - `matches_semantic_label` -- does that object match the plan's `semantic_label` (task Q1).
    - `animatable` -- is the object potentially animatable at all (task Q3).
    - `movable_parts` / `static_parts` -- what may move / must stay fixed (tasks Q4/Q5).
    - `motion_kind` -- which motion category fits (task Q6); required iff `animatable`.
    - `direction` -- direction for DRIFT (task Q7); required iff `motion_kind == drift`.
    - `amplitude_band` / `speed_band` -- relative amplitude and character/speed (tasks Q8/Q9).
    - `pivot_hint` -- anchoring point for ROTATE/SWAY.
    - `constraints` -- rules that must not be violated while animating (task Q10).
    - `neighbor_conflicts` -- problems with neighboring objects/background/occlusion (task Q11).
    - `confidence` -- confidence in the whole read (task Q12).
    - `reason` -- one-sentence justification.
    """

    bbox_assessment: BBoxAssessment
    object_identity: str = Field(min_length=1)
    matches_semantic_label: bool
    animatable: bool
    movable_parts: list[str] = Field(default_factory=list)
    static_parts: list[str] = Field(default_factory=list)
    motion_kind: MotionKind | None = None
    direction: DirectionWord | None = None
    amplitude_band: AmplitudeBand = AmplitudeBand.MODERATE
    speed_band: SpeedBand = SpeedBand.NORMAL
    pivot_hint: PivotHint = PivotHint.CENTER
    constraints: list[str] = Field(default_factory=list)
    neighbor_conflicts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_motion_when_animatable(self) -> ObjectDescriptionResponse:
        if self.animatable and self.motion_kind is None:
            raise ValueError("animatable=true requires a motion_kind")
        if not self.animatable and self.motion_kind is not None:
            raise ValueError("animatable=false must not carry a motion_kind")
        if self.motion_kind == MotionKind.DRIFT and self.direction is None:
            raise ValueError("motion_kind=drift requires a direction")
        if self.motion_kind != MotionKind.DRIFT and self.direction is not None:
            raise ValueError("direction is only meaningful for motion_kind=drift")
        return self


# The motion categories the VLM is offered, with the plain-language description of each.
MOTION_KIND_DESCRIPTIONS: dict[MotionKind, str] = {
    MotionKind.SWAY: "gentle back-and-forth sway around a point (hair, cloth, flags, banners)",
    MotionKind.FLOW: "continuous flowing deformation (smoke, water, energy wisps, speed lines)",
    MotionKind.DRIFT: (
        "steady movement in one direction (rain, drifting particles, a drifting object)"
    ),
    MotionKind.ROTATE: "rotation around a pivot point (a raised weapon, an arm swing)",
    MotionKind.PULSE: (
        "radial pulse outward/inward from the object's own center (impact bursts, glow, "
        "energy fields)"
    ),
    MotionKind.BREATHE: "slow uniform scale breathing (an energy aura)",
    MotionKind.FLICKER: "rapid intensity variation (a blink, sparks)",
}

# The transform kinds whose pivot matters (kept explicit so `mapping.py` and the prompt text
# share the same list).
PIVOT_RELEVANT_KINDS: frozenset[TransformKind] = frozenset(
    {TransformKind.ROTATE, TransformKind.MESH_WARP}
)
