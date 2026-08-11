"""The Animation Plan: canonical internal representation of animation decisions.

Design rationale lives in docs/animation-plan-schema.md. In short: this schema is the
contract between the semantic stages (VLM/analysis, which decide *what* should move and
*why*) and the mechanical stages (grounding/segmentation/cv/rendering, which decide *how*
pixels move). It intentionally excludes pixel-space data (bounding boxes on objects, masks)
because those are produced by the grounding/segmentation stages *from* this plan, not
before it — see the pipeline order in docs/pipeline.md.

All spatial fields are resolution-independent (normalized to [0, 1]) so the same plan is
valid whether the source page was analyzed at a debug preview resolution locally or at full
resolution on a remote GPU worker.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_EPS = 1e-6


class MotionType(StrEnum):
    """How central this object's motion is to expressing the page's action.

    Ordering encodes intent, not priority: STATIC is the default and preferred outcome
    (see the "Static Is a Valid Result" principle in docs/architecture.md). PRIMARY drives
    the readable action; SECONDARY and MICRO are motion that follows from a PRIMARY mover
    (cloth, hair) or adds subtle life (a blink, a sway) without carrying meaning on its own.
    """

    STATIC = "static"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    MICRO = "micro"


class TransformKind(StrEnum):
    """The mechanical transform a downstream CV stage should apply."""

    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"
    SHEAR = "shear"
    MESH_WARP = "mesh_warp"
    OPACITY = "opacity"


class Easing(StrEnum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SINE = "sine"


class Vector2(BaseModel):
    """A 2D vector in image coordinates (x: right-positive, y: down-positive)."""

    x: float
    y: float

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)


class BBox(BaseModel):
    """A normalized ([0, 1]) axis-aligned box, relative to the source page."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> BBox:
        if self.x + self.width > 1.0 + _EPS:
            raise ValueError(
                f"bbox extends past the right edge: x={self.x} + width={self.width} > 1"
            )
        if self.y + self.height > 1.0 + _EPS:
            raise ValueError(
                f"bbox extends past the bottom edge: y={self.y} + height={self.height} > 1"
            )
        return self


class PivotSpec(BaseModel):
    """The point a rotation/scale/shear is applied around.

    Coordinates are normalized to the `reference` frame, e.g. (0.5, 0.0) on an
    `object_bbox` reference means "top-center of the object's box" — a natural anchor for
    hair swaying from the scalp or a flag waving from its pole.
    """

    x: float = Field(ge=0.0, le=1.0, default=0.5)
    y: float = Field(ge=0.0, le=1.0, default=0.5)
    reference: Literal["object_bbox", "panel", "page"] = "object_bbox"


class TimingSpec(BaseModel):
    """When, within the loop, an object's motion is active."""

    delay_s: float = Field(ge=0.0, default=0.0)
    duration_s: float | None = Field(
        default=None, gt=0.0, description="None = spans the rest of the loop"
    )
    loop_mode: Literal["cycle", "once_hold", "ping_pong"] = "cycle"


class MotionSpec(BaseModel):
    """The kinematic parameters of a non-STATIC object's motion."""

    transform_kind: TransformKind
    direction: Vector2 | None = Field(
        default=None,
        description="Unit vector; required for translate/shear, optional/unused otherwise.",
    )
    amplitude: float = Field(
        gt=0.0,
        description=(
            "Magnitude of motion, meaning depends on transform_kind: translate = fraction "
            "of the panel diagonal; rotate = degrees; scale/opacity = fractional delta; "
            "mesh_warp = normalized warp strength."
        ),
    )
    phase: float = Field(
        ge=0.0, lt=1.0, default=0.0, description="Cycle offset at t=0, as a fraction of one cycle."
    )
    speed: float = Field(gt=0.0, default=1.0, description="Cycles completed per loop duration.")
    easing: Easing = Easing.EASE_IN_OUT
    pivot: PivotSpec = Field(default_factory=PivotSpec)
    timing: TimingSpec = Field(default_factory=TimingSpec)

    @model_validator(mode="after")
    def _validate_direction(self) -> MotionSpec:
        needs_direction = self.transform_kind in (TransformKind.TRANSLATE, TransformKind.SHEAR)
        if needs_direction:
            if self.direction is None:
                raise ValueError(
                    f"transform_kind={self.transform_kind.value} requires a direction vector"
                )
            mag = self.direction.magnitude()
            if mag < _EPS:
                raise ValueError("direction vector must be non-zero")
            if abs(mag - 1.0) > _EPS:
                self.direction = Vector2(x=self.direction.x / mag, y=self.direction.y / mag)
        return self


class ObjectPlan(BaseModel):
    """One object's animation decision: whether, how, and why it should move."""

    object_id: str = Field(min_length=1)
    panel_id: str = Field(min_length=1, description="Which PanelPlan this object belongs to.")
    semantic_label: str = Field(
        min_length=1, description='e.g. "character_hair", "flag", "left_hand".'
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the STATIC/ANIMATED decision and params."
    )
    motion_type: MotionType
    parent_id: str | None = Field(
        default=None, description="object_id of the kinematic parent, if any."
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional explicit list of child object_ids; cross-checked against their parent_id."
        ),
    )
    motion: MotionSpec | None = None

    @model_validator(mode="after")
    def _validate_motion_presence(self) -> ObjectPlan:
        if self.motion_type == MotionType.STATIC and self.motion is not None:
            raise ValueError(
                f"object '{self.object_id}' is STATIC but defines a motion spec — "
                "prefer STATIC with no motion over unjustified motion"
            )
        if self.motion_type != MotionType.STATIC and self.motion is None:
            raise ValueError(
                f"object '{self.object_id}' has motion_type={self.motion_type.value} "
                "but no motion spec"
            )
        return self

    @model_validator(mode="after")
    def _validate_self_parent(self) -> ObjectPlan:
        if self.parent_id is not None and self.parent_id == self.object_id:
            raise ValueError(f"object '{self.object_id}' cannot be its own parent")
        if self.object_id in self.children_ids:
            raise ValueError(f"object '{self.object_id}' cannot be its own child")
        return self


class PanelPlan(BaseModel):
    """A panel (or, for a splash page, the whole page) as a distinct scene."""

    panel_id: str = Field(min_length=1)
    bbox: BBox
    description: str | None = Field(
        default=None, description="Brief scene description from the analysis stage."
    )


class SourceImage(BaseModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    checksum: str | None = None


class LoopSpec(BaseModel):
    """Global timing for the output loop."""

    duration_s: float = Field(gt=0.0, le=30.0, default=4.0)
    fps: int = Field(gt=0, le=60, default=24)
    seamless: bool = True
    crossfade_frames: int = Field(ge=0, default=0)

    @property
    def frame_count(self) -> int:
        return round(self.duration_s * self.fps)


class AnimationPlan(BaseModel):
    """The full, machine-readable animation decision for one source image."""

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: Literal["1.0"] = "1.0"
    source: SourceImage
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    panels: list[PanelPlan] = Field(min_length=1)
    objects: list[ObjectPlan] = Field(default_factory=list)
    loop: LoopSpec = Field(default_factory=LoopSpec)

    @model_validator(mode="after")
    def _validate_plan(self) -> AnimationPlan:
        panel_ids = [p.panel_id for p in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            dupes = sorted({p for p in panel_ids if panel_ids.count(p) > 1})
            raise ValueError(f"duplicate panel_id(s): {dupes}")
        panel_id_set = set(panel_ids)

        object_ids = [o.object_id for o in self.objects]
        if len(object_ids) != len(set(object_ids)):
            dupes = sorted({o for o in object_ids if object_ids.count(o) > 1})
            raise ValueError(f"duplicate object_id(s): {dupes}")
        by_id = {o.object_id: o for o in self.objects}

        for obj in self.objects:
            if obj.panel_id not in panel_id_set:
                raise ValueError(
                    f"object '{obj.object_id}' references unknown panel_id '{obj.panel_id}'"
                )
            if obj.parent_id is not None and obj.parent_id not in by_id:
                raise ValueError(
                    f"object '{obj.object_id}' references unknown parent_id '{obj.parent_id}'"
                )
            for child_id in obj.children_ids:
                if child_id not in by_id:
                    raise ValueError(
                        f"object '{obj.object_id}' references unknown child_id '{child_id}'"
                    )
                child = by_id[child_id]
                if child.parent_id != obj.object_id:
                    raise ValueError(
                        f"inconsistent hierarchy: '{obj.object_id}' lists '{child_id}' as a child, "
                        f"but '{child_id}'.parent_id is '{child.parent_id}' — both sides of a "
                        "parent/child link must agree"
                    )

        for obj in self.objects:
            seen = {obj.object_id}
            current = obj.parent_id
            while current is not None:
                if current in seen:
                    raise ValueError(
                        f"cycle detected in parent hierarchy involving '{obj.object_id}'"
                    )
                seen.add(current)
                current = by_id[current].parent_id

        for obj in self.objects:
            if obj.motion is None:
                continue
            timing = obj.motion.timing
            if timing.duration_s is not None:
                span_end = timing.delay_s + timing.duration_s
                if span_end > self.loop.duration_s + _EPS:
                    raise ValueError(
                        f"object '{obj.object_id}' motion window (delay={timing.delay_s}s + "
                        f"duration={timing.duration_s}s = {span_end}s) exceeds loop duration "
                        f"{self.loop.duration_s}s"
                    )
            elif timing.delay_s > self.loop.duration_s + _EPS:
                raise ValueError(
                    f"object '{obj.object_id}' delay {timing.delay_s}s exceeds loop duration "
                    f"{self.loop.duration_s}s"
                )

            if (
                self.loop.seamless
                and timing.loop_mode == "cycle"
                and abs(obj.motion.speed - round(obj.motion.speed)) > 1e-6
            ):
                raise ValueError(
                    f"object '{obj.object_id}' has non-integer speed={obj.motion.speed} with "
                    "loop_mode='cycle' under a seamless loop; a cyclic motion must complete a "
                    "whole number of cycles to return to its start state (use an integer speed, "
                    "switch loop_mode to 'once_hold'/'ping_pong', or set loop.seamless=False)"
                )
        return self

    def children_of(self, object_id: str) -> list[str]:
        """Children derived from parent_id links (the source of truth for hierarchy)."""
        return [o.object_id for o in self.objects if o.parent_id == object_id]

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> AnimationPlan:
        return cls.model_validate_json(Path(path).read_text())
