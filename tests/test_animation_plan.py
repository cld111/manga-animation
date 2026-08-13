from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from manga_animation.schemas.animation_plan import (
    AnimationPlan,
    Easing,
    LoopSpec,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PanelPlan,
    PivotSpec,
    SourceImage,
    TransformKind,
    Vector2,
)


def _translate_motion(**overrides) -> MotionSpec:
    defaults = dict(
        transform_kind=TransformKind.TRANSLATE, direction=Vector2(x=1.0, y=0.0), amplitude=0.05
    )
    defaults.update(overrides)
    return MotionSpec(**defaults)


class TestObjectPlanMotionPresence:
    def test_static_object_needs_no_motion(self):
        obj = ObjectPlan(
            object_id="bg",
            panel_id="p1",
            semantic_label="background",
            confidence=0.9,
            motion_type=MotionType.STATIC,
        )
        assert obj.motion is None

    def test_static_object_with_motion_is_rejected(self):
        with pytest.raises(ValidationError, match="STATIC"):
            ObjectPlan(
                object_id="bg",
                panel_id="p1",
                semantic_label="background",
                confidence=0.9,
                motion_type=MotionType.STATIC,
                motion=_translate_motion(),
            )

    def test_non_static_object_requires_motion(self):
        with pytest.raises(ValidationError, match="no motion spec"):
            ObjectPlan(
                object_id="flag",
                panel_id="p1",
                semantic_label="flag",
                confidence=0.8,
                motion_type=MotionType.PRIMARY,
            )

    def test_object_cannot_be_its_own_parent(self):
        with pytest.raises(ValidationError, match="own parent"):
            ObjectPlan(
                object_id="x",
                panel_id="p1",
                semantic_label="x",
                confidence=0.5,
                motion_type=MotionType.STATIC,
                parent_id="x",
            )


class TestMotionSpecDirection:
    def test_translate_without_direction_is_rejected(self):
        with pytest.raises(ValidationError, match="requires a direction vector"):
            MotionSpec(transform_kind=TransformKind.TRANSLATE, amplitude=0.1)

    def test_translate_direction_is_normalized_to_unit_length(self):
        motion = MotionSpec(
            transform_kind=TransformKind.TRANSLATE, direction=Vector2(x=3.0, y=4.0), amplitude=0.1
        )
        assert motion.direction is not None
        assert motion.direction.magnitude() == pytest.approx(1.0)
        assert motion.direction.x == pytest.approx(0.6)
        assert motion.direction.y == pytest.approx(0.8)

    def test_rotate_does_not_require_direction(self):
        motion = MotionSpec(transform_kind=TransformKind.ROTATE, amplitude=5.0)
        assert motion.direction is None

    def test_zero_amplitude_is_rejected(self):
        with pytest.raises(ValidationError):
            MotionSpec(transform_kind=TransformKind.ROTATE, amplitude=0.0)

    def test_zero_direction_vector_is_rejected(self):
        with pytest.raises(ValidationError, match="non-zero"):
            MotionSpec(
                transform_kind=TransformKind.TRANSLATE,
                direction=Vector2(x=0.0, y=0.0),
                amplitude=0.1,
            )


class TestAnimationPlanHierarchy:
    def test_minimal_valid_plan(self, base_plan_kwargs):
        plan = AnimationPlan(**base_plan_kwargs)
        assert plan.objects == []
        assert plan.schema_version == "1.0"

    def test_duplicate_object_id_is_rejected(self, base_plan_kwargs):
        obj_kwargs = dict(
            panel_id="panel_1", semantic_label="x", confidence=0.5, motion_type=MotionType.STATIC
        )
        with pytest.raises(ValidationError, match="duplicate object_id"):
            AnimationPlan(
                **base_plan_kwargs,
                objects=[
                    ObjectPlan(object_id="a", **obj_kwargs),
                    ObjectPlan(object_id="a", **obj_kwargs),
                ],
            )

    def test_object_referencing_unknown_panel_is_rejected(self, base_plan_kwargs):
        with pytest.raises(ValidationError, match="unknown panel_id"):
            AnimationPlan(
                **base_plan_kwargs,
                objects=[
                    ObjectPlan(
                        object_id="a",
                        panel_id="does_not_exist",
                        semantic_label="x",
                        confidence=0.5,
                        motion_type=MotionType.STATIC,
                    )
                ],
            )

    def test_object_referencing_unknown_parent_is_rejected(self, base_plan_kwargs):
        with pytest.raises(ValidationError, match="unknown parent_id"):
            AnimationPlan(
                **base_plan_kwargs,
                objects=[
                    ObjectPlan(
                        object_id="a",
                        panel_id="panel_1",
                        semantic_label="x",
                        confidence=0.5,
                        motion_type=MotionType.STATIC,
                        parent_id="ghost",
                    )
                ],
            )

    def test_valid_parent_child_relationship(self, base_plan_kwargs):
        head = ObjectPlan(
            object_id="head",
            panel_id="panel_1",
            semantic_label="head",
            confidence=0.9,
            motion_type=MotionType.STATIC,
            children_ids=["hair"],
        )
        hair = ObjectPlan(
            object_id="hair",
            panel_id="panel_1",
            semantic_label="hair",
            confidence=0.8,
            motion_type=MotionType.SECONDARY,
            parent_id="head",
            motion=_translate_motion(),
        )
        plan = AnimationPlan(**base_plan_kwargs, objects=[head, hair])
        assert plan.children_of("head") == ["hair"]

    def test_inconsistent_children_ids_vs_parent_id_is_rejected(self, base_plan_kwargs):
        head = ObjectPlan(
            object_id="head",
            panel_id="panel_1",
            semantic_label="head",
            confidence=0.9,
            motion_type=MotionType.STATIC,
            children_ids=["hair"],
        )
        # hair claims a *different* parent than head's declared child link
        hair = ObjectPlan(
            object_id="hair",
            panel_id="panel_1",
            semantic_label="hair",
            confidence=0.8,
            motion_type=MotionType.SECONDARY,
            parent_id=None,
            motion=_translate_motion(),
        )
        with pytest.raises(ValidationError, match="inconsistent hierarchy"):
            AnimationPlan(**base_plan_kwargs, objects=[head, hair])

    def test_parent_cycle_is_rejected(self, base_plan_kwargs):
        a = ObjectPlan(
            object_id="a",
            panel_id="panel_1",
            semantic_label="a",
            confidence=0.5,
            motion_type=MotionType.STATIC,
            parent_id="b",
        )
        b = ObjectPlan(
            object_id="b",
            panel_id="panel_1",
            semantic_label="b",
            confidence=0.5,
            motion_type=MotionType.STATIC,
            parent_id="a",
        )
        with pytest.raises(ValidationError, match="cycle"):
            AnimationPlan(**base_plan_kwargs, objects=[a, b])


class TestSeamlessLoopTiming:
    def _plan_with_speed(
        self, base_plan_kwargs, speed: float, *, seamless: bool = True, loop_mode: str = "cycle"
    ):
        obj = ObjectPlan(
            object_id="flag",
            panel_id="panel_1",
            semantic_label="flag",
            confidence=0.9,
            motion_type=MotionType.PRIMARY,
            motion=_translate_motion(speed=speed, timing={"loop_mode": loop_mode}),
        )
        return AnimationPlan(**base_plan_kwargs, objects=[obj], loop=LoopSpec(seamless=seamless))

    def test_integer_speed_with_cycle_loop_mode_is_accepted(self, base_plan_kwargs):
        plan = self._plan_with_speed(base_plan_kwargs, speed=2.0)
        assert plan.objects[0].motion.speed == 2.0

    def test_non_integer_speed_with_seamless_cycle_is_rejected(self, base_plan_kwargs):
        with pytest.raises(ValidationError, match="whole number of cycles"):
            self._plan_with_speed(base_plan_kwargs, speed=1.5)

    def test_non_integer_speed_is_allowed_when_not_seamless(self, base_plan_kwargs):
        plan = self._plan_with_speed(base_plan_kwargs, speed=1.5, seamless=False)
        assert plan.objects[0].motion.speed == 1.5

    def test_non_integer_speed_is_allowed_with_ping_pong(self, base_plan_kwargs):
        plan = self._plan_with_speed(base_plan_kwargs, speed=1.5, loop_mode="ping_pong")
        assert plan.objects[0].motion.speed == 1.5

    def test_non_integer_speed_seamless_cycle_error_does_not_suggest_once_hold(
        self, base_plan_kwargs
    ):
        # once_hold cannot actually satisfy the seamless-loop promise (it holds its end
        # state rather than returning to rest), so the error guiding users away from a
        # non-integer cycle speed must not present it as an equivalent fix to ping_pong.
        with pytest.raises(ValidationError) as exc_info:
            self._plan_with_speed(base_plan_kwargs, speed=1.5)
        message = str(exc_info.value)
        assert "once_hold" not in message
        assert "ping_pong" in message

    def test_once_hold_with_seamless_loop_is_rejected(self, base_plan_kwargs):
        # once_hold sweeps to its end state and holds there forever; a fresh loop
        # iteration restarts the object at rest, so this combination always produces a
        # discontinuity at the loop boundary, regardless of speed.
        with pytest.raises(ValidationError, match="always breaks the seamless-loop boundary"):
            self._plan_with_speed(base_plan_kwargs, speed=1.0, loop_mode="once_hold")

    def test_once_hold_with_non_integer_speed_and_seamless_loop_is_rejected(self, base_plan_kwargs):
        # The once_hold rejection is independent of the (cycle-only) integer-speed rule —
        # it must trigger even when speed alone would have been fine.
        with pytest.raises(ValidationError, match="always breaks the seamless-loop boundary"):
            self._plan_with_speed(base_plan_kwargs, speed=1.5, loop_mode="once_hold")

    def test_once_hold_is_allowed_when_not_seamless(self, base_plan_kwargs):
        plan = self._plan_with_speed(
            base_plan_kwargs, speed=1.0, seamless=False, loop_mode="once_hold"
        )
        assert plan.objects[0].motion.timing.loop_mode == "once_hold"
        assert plan.loop.seamless is False

    def test_motion_window_exceeding_loop_duration_is_rejected(self, base_plan_kwargs):
        obj = ObjectPlan(
            object_id="flag",
            panel_id="panel_1",
            semantic_label="flag",
            confidence=0.9,
            motion_type=MotionType.PRIMARY,
            motion=_translate_motion(timing={"delay_s": 3.0, "duration_s": 3.0}),
        )
        with pytest.raises(ValidationError, match="exceeds loop duration"):
            AnimationPlan(**base_plan_kwargs, objects=[obj], loop=LoopSpec(duration_s=4.0))


class TestSerialization:
    def test_round_trip_via_json_string(self, base_plan_kwargs):
        obj = ObjectPlan(
            object_id="flag",
            panel_id="panel_1",
            semantic_label="flag",
            confidence=0.9,
            motion_type=MotionType.PRIMARY,
            motion=_translate_motion(easing=Easing.SINE, pivot=PivotSpec(x=0.5, y=0.0)),
        )
        plan = AnimationPlan(**base_plan_kwargs, objects=[obj])

        raw = plan.model_dump_json()
        restored = AnimationPlan.model_validate_json(raw)

        assert restored == plan
        assert json.loads(raw)["objects"][0]["motion"]["easing"] == "sine"

    def test_round_trip_via_file(self, base_plan_kwargs, tmp_path):
        plan = AnimationPlan(**base_plan_kwargs)
        path = tmp_path / "plan.json"

        plan.to_json_file(path)
        restored = AnimationPlan.from_json_file(path)

        assert restored == plan
        assert path.read_text().endswith("\n")


def test_bbox_out_of_bounds_is_rejected():
    with pytest.raises(ValidationError, match="right edge"):
        PanelPlan(panel_id="p1", bbox={"x": 0.6, "y": 0.0, "width": 0.6, "height": 1.0})


def test_loop_frame_count_rounds_duration_times_fps():
    loop = LoopSpec(duration_s=4.0, fps=24)
    assert loop.frame_count == 96


def test_source_image_requires_positive_dimensions():
    with pytest.raises(ValidationError):
        SourceImage(path="x.png", width=0, height=100)
