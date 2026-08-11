from __future__ import annotations

import numpy as np
import pytest

from manga_animation.animation.curves import sample_motion_value
from manga_animation.animation.transforms import (
    bbox_of_mask,
    generate_transformed_layer,
    resolve_pivot_px,
)
from manga_animation.pipeline.types import BBoxPx
from manga_animation.schemas.animation_plan import (
    MotionSpec,
    PivotSpec,
    TimingSpec,
    TransformKind,
    Vector2,
)

PAGE_SHAPE = (60, 80)  # (h, w)


def make_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    h, w = PAGE_SHAPE
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    image[20:40, 30:50] = (200, 100, 50)  # the "object" region, distinct color
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    return image, mask


def make_motion(
    *,
    transform_kind: TransformKind = TransformKind.TRANSLATE,
    direction: Vector2 | None = None,
    amplitude: float = 0.1,
    speed: float = 1.0,
    phase: float = 0.0,
    pivot: PivotSpec | None = None,
    timing: TimingSpec | None = None,
) -> MotionSpec:
    if direction is None and transform_kind in (TransformKind.TRANSLATE, TransformKind.SHEAR):
        direction = Vector2(x=1.0, y=0.0)
    return MotionSpec(
        transform_kind=transform_kind,
        direction=direction,
        amplitude=amplitude,
        speed=speed,
        phase=phase,
        pivot=pivot or PivotSpec(),
        timing=timing or TimingSpec(),
    )


PANEL_BBOX = BBoxPx(x0=0, y0=0, x1=PAGE_SHAPE[1], y1=PAGE_SHAPE[0])


# --- determinism -------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,direction",
    [
        (TransformKind.TRANSLATE, Vector2(x=1.0, y=0.0)),
        (TransformKind.ROTATE, None),
        (TransformKind.SCALE, None),
        (TransformKind.SHEAR, Vector2(x=0.0, y=1.0)),
        (TransformKind.MESH_WARP, None),
        (TransformKind.OPACITY, None),
    ],
)
def test_generate_transformed_layer_is_deterministic(kind, direction):
    image, mask = make_image_and_mask()
    motion = make_motion(transform_kind=kind, direction=direction, amplitude=0.15)

    layer_a, mask_a = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.37, loop_duration_s=4.0
    )
    layer_b, mask_b = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.37, loop_duration_s=4.0
    )

    np.testing.assert_array_equal(layer_a, layer_b)
    np.testing.assert_array_equal(mask_a, mask_b)


# --- transform correctness ----------------------------------------------------


def test_translate_moves_mask_in_direction():
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.TRANSLATE,
        direction=Vector2(x=1.0, y=0.0),
        amplitude=0.3,
        speed=1.0,
        phase=0.25,  # sin(2*pi*0.25) = 1.0 -> peak positive displacement
    )
    _, warped_mask = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    orig_bbox = bbox_of_mask(mask)
    warped_bbox = bbox_of_mask(warped_mask)
    assert warped_bbox.x0 > orig_bbox.x0
    assert warped_bbox.y0 == orig_bbox.y0  # pure x-direction translate: y unchanged


def test_rotate_about_object_bbox_pivot_changes_shape_but_not_center():
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        direction=None,
        amplitude=45.0,
        speed=1.0,
        phase=0.25,
    )
    _, warped_mask = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    assert np.any(warped_mask > 0)
    orig_bbox = bbox_of_mask(mask)
    warped_bbox = bbox_of_mask(warped_mask)
    # A 45-degree rotation of a square about its own center should visibly grow its bbox.
    assert (warped_bbox.width * warped_bbox.height) > (orig_bbox.width * orig_bbox.height)


def test_opacity_scales_mask_without_moving_it():
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.OPACITY,
        direction=None,
        amplitude=0.5,
        speed=1.0,
        phase=0.75,  # sin(2*pi*0.75) = -1.0 -> minimum alpha_scale
    )
    layer, warped_mask = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    np.testing.assert_array_equal(layer, image)  # opacity never moves pixels, only alpha
    assert bbox_of_mask(warped_mask) == bbox_of_mask(mask)  # same footprint
    assert warped_mask[25, 35] < mask[25, 35]  # dimmed


# --- periodic trajectory / seamless loop --------------------------------------


def test_cycle_with_integer_speed_returns_to_start_value():
    motion = make_motion(transform_kind=TransformKind.ROTATE, speed=2.0, phase=0.1)
    start = sample_motion_value(motion, t_s=0.0, loop_duration_s=4.0)
    end = sample_motion_value(motion, t_s=4.0, loop_duration_s=4.0)
    assert start == pytest.approx(end, abs=1e-9)


def test_cycle_frame_pixels_match_at_loop_boundary():
    image, mask = make_image_and_mask()
    motion = make_motion(transform_kind=TransformKind.ROTATE, amplitude=20.0, speed=2.0)

    frame_0, mask_0 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    frame_1, mask_1 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=1.0, loop_duration_s=4.0
    )
    np.testing.assert_array_equal(frame_0, frame_1)
    np.testing.assert_array_equal(mask_0, mask_1)


# --- loop_mode distinctness ----------------------------------------------------


def test_cycle_once_hold_ping_pong_produce_distinct_values_mid_window():
    base = dict(transform_kind=TransformKind.ROTATE, amplitude=10.0, speed=1.0, phase=0.0)

    cycle_motion = make_motion(**base, timing=TimingSpec(loop_mode="cycle"))
    hold_motion = make_motion(**base, timing=TimingSpec(loop_mode="once_hold"))
    ping_motion = make_motion(**base, timing=TimingSpec(loop_mode="ping_pong"))

    t_s, duration = 2.0, 4.0  # halfway through the window
    cycle_value = sample_motion_value(cycle_motion, t_s, duration)
    hold_value = sample_motion_value(hold_motion, t_s, duration)
    ping_value = sample_motion_value(ping_motion, t_s, duration)

    assert cycle_value == pytest.approx(0.0, abs=1e-9)  # sin(pi) == 0
    assert hold_value == pytest.approx(0.5, abs=1e-9)  # linear easing halfway through 0->1
    assert ping_value == pytest.approx(1.0, abs=1e-9)  # triangle peak at the window midpoint

    assert len({round(cycle_value, 6), round(hold_value, 6), round(ping_value, 6)}) == 3


def test_once_hold_freezes_after_window_closes():
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        amplitude=10.0,
        timing=TimingSpec(duration_s=2.0, loop_mode="once_hold"),
    )
    at_end = sample_motion_value(motion, t_s=2.0, loop_duration_s=4.0)
    after_end = sample_motion_value(motion, t_s=3.5, loop_duration_s=4.0)
    assert at_end == pytest.approx(1.0)
    assert after_end == pytest.approx(1.0)


def test_ping_pong_returns_to_rest_after_window_closes():
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        amplitude=10.0,
        timing=TimingSpec(duration_s=2.0, loop_mode="ping_pong"),
    )
    after_end = sample_motion_value(motion, t_s=3.5, loop_duration_s=4.0)
    assert after_end == pytest.approx(0.0, abs=1e-9)


def test_before_delay_object_is_at_rest():
    motion = make_motion(
        transform_kind=TransformKind.ROTATE, amplitude=10.0, timing=TimingSpec(delay_s=1.0)
    )
    assert sample_motion_value(motion, t_s=0.0, loop_duration_s=4.0) == 0.0
    assert sample_motion_value(motion, t_s=0.999, loop_duration_s=4.0) == 0.0


# --- pivot resolution -----------------------------------------------------------


def test_resolve_pivot_object_bbox_reference():
    object_bbox = BBoxPx(x0=10, y0=20, x1=30, y1=60)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=100, y1=100)
    pivot = resolve_pivot_px(
        PivotSpec(x=0.5, y=0.0, reference="object_bbox"), object_bbox, panel_bbox, (200, 200)
    )
    assert pivot == pytest.approx((20.0, 20.0))  # top-center of the object bbox


def test_resolve_pivot_panel_reference():
    object_bbox = BBoxPx(x0=10, y0=20, x1=30, y1=60)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=100, y1=50)
    pivot = resolve_pivot_px(
        PivotSpec(x=0.5, y=0.5, reference="panel"), object_bbox, panel_bbox, (200, 200)
    )
    assert pivot == pytest.approx((50.0, 25.0))  # panel center, independent of the object bbox


def test_resolve_pivot_page_reference():
    object_bbox = BBoxPx(x0=10, y0=20, x1=30, y1=60)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=100, y1=50)
    pivot = resolve_pivot_px(
        PivotSpec(x=0.0, y=1.0, reference="page"), object_bbox, panel_bbox, (200, 400)
    )
    assert pivot == pytest.approx((0.0, 200.0))  # bottom-left of the full page (h=200, w=400)


def test_bbox_of_mask_rejects_empty_mask():
    empty_mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        bbox_of_mask(empty_mask)
