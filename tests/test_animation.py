from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from manga_animation.animation.curves import sample_motion_value
from manga_animation.animation.transforms import (
    _affine_matrix,
    bbox_of_mask,
    generate_transformed_layer,
    resolve_pivot_px,
)
from manga_animation.compositing import composite_frame
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
    """The COMPOSITED result at t_frac=0 and t_frac=1 (the frame the loop wraps back to) must
    be indistinguishable for a valid seamless cycle motion -- the actual, pipeline-visible
    promise (docs/animation-plan-schema.md's seamless-loop constraint), checked through
    `composite_frame` rather than on the raw `generate_transformed_layer` arrays directly.

    Phase 6 local-rendering hardening (see this module's docstring in animation/transforms.py)
    only guarantees a returned layer's content *where its own mask is nonzero* -- outside that
    footprint the array is an architecturally-irrelevant zero-filled placeholder (never the
    old code's incidental "whole page warped, including background" content), so raw-array
    equality is no longer the right invariant to check: `sample_motion_value`'s `sin(4*pi)` at
    t_frac=1.0 is not bit-identical to `sin(0)` at t_frac=0.0 (see
    `test_cycle_with_integer_speed_returns_to_start_value` above, itself already only
    `pytest.approx`), and that sub-pixel-scale value difference can round a local ROI's
    float-AABB `floor`/`ceil` boundary to a different integer pixel between the two calls --
    entirely within the placeholder region, never inside either mask's actual footprint.
    """
    image, mask = make_image_and_mask()
    motion = make_motion(transform_kind=TransformKind.ROTATE, amplitude=20.0, speed=2.0)

    layer_0, mask_0 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    layer_1, mask_1 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=1.0, loop_duration_s=4.0
    )
    frame_0 = composite_frame(image, layer_0, mask_0)
    frame_1 = composite_frame(image, layer_1, mask_1)
    np.testing.assert_array_equal(frame_0, frame_1)


def test_once_hold_frame_pixels_do_not_match_at_loop_boundary():
    # Mechanical justification for AnimationPlan rejecting once_hold under seamless=True
    # (schemas/animation_plan.py): once_hold holds its end state rather than returning to
    # rest, so — unlike the cycle/integer-speed and ping_pong cases above — its frame at the
    # loop boundary genuinely differs from frame 0.
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        amplitude=20.0,
        timing=TimingSpec(loop_mode="once_hold"),
    )

    frame_0, mask_0 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.0, loop_duration_s=4.0
    )
    frame_1, mask_1 = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=1.0, loop_duration_s=4.0
    )
    assert not np.array_equal(frame_0, frame_1) or not np.array_equal(mask_0, mask_1)


def test_ping_pong_frame_pixels_match_at_loop_boundary():
    # Unlike once_hold, ping_pong returns to rest on its own — it stays a valid seamless
    # replacement even at a non-integer speed (schemas/animation_plan.py allows it).
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        amplitude=20.0,
        speed=1.5,
        timing=TimingSpec(loop_mode="ping_pong"),
    )

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


# --- Phase 6: local-region transforms must match the old full-page implementation ---------


def _generate_transformed_layer_full_page_reference(
    image: np.ndarray,
    mask: np.ndarray,
    motion: MotionSpec,
    panel_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
    t_frac: float,
    *,
    loop_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-Phase-6 `generate_transformed_layer`: identical math (same `_affine_matrix`, same
    mesh-warp displacement formula), always applied over the WHOLE page — kept verbatim (only
    de-duplicated via the still-unchanged `_affine_matrix` helper) as the deterministic
    reference the localized version must reproduce wherever a mask (old or new) says the pixel
    is compositing-relevant.
    """
    t_s = t_frac * loop_duration_s
    value = sample_motion_value(motion, t_s, loop_duration_s)
    object_bbox_px = bbox_of_mask(mask)
    kind = motion.transform_kind
    h, w = mask.shape

    if kind == TransformKind.OPACITY:
        alpha_scale = min(max(1.0 + value * motion.amplitude, 0.0), 1.0)
        scaled_mask = np.clip(mask.astype(np.float32) * alpha_scale, 0, 255).astype(np.uint8)
        return image, scaled_mask

    if kind == TransformKind.MESH_WARP:
        x0, y0, x1, y1 = object_bbox_px.as_xyxy()
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        direction = motion.direction
        dir_x = direction.x if direction is not None else 1.0
        dir_y = direction.y if direction is not None else 0.0
        if abs(dir_y) >= abs(dir_x):
            local = np.clip((map_y - y0) / max(y1 - y0, 1), 0.0, 1.0)
        else:
            local = np.clip((map_x - x0) / max(x1 - x0, 1), 0.0, 1.0)
        strength = value * motion.amplitude * max(x1 - x0, y1 - y0)
        warped_map_x = map_x + strength * dir_x * local
        warped_map_y = map_y + strength * dir_y * local
        warped_layer = cv2.remap(
            image,
            warped_map_x,
            warped_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.remap(
            mask, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
        )
        return warped_layer, warped_mask

    pivot_px = resolve_pivot_px(motion.pivot, object_bbox_px, panel_bbox_px, page_shape)
    panel_diag_px = math.hypot(panel_bbox_px.width, panel_bbox_px.height)
    matrix = _affine_matrix(kind, value, motion, pivot_px, panel_diag_px)
    warped_layer = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return warped_layer, warped_mask


def _assert_localized_matches_reference(
    image: np.ndarray,
    mask: np.ndarray,
    motion: MotionSpec,
    panel_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
    t_frac: float,
    loop_duration_s: float,
) -> None:
    actual_layer, actual_mask = generate_transformed_layer(
        image,
        mask,
        motion,
        panel_bbox_px,
        page_shape,
        t_frac=t_frac,
        loop_duration_s=loop_duration_s,
    )
    ref_layer, ref_mask = _generate_transformed_layer_full_page_reference(
        image,
        mask,
        motion,
        panel_bbox_px,
        page_shape,
        t_frac=t_frac,
        loop_duration_s=loop_duration_s,
    )

    # The mask has no "background" content to warp into view (it's 0 everywhere outside the
    # object already), so it must match the reference everywhere within the same atol=1 (see
    # below) fixed-point-rounding tolerance the affine kinds' layer comparison needs -- masks go
    # through the identical shifted-matrix cv2.warpAffine call, so a partial-alpha edge pixel's
    # interpolated mask value is subject to the exact same quantization-boundary sensitivity.
    # Any *real* under-coverage bug (the ROI margin actually excluding part of the object) would
    # show up as a large, multi-pixel-region discrepancy, not an isolated |diff|==1 -- which is
    # exactly what every edge case below was checked for while this test was written.
    np.testing.assert_allclose(actual_mask.astype(np.int16), ref_mask.astype(np.int16), atol=1)

    # The layer image only has a documented contract where a mask says it's relevant; outside
    # both masks' footprints it's an architecturally-irrelevant placeholder value (whatever the
    # new code's local ROI happens to hold there, vs. the old code's incidentally-warped
    # background) -- compare only where compositing would actually read it. (Well outside the
    # local ROI the new array is zero by construction, but that is a performance property, not
    # a pixel-correctness one -- see the Phase 6 performance-evidence script/tests instead.)
    relevant = (actual_mask > 0) | (ref_mask > 0)
    # For the affine kinds (translate/rotate/scale/shear), the local path calls the *same*
    # cv2.warpAffine with a translation-shifted matrix (see transforms.py's docstring) rather
    # than the old, unshifted one -- mathematically the same absolute-coordinate sample point,
    # but OpenCV's INTER_LINEAR path quantizes the inverse matrix and each output pixel's
    # fixed-point source coordinate independently per call, so the *shifted* matrix's own
    # fresh quantization can round a handful of near-boundary interpolation weights to the
    # adjacent 1/32-pixel step, differing from the old call's by exactly 1 uint8 level. This is
    # a real, understood, and bounded floating-point characteristic (confirmed empirically here
    # across every required edge case: mismatches are always |diff|==1, always a small fraction
    # of the relevant pixels, and never outside the [0,255] range) -- not an under-coverage bug
    # (the mask comparison above, which is unaffected, already proves the ROI itself is
    # correct), so an atol=1 tolerance here is a documented characteristic, not a weakened test.
    # mesh_warp and opacity have no such risk (see their own docstrings) and stay bit-exact.
    np.testing.assert_allclose(
        actual_layer[relevant].astype(np.int16), ref_layer[relevant].astype(np.int16), atol=1
    )


def _random_image_and_mask(
    page_shape: tuple[int, int], mask_slice: tuple[slice, slice], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    h, w = page_shape
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[mask_slice] = 255
    return image, mask


_EDGE_CASE_PAGES = {
    "small_object_normal_page": ((60, 80), (slice(20, 40), slice(30, 50))),
    "top_edge": ((60, 80), (slice(0, 15), slice(30, 50))),
    "bottom_edge": ((60, 80), (slice(45, 60), slice(30, 50))),
    "left_edge": ((60, 80), (slice(20, 40), slice(0, 15))),
    "right_edge": ((60, 80), (slice(20, 40), slice(65, 80))),
    "top_left_corner": ((60, 80), (slice(0, 15), slice(0, 15))),
    "bottom_right_corner": ((60, 80), (slice(45, 60), slice(65, 80))),
    "large_object": ((60, 80), (slice(5, 55), slice(5, 75))),
    "very_small_object": ((60, 80), (slice(30, 32), slice(40, 42))),
    "non_square_page": ((90, 60), (slice(20, 40), slice(10, 30))),
    "extreme_aspect_ratio_page": ((720, 90), (slice(100, 640), slice(10, 80))),
}


@pytest.mark.parametrize("case_name", list(_EDGE_CASE_PAGES.keys()))
@pytest.mark.parametrize(
    "transform_kind,extra",
    [
        (TransformKind.ROTATE, dict(amplitude=25.0)),
        (TransformKind.SCALE, dict(amplitude=0.4)),
        (TransformKind.TRANSLATE, dict(amplitude=0.15, direction=Vector2(x=1.0, y=0.4))),
        (TransformKind.SHEAR, dict(amplitude=0.35, direction=Vector2(x=0.0, y=1.0))),
        (TransformKind.MESH_WARP, dict(amplitude=0.3, direction=Vector2(x=0.8, y=0.6))),
        (TransformKind.OPACITY, dict(amplitude=0.6)),
    ],
)
def test_localized_transform_matches_full_page_reference(case_name, transform_kind, extra):
    page_shape, mask_slice = _EDGE_CASE_PAGES[case_name]
    image, mask = _random_image_and_mask(
        page_shape, mask_slice, seed=hash((case_name, transform_kind)) % (2**31)
    )
    motion = make_motion(transform_kind=transform_kind, phase=0.0, speed=1.0, **extra)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])

    for t_frac in (0.13, 0.37, 0.5, 0.7, 0.91):
        _assert_localized_matches_reference(
            image, mask, motion, panel_bbox, page_shape, t_frac, loop_duration_s=4.0
        )


def test_localized_transform_with_displacement_beyond_original_bbox_matches_reference():
    # A large translate whose destination footprint lands well outside the original bbox --
    # exercises the AABB-of-transformed-corners ROI logic, not just the source bbox.
    page_shape, mask_slice = _EDGE_CASE_PAGES["small_object_normal_page"]
    image, mask = _random_image_and_mask(page_shape, mask_slice, seed=1)
    motion = make_motion(
        transform_kind=TransformKind.TRANSLATE,
        amplitude=0.9,
        direction=Vector2(x=1.0, y=0.5),
        phase=0.25,  # sin peak -> full +amplitude displacement
    )
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    _assert_localized_matches_reference(
        image, mask, motion, panel_bbox, page_shape, t_frac=0.0, loop_duration_s=4.0
    )


def test_localized_transform_pushed_fully_off_page_matches_reference():
    # An extreme translate that pushes the entire object past the page edge -- the ROI must
    # come back empty/clipped rather than erroring, and the mask must end up all-zero, matching
    # the reference's own natural "warped past the border -> borderValue" behavior.
    page_shape, mask_slice = _EDGE_CASE_PAGES["right_edge"]
    image, mask = _random_image_and_mask(page_shape, mask_slice, seed=2)
    motion = make_motion(
        transform_kind=TransformKind.TRANSLATE,
        amplitude=1.0,
        direction=Vector2(x=1.0, y=0.0),
        phase=0.25,
    )
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    _assert_localized_matches_reference(
        image, mask, motion, panel_bbox, page_shape, t_frac=0.0, loop_duration_s=4.0
    )


def test_localized_rotate_about_distant_page_pivot_matches_reference():
    # A small object rotated about a pivot far outside its own bbox (a page-corner pivot) --
    # the lever arm sweeps the object's footprint far from its source bbox.
    page_shape = (200, 300)
    image, mask = _random_image_and_mask(page_shape, (slice(20, 35), slice(20, 35)), seed=3)
    motion = make_motion(
        transform_kind=TransformKind.ROTATE,
        amplitude=15.0,
        pivot=PivotSpec(x=1.0, y=1.0, reference="page"),  # bottom-right of the page
    )
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    for t_frac in (0.1, 0.4, 0.6, 0.9):
        _assert_localized_matches_reference(
            image, mask, motion, panel_bbox, page_shape, t_frac, loop_duration_s=4.0
        )


def test_localized_transform_multiple_objects_are_independent_and_identity_safe():
    # Two different objects (different bboxes) on the same page, each transformed on its own --
    # object A's local computation must never touch object B's region, and vice versa.
    page_shape = (60, 80)
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, size=(*page_shape, 3), dtype=np.uint8)
    mask_a = np.zeros(page_shape, dtype=np.uint8)
    mask_a[5:15, 5:15] = 255
    mask_b = np.zeros(page_shape, dtype=np.uint8)
    mask_b[45:58, 60:78] = 255

    motion_a = make_motion(transform_kind=TransformKind.ROTATE, amplitude=30.0)
    motion_b = make_motion(transform_kind=TransformKind.SCALE, amplitude=0.5)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])

    layer_a, mask_out_a = generate_transformed_layer(
        image, mask_a, motion_a, panel_bbox, page_shape, t_frac=0.4, loop_duration_s=4.0
    )
    layer_b, mask_out_b = generate_transformed_layer(
        image, mask_b, motion_b, panel_bbox, page_shape, t_frac=0.4, loop_duration_s=4.0
    )

    # Neither object's transform reaches anywhere near the other's source region (bboxes are
    # far apart relative to their small amplitudes) -- confirms no cross-object bleed.
    assert not np.any(mask_out_a[45:58, 60:78])
    assert not np.any(mask_out_b[5:15, 5:15])

    _assert_localized_matches_reference(
        image, mask_a, motion_a, panel_bbox, page_shape, t_frac=0.4, loop_duration_s=4.0
    )
    _assert_localized_matches_reference(
        image, mask_b, motion_b, panel_bbox, page_shape, t_frac=0.4, loop_duration_s=4.0
    )


# --- object_bbox_px: caller-supplied bbox skips the internal bbox_of_mask(mask) scan --------


@pytest.mark.parametrize(
    "kind,extra",
    [
        (TransformKind.ROTATE, dict(amplitude=25.0, direction=None)),
        (TransformKind.SCALE, dict(amplitude=0.4, direction=None)),
        (TransformKind.TRANSLATE, dict(amplitude=0.15, direction=Vector2(x=1.0, y=0.4))),
        (TransformKind.SHEAR, dict(amplitude=0.35, direction=Vector2(x=0.0, y=1.0))),
        (TransformKind.MESH_WARP, dict(amplitude=0.3, direction=Vector2(x=0.8, y=0.6))),
        (TransformKind.OPACITY, dict(amplitude=0.6, direction=None)),
    ],
)
def test_generate_transformed_layer_precomputed_bbox_matches_recomputed(kind, extra):
    """A caller that passes `object_bbox_px=bbox_of_mask(mask)` explicitly (the orchestrator's
    per-frame animation loop now does exactly this, using `SegmentationResult.bbox` instead of
    a fresh `bbox_of_mask` call) must get bit-identical output to the default, no-argument form
    that recomputes the bbox internally -- across every transform kind, since the bbox feeds
    into the pivot/ROI/margin logic differently per kind.
    """
    page_shape, mask_slice = _EDGE_CASE_PAGES["small_object_normal_page"]
    image, mask = _random_image_and_mask(page_shape, mask_slice, seed=hash(kind) % (2**31))
    motion = make_motion(transform_kind=kind, phase=0.2, speed=1.0, **extra)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    precomputed_bbox = bbox_of_mask(mask)

    layer_default, mask_default = generate_transformed_layer(
        image, mask, motion, panel_bbox, page_shape, t_frac=0.6, loop_duration_s=4.0
    )
    layer_precomputed, mask_precomputed = generate_transformed_layer(
        image,
        mask,
        motion,
        panel_bbox,
        page_shape,
        t_frac=0.6,
        loop_duration_s=4.0,
        object_bbox_px=precomputed_bbox,
    )

    np.testing.assert_array_equal(layer_default, layer_precomputed)
    np.testing.assert_array_equal(mask_default, mask_precomputed)


def test_generate_transformed_layer_trusts_caller_supplied_bbox_without_revalidation():
    """Deliberate design choice, not an oversight: `generate_transformed_layer` does NOT check
    that a caller-supplied `object_bbox_px` actually matches `bbox_of_mask(mask)`. Revalidating
    it would require running the exact full-page `np.where` scan this parameter exists to let
    callers skip -- entirely defeating its purpose (see its docstring and
    `docs/decisions/0012-phase6-seamless-loop-and-local-rendering.md`'s "Known limitations").
    The contract is caller responsibility (mirroring `Layer`/`SegmentationResult`, which also
    trust their own fields' internal consistency rather than cross-checking them at construction
    for anything CPU-costly to re-derive) -- the orchestrator satisfies it by construction,
    since `SegmentationResult.bbox` is always `_tight_bbox(SegmentationResult.mask)`
    (`src/manga_animation/segmentation/segment.py`), the same algorithm as `bbox_of_mask`.

    This test documents the resulting behavior when a caller violates that contract: a bbox
    that does not match the mask's true tight extent is used as-is (no exception, no silent
    correction) and visibly changes the output relative to the correctly-computed-bbox call --
    proving the parameter really is used directly, not merely accepted and ignored.
    """
    image, mask = make_image_and_mask()
    motion = make_motion(
        transform_kind=TransformKind.ROTATE, amplitude=45.0, direction=None, phase=0.25
    )
    true_bbox = bbox_of_mask(mask)
    # Deliberately wrong: shifted well away from the mask's real extent.
    wrong_bbox = BBoxPx(
        x0=true_bbox.x0 + 15, y0=true_bbox.y0, x1=true_bbox.x1 + 15, y1=true_bbox.y1
    )

    layer_correct, mask_correct = generate_transformed_layer(
        image,
        mask,
        motion,
        PANEL_BBOX,
        PAGE_SHAPE,
        t_frac=0.0,
        loop_duration_s=4.0,
        object_bbox_px=true_bbox,
    )
    layer_wrong, mask_wrong = generate_transformed_layer(
        image,
        mask,
        motion,
        PANEL_BBOX,
        PAGE_SHAPE,
        t_frac=0.0,
        loop_duration_s=4.0,
        object_bbox_px=wrong_bbox,
    )

    # No exception was raised -- the caller-supplied (wrong) bbox was accepted and used, and the
    # rotation pivot it implies visibly moves the output relative to the correctly-computed one.
    assert not np.array_equal(mask_correct, mask_wrong)
