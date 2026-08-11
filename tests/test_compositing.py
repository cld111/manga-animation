from __future__ import annotations

import numpy as np
import pytest

from manga_animation.animation.transforms import generate_transformed_layer
from manga_animation.compositing import composite_frame
from manga_animation.pipeline.types import BBoxPx, ReconstructionResult
from manga_animation.schemas.animation_plan import MotionSpec, PivotSpec, TimingSpec, Vector2

PAGE_SHAPE = (40, 50)  # (h, w)
PANEL_BBOX = BBoxPx(x0=0, y0=0, x1=PAGE_SHAPE[1], y1=PAGE_SHAPE[0])


def make_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    h, w = PAGE_SHAPE
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (5, 5, 5)
    image[10:25, 15:35] = (200, 100, 50)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:25, 15:35] = 255
    return image, mask


# --- the core invariant: bit-exact preservation outside the mask ---------------


@pytest.mark.parametrize(
    "transform_kind,direction",
    [("translate", Vector2(x=1.0, y=0.0)), ("rotate", None)],
)
def test_composite_frame_is_bit_exact_outside_the_mask_across_a_sequence(transform_kind, direction):
    image, mask = make_image_and_mask()
    motion = MotionSpec(
        transform_kind=transform_kind,
        direction=direction,
        amplitude=0.2 if transform_kind == "translate" else 30.0,
        speed=1.0,
        pivot=PivotSpec(),
        timing=TimingSpec(),
    )

    fps, duration_s = 12, 2.0
    frame_count = fps * int(duration_s)
    for i in range(frame_count):
        t_frac = i / frame_count
        layer, layer_mask = generate_transformed_layer(
            image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=t_frac, loop_duration_s=duration_s
        )
        frame = composite_frame(image, layer, layer_mask)

        outside = layer_mask == 0
        assert np.array_equal(frame[outside], image[outside]), f"frame {i} altered a static pixel"


def test_composite_frame_with_full_mask_uses_layer_entirely():
    image, mask = make_image_and_mask()
    layer = np.full_like(image, 123)
    full_mask = np.full(mask.shape, 255, dtype=np.uint8)
    frame = composite_frame(image, layer, full_mask)
    np.testing.assert_array_equal(frame, layer)


def test_composite_frame_with_empty_mask_returns_original_exactly():
    image, mask = make_image_and_mask()
    layer = np.full_like(image, 123)
    empty_mask = np.zeros(mask.shape, dtype=np.uint8)
    frame = composite_frame(image, layer, empty_mask)
    np.testing.assert_array_equal(frame, image)


def test_composite_frame_blends_partial_alpha():
    image, mask = make_image_and_mask()
    layer = np.full_like(image, 200)
    partial_mask = np.full(mask.shape, 128, dtype=np.uint8)  # ~50% alpha
    frame = composite_frame(image, layer, partial_mask)
    # A pixel where image=5 blended ~50% toward layer=200 should land roughly in between,
    # and strictly greater than the original everywhere the layer is brighter.
    assert np.all(frame >= image)
    assert np.any(frame > image)


# --- reconstruction-hole compositing --------------------------------------------


def test_composite_frame_uses_reconstructed_pixels_in_the_revealed_hole():
    image, mask = make_image_and_mask()
    h, w = PAGE_SHAPE

    hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask[10:25, 15:20] = 255  # left slice of the original object's footprint

    filled_pixels = image.copy()
    filled_pixels[10:25, 15:20] = (9, 9, 9)  # a distinct "reconstructed" color

    reconstruction = ReconstructionResult(
        object_id="obj",
        hole_mask=hole_mask,
        filled_pixels=filled_pixels,
        model_id="fake-lama",
    )

    # This frame's layer has moved fully off the hole region (layer_mask==0 there).
    layer = image.copy()
    layer_mask = np.zeros((h, w), dtype=np.uint8)
    layer_mask[10:25, 30:35] = 255  # object now sits elsewhere, hole is fully revealed

    frame = composite_frame(image, layer, layer_mask, reconstruction=reconstruction)

    np.testing.assert_array_equal(frame[10:25, 15:20], filled_pixels[10:25, 15:20])
    # Everywhere else outside the (moved) layer mask and outside the hole: still untouched.
    outside_layer_and_hole = (layer_mask == 0) & (hole_mask == 0)
    np.testing.assert_array_equal(frame[outside_layer_and_hole], image[outside_layer_and_hole])


def test_composite_frame_does_not_touch_hole_where_layer_still_covers_it():
    image, mask = make_image_and_mask()
    h, w = PAGE_SHAPE

    hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask[10:25, 15:20] = 255

    filled_pixels = image.copy()
    filled_pixels[10:25, 15:20] = (9, 9, 9)

    reconstruction = ReconstructionResult(
        object_id="obj", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    # This frame's layer still covers the hole region (layer_mask != 0 there) — the
    # reconstruction must NOT be substituted in, since nothing is actually revealed yet.
    layer = np.full_like(image, 77)
    layer_mask = mask.copy()

    frame = composite_frame(image, layer, layer_mask, reconstruction=reconstruction)
    assert not np.array_equal(frame[10:25, 15:20], filled_pixels[10:25, 15:20])
    np.testing.assert_array_equal(frame[10:25, 15:20], layer[10:25, 15:20])
