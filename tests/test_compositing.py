from __future__ import annotations

import numpy as np
import pytest

from manga_animation.animation.transforms import generate_transformed_layer
from manga_animation.compositing import composite_frame, composite_frame_stack
from manga_animation.pipeline.types import BBoxPx, Layer, ReconstructionResult
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


# --- composite_frame_stack: multi-layer compositing (Phase 4) -------------------


def _make_layer(
    image: np.ndarray,
    region: tuple[slice, slice],
    color: tuple[int, int, int],
    *,
    object_id: str,
    z_order: int,
    n_frames: int = 1,
) -> Layer:
    """A single-frame-repeated `Layer` that paints `color` into `region` and is transparent

    (mask == 0) elsewhere -- direct, hand-built fixtures rather than routing through
    `generate_transformed_layer`, since these tests are about `composite_frame_stack`'s own
    ordering/hole-filling logic, not about transform generation (already covered by
    tests/test_animation.py).
    """
    layer_image = image.copy()
    layer_image[region] = color
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[region] = 255
    frames = tuple((layer_image, mask) for _ in range(n_frames))
    return Layer(object_id=object_id, frames=frames, z_order=z_order)


def test_composite_frame_stack_two_nonoverlapping_layers():
    image, _ = make_image_and_mask()
    region_a = (slice(2, 8), slice(2, 8))
    region_b = (slice(30, 36), slice(40, 46))
    layer_a = _make_layer(image, region_a, (200, 0, 0), object_id="a", z_order=0)
    layer_b = _make_layer(image, region_b, (0, 200, 0), object_id="b", z_order=0)

    frame = composite_frame_stack(image, [layer_a, layer_b], frame_index=0)

    assert np.all(frame[region_a] == (200, 0, 0))
    assert np.all(frame[region_b] == (0, 200, 0))
    mask_a = np.zeros(image.shape[:2], dtype=bool)
    mask_a[region_a] = True
    mask_b = np.zeros(image.shape[:2], dtype=bool)
    mask_b[region_b] = True
    outside = ~(mask_a | mask_b)
    np.testing.assert_array_equal(frame[outside], image[outside])


def test_composite_frame_stack_overlapping_layers_respect_z_order():
    image, _ = make_image_and_mask()
    region_a = (slice(5, 20), slice(5, 20))
    region_b = (slice(12, 27), slice(12, 27))  # overlaps region_a in rows/cols 12-20

    layer_a = _make_layer(image, region_a, (200, 0, 0), object_id="a", z_order=0)  # behind
    layer_b = _make_layer(image, region_b, (0, 200, 0), object_id="b", z_order=1)  # in front

    frame = composite_frame_stack(image, [layer_a, layer_b], frame_index=0)

    overlap = (slice(12, 20), slice(12, 20))
    assert np.all(frame[overlap] == (0, 200, 0)), "higher z_order layer must win in the overlap"
    # A's non-overlapping region still shows A.
    assert np.all(frame[5:12, 5:12] == (200, 0, 0))
    # B's non-overlapping region still shows B.
    assert np.all(frame[20:27, 20:27] == (0, 200, 0))


def test_composite_frame_stack_z_order_ties_broken_by_object_id_deterministically():
    image, _ = make_image_and_mask()
    region_a = (slice(5, 20), slice(5, 20))
    region_b = (slice(12, 27), slice(12, 27))

    # Equal z_order -- tie-break must fall to object_id ("b" sorts after "a", so "b" is drawn
    # last/on top).
    layer_a = _make_layer(image, region_a, (200, 0, 0), object_id="a", z_order=0)
    layer_b = _make_layer(image, region_b, (0, 200, 0), object_id="b", z_order=0)

    overlap = (slice(12, 20), slice(12, 20))
    frame_ab = composite_frame_stack(image, [layer_a, layer_b], frame_index=0)
    frame_ba = composite_frame_stack(image, [layer_b, layer_a], frame_index=0)

    # Reproducible regardless of input list order, and matches the documented tie-break rule.
    np.testing.assert_array_equal(frame_ab, frame_ba)
    assert np.all(frame_ab[overlap] == (0, 200, 0))


@pytest.mark.parametrize("n_layers", [0, 1, 2])
def test_composite_frame_stack_static_region_preserved_for_n_layers(n_layers):
    image, _ = make_image_and_mask()
    regions = [(slice(2, 8), slice(2, 8)), (slice(30, 36), slice(40, 46))]
    colors = [(200, 0, 0), (0, 200, 0)]
    layers = [
        _make_layer(image, regions[i], colors[i], object_id=f"obj_{i}", z_order=i)
        for i in range(n_layers)
    ]

    frame = composite_frame_stack(image, layers, frame_index=0)

    if n_layers == 0:
        np.testing.assert_array_equal(frame, image)
        assert frame is not image  # a copy, not the same array reference
        return

    covered = np.zeros(image.shape[:2], dtype=bool)
    for region in regions[:n_layers]:
        covered[region] = True
    outside = ~covered
    np.testing.assert_array_equal(frame[outside], image[outside])


def test_composite_frame_stack_single_layer_matches_composite_frame_bit_for_bit():
    image, mask = make_image_and_mask()
    motion = MotionSpec(
        transform_kind="translate",
        direction=Vector2(x=1.0, y=0.0),
        amplitude=0.2,
        speed=1.0,
        pivot=PivotSpec(),
        timing=TimingSpec(),
    )
    layer_image, layer_mask = generate_transformed_layer(
        image, mask, motion, PANEL_BBOX, PAGE_SHAPE, t_frac=0.25, loop_duration_s=2.0
    )

    hole_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    hole_mask[10:25, 15:20] = 255
    filled_pixels = image.copy()
    filled_pixels[10:25, 15:20] = (9, 9, 9)
    reconstruction = ReconstructionResult(
        object_id="obj", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    expected = composite_frame(image, layer_image, layer_mask, reconstruction=reconstruction)

    layer = Layer(object_id="obj", frames=((layer_image, layer_mask),), z_order=0)
    actual = composite_frame_stack(
        image, [layer], frame_index=0, reconstructions={"obj": reconstruction}
    )

    np.testing.assert_array_equal(actual, expected)


def test_composite_frame_stack_reconstruction_fills_hole_when_uncontested():
    image, _ = make_image_and_mask()
    h, w = PAGE_SHAPE

    hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask[10:25, 15:20] = 255
    filled_pixels = image.copy()
    filled_pixels[10:25, 15:20] = (9, 9, 9)
    reconstruction = ReconstructionResult(
        object_id="a", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    # Object "a" has moved off the hole region entirely this frame.
    layer_a = _make_layer(
        image, (slice(10, 25), slice(30, 35)), (200, 0, 0), object_id="a", z_order=0
    )
    # A second, unrelated object elsewhere -- does not touch the hole region.
    layer_b = _make_layer(image, (slice(0, 5), slice(0, 5)), (0, 200, 0), object_id="b", z_order=1)

    frame = composite_frame_stack(
        image, [layer_a, layer_b], frame_index=0, reconstructions={"a": reconstruction}
    )

    np.testing.assert_array_equal(frame[10:25, 15:20], filled_pixels[10:25, 15:20])


def test_composite_frame_stack_reconstruction_skipped_when_another_layer_covers_the_hole():
    image, _ = make_image_and_mask()
    h, w = PAGE_SHAPE

    hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask[10:25, 15:20] = 255
    filled_pixels = image.copy()
    filled_pixels[10:25, 15:20] = (9, 9, 9)
    reconstruction = ReconstructionResult(
        object_id="a", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    # Object "a" moved off the hole region.
    layer_a = _make_layer(
        image, (slice(10, 25), slice(30, 35)), (200, 0, 0), object_id="a", z_order=0
    )
    # Object "b" -- a higher-z-order layer -- happens to be sitting exactly on what would be
    # "a"'s revealed hole this frame.
    layer_b = _make_layer(
        image, (slice(10, 25), slice(15, 20)), (0, 200, 0), object_id="b", z_order=1
    )

    frame = composite_frame_stack(
        image, [layer_a, layer_b], frame_index=0, reconstructions={"a": reconstruction}
    )

    # "b"'s pixels win -- the reconstruction fill must not have been applied/fought over.
    assert not np.array_equal(frame[10:25, 15:20], filled_pixels[10:25, 15:20])
    expected_b_color = np.full((15, 5, 3), (0, 200, 0), dtype=np.uint8)
    np.testing.assert_array_equal(frame[10:25, 15:20], expected_b_color)
