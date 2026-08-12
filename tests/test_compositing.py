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


def test_composite_frame_stack_reconstruction_partially_covered_by_another_layer():
    """Phase 4 reconstruction-hardening: the "other layer covers it" check in

    `composite_frame_stack` is per-pixel, not all-or-nothing -- when a second object's layer
    only covers HALF of what would be object "a"'s revealed hole, the covered half must show
    "b"'s pixels and the UNCOVERED half must still get "a"'s reconstruction fill, not be left
    showing raw (wrong) original-page content.
    """
    image, _ = make_image_and_mask()

    hole_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    hole_mask[10:25, 15:25] = 255  # a's full potential hole: columns 15-25
    filled_pixels = image.copy()
    filled_pixels[10:25, 15:25] = (9, 9, 9)
    reconstruction = ReconstructionResult(
        object_id="a", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    layer_a = _make_layer(
        image, (slice(10, 25), slice(30, 35)), (200, 0, 0), object_id="a", z_order=0
    )
    # "b" only covers the LEFT half (columns 15-20) of what would be "a"'s hole -- the right
    # half (columns 20-25) is not covered by anything and must still get "a"'s fill.
    layer_b = _make_layer(
        image, (slice(10, 25), slice(15, 20)), (0, 200, 0), object_id="b", z_order=1
    )

    frame = composite_frame_stack(
        image, [layer_a, layer_b], frame_index=0, reconstructions={"a": reconstruction}
    )

    expected_b_color = np.full((15, 5, 3), (0, 200, 0), dtype=np.uint8)
    np.testing.assert_array_equal(frame[10:25, 15:20], expected_b_color)  # "b" wins here
    np.testing.assert_array_equal(frame[10:25, 20:25], filled_pixels[10:25, 20:25])  # "a"'s fill


def test_composite_frame_stack_five_objects_respect_z_order_and_static_region():
    """Deterministic, human-checkable companion to the 2-layer z-order test above, at the
    5-object scale this project has actually observed in a real plan (Phase 5/5.1 audits;
    see docs/decisions/0010-multi-object-layer-decomposition.md).
    """
    image, _ = make_image_and_mask()
    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (0, 200, 200)]
    # Each region overlaps the next by a few columns, forming a staircase where every higher
    # z_order object must win exactly in its overlap band.
    regions = [
        (slice(2, 12), slice(0, 10)),
        (slice(2, 12), slice(8, 18)),
        (slice(2, 12), slice(16, 26)),
        (slice(2, 12), slice(24, 34)),
        (slice(2, 12), slice(32, 42)),
    ]
    layers = [
        _make_layer(image, regions[i], colors[i], object_id=f"obj_{i}", z_order=i) for i in range(5)
    ]

    frame = composite_frame_stack(image, layers, frame_index=0)

    # Each object's own non-overlapping band still shows its own color.
    assert np.all(frame[2:12, 0:8] == colors[0])
    assert np.all(frame[2:12, 34:42] == colors[4])
    # Every overlap band is won by the higher z_order (later-drawn) object.
    assert np.all(frame[2:12, 8:10] == colors[1])
    assert np.all(frame[2:12, 16:18] == colors[2])
    assert np.all(frame[2:12, 24:26] == colors[3])
    assert np.all(frame[2:12, 32:34] == colors[4])
    # Static region (untouched by any of the 5 objects) stays bit-exact.
    covered = np.zeros(image.shape[:2], dtype=bool)
    for region in regions:
        covered[region] = True
    np.testing.assert_array_equal(frame[~covered], image[~covered])


# --- Phase 6: local-region blending must match the old full-page formula ------------------


def _composite_frame_full_page_reference(
    original: np.ndarray,
    layer: np.ndarray,
    layer_mask: np.ndarray,
    *,
    reconstruction: ReconstructionResult | None = None,
) -> np.ndarray:
    """Pre-Phase-6 `composite_frame`: identical math, evaluated over the whole page every call
    instead of each mask's own local bbox — kept verbatim as the deterministic reference the
    localized version must reproduce exactly.
    """
    if reconstruction is not None:
        revealed_this_frame = (layer_mask == 0) & (reconstruction.hole_mask != 0)
        plate = original.copy()
        plate[revealed_this_frame] = reconstruction.filled_pixels[revealed_this_frame]
    else:
        plate = original.copy()

    alpha = (layer_mask.astype(np.float32) / 255.0)[..., None]
    frame = layer.astype(np.float32) * alpha + plate.astype(np.float32) * (1.0 - alpha)
    return frame.astype(np.uint8)


def _composite_frame_stack_full_page_reference(
    original: np.ndarray,
    layers: list[Layer],
    frame_index: int,
    *,
    reconstructions: dict[str, ReconstructionResult] | None = None,
) -> np.ndarray:
    """Pre-Phase-6 `composite_frame_stack`: identical math, evaluated over the whole page
    instead of each mask's local bbox — kept verbatim as the reference the localized version
    must reproduce exactly.
    """
    if not layers:
        return original.copy()

    ordered = sorted(layers, key=lambda layer: (layer.z_order, layer.object_id))
    current_frames = {layer.object_id: layer.frames[frame_index] for layer in ordered}

    plate = original.copy()
    if reconstructions:
        for object_id, recon in reconstructions.items():
            own_frame = current_frames.get(object_id)
            if own_frame is None:
                continue
            own_mask = own_frame[1]
            other_covered = np.zeros(original.shape[:2], dtype=bool)
            for other_id, (_, other_mask) in current_frames.items():
                if other_id == object_id:
                    continue
                other_covered |= other_mask > 0
            revealed_this_frame = (own_mask == 0) & (recon.hole_mask != 0) & ~other_covered
            plate[revealed_this_frame] = recon.filled_pixels[revealed_this_frame]

    frame = plate.astype(np.float32)
    for layer in ordered:
        layer_image, layer_mask = current_frames[layer.object_id]
        alpha = (layer_mask.astype(np.float32) / 255.0)[..., None]
        frame = layer_image.astype(np.float32) * alpha + frame * (1.0 - alpha)

    return frame.astype(np.uint8)


@pytest.mark.parametrize(
    "page_shape,mask_slice",
    [
        ((60, 80), (slice(20, 40), slice(30, 50))),  # small object, normal page
        ((60, 80), (slice(0, 12), slice(30, 50))),  # touching the top edge
        ((60, 80), (slice(48, 60), slice(30, 50))),  # touching the bottom edge
        ((60, 80), (slice(20, 40), slice(0, 12))),  # touching the left edge
        ((60, 80), (slice(20, 40), slice(68, 80))),  # touching the right edge
        ((60, 80), (slice(0, 10), slice(70, 80))),  # touching the top-right corner
        ((60, 80), (slice(2, 58), slice(2, 78))),  # large object, near-full-page
        ((60, 80), (slice(30, 31), slice(40, 41))),  # 1x1 pixel object
        ((720, 90), (slice(50, 690), slice(5, 85))),  # extreme-aspect-ratio page
    ],
)
def test_composite_frame_matches_full_page_reference_with_partial_alpha_and_hole(
    page_shape, mask_slice
):
    rng = np.random.default_rng(42)
    h, w = page_shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (5, 5, 5)
    image[mask_slice] = (200, 100, 50)

    layer = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    layer_mask = np.zeros((h, w), dtype=np.uint8)
    layer_mask[mask_slice] = rng.integers(1, 256, size=layer_mask[mask_slice].shape, dtype=np.uint8)

    hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_mask[mask_slice] = 255
    filled_pixels = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    reconstruction = ReconstructionResult(
        object_id="obj", hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake-lama"
    )

    actual = composite_frame(image, layer, layer_mask, reconstruction=reconstruction)
    expected = _composite_frame_full_page_reference(
        image, layer, layer_mask, reconstruction=reconstruction
    )
    np.testing.assert_array_equal(actual, expected)


def test_composite_frame_stack_matches_full_page_reference_at_five_object_scale():
    # Realistic upper bound observed in this project's own multi-object evidence (Phase 5/5.1
    # audits) — see docs/decisions/0010-multi-object-layer-decomposition.md.
    rng = np.random.default_rng(7)
    h, w = 100, 140
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (5, 5, 5)

    regions = [
        (slice(0, 15), slice(0, 15)),
        (slice(0, 15), slice(120, 140)),
        (slice(85, 100), slice(0, 20)),
        (slice(40, 60), slice(60, 90)),  # overlaps region below
        (slice(50, 70), slice(70, 100)),  # overlaps region above
    ]
    layers = []
    reconstructions = {}
    for i, region in enumerate(regions):
        object_id = f"obj_{i}"
        layer_image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Partial alpha (not a hard 0/255 mask) to exercise real float accumulation, especially
        # in the two overlapping regions above.
        mask[region] = rng.integers(1, 256, size=mask[region].shape, dtype=np.uint8)
        layers.append(Layer(object_id=object_id, frames=((layer_image, mask),), z_order=i))

        hole_mask = np.zeros((h, w), dtype=np.uint8)
        hole_mask[region] = 255
        filled_pixels = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        reconstructions[object_id] = ReconstructionResult(
            object_id=object_id, hole_mask=hole_mask, filled_pixels=filled_pixels, model_id="fake"
        )

    actual = composite_frame_stack(image, layers, frame_index=0, reconstructions=reconstructions)
    expected = _composite_frame_stack_full_page_reference(
        image, layers, frame_index=0, reconstructions=reconstructions
    )
    np.testing.assert_array_equal(actual, expected)


def test_composite_frame_stack_matches_full_page_reference_with_larger_frame_count():
    rng = np.random.default_rng(11)
    h, w = 50, 70
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (5, 5, 5)

    region_a = (slice(5, 30), slice(5, 30))
    region_b = (slice(15, 40), slice(15, 45))  # overlaps region_a

    n_frames = 96  # a 4s loop at 24fps -- the schema's real default duration/fps
    frames_a = []
    frames_b = []
    for _ in range(n_frames):
        img_a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask_a = np.zeros((h, w), dtype=np.uint8)
        mask_a[region_a] = rng.integers(1, 256, size=mask_a[region_a].shape, dtype=np.uint8)
        frames_a.append((img_a, mask_a))

        img_b = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask_b = np.zeros((h, w), dtype=np.uint8)
        mask_b[region_b] = rng.integers(1, 256, size=mask_b[region_b].shape, dtype=np.uint8)
        frames_b.append((img_b, mask_b))

    layer_a = Layer(object_id="a", frames=tuple(frames_a), z_order=0)
    layer_b = Layer(object_id="b", frames=tuple(frames_b), z_order=1)

    for frame_index in (0, n_frames // 2, n_frames - 1):
        actual = composite_frame_stack(image, [layer_a, layer_b], frame_index=frame_index)
        expected = _composite_frame_stack_full_page_reference(
            image, [layer_a, layer_b], frame_index=frame_index
        )
        np.testing.assert_array_equal(actual, expected)
