from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from manga_animation.reconstruction import reconstruct_hidden_region

PAGE_SHAPE = (30, 40)  # (h, w)


def make_image() -> np.ndarray:
    h, w = PAGE_SHAPE
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (5, 5, 5)
    return image


class FakeLamaClient:
    """A `ReconstructionClient` double: returns a canned, controllable output image."""

    def __init__(self, output: Image.Image, *, fail_on_load: bool = False):
        self.output = output
        self.fail_on_load = fail_on_load
        self.loaded = False
        self.unloaded = False
        self.received_masks: list[Image.Image] = []

    def load(self) -> None:
        if self.fail_on_load:
            raise RuntimeError("simulated load failure")
        self.loaded = True

    def inpaint(self, image: Image.Image, hole_mask: Image.Image) -> Image.Image:
        self.received_masks.append(hole_mask)
        return self.output

    def unload(self) -> None:
        self.unloaded = True


def test_no_hole_when_object_never_uncovered_returns_none():
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    # Every transformed-mask frame covers the exact same region -> nothing ever revealed.
    transformed_masks = [original_mask.copy() for _ in range(5)]

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, object_id="obj_1"
    )

    assert result is None
    assert client.loaded is False  # never called — no reconstruction work needed


def test_hole_includes_a_region_uncovered_at_only_one_frame_even_if_another_frame_covers_it():
    """Reconstruction-hardening regression test for the real bug this fix closes: a pixel that

    is left uncovered at frame A (and would show ghosted original-page content there) still
    needs a real fill available for frame A, even though a *different* frame (B) covers that
    same pixel -- compositing blends each frame from its own transformed mask independently
    (see `compositing.composite_frame`), so "some other frame eventually re-covers it" does not
    help the frame where it doesn't. This is exactly what happens in real motion: frame index 0
    (`t_frac=0`) is every `cycle`-mode motion's rest pose and therefore always reproduces
    `original_mask` almost exactly -- a hole formula keyed on "never covered by ANY frame"
    would be vacuous for practically every real render, which is the bug this test guards
    against regressing to.
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255

    # Frame A (e.g. the rest pose) covers the whole original region -- on its own, "no hole
    # needed". Frame B (e.g. a rotated mid-loop frame) covers only the left half, leaving the
    # right half uncovered *at frame B's time* -- that half needs a real fill value for
    # whichever frame(s) actually leave it exposed, regardless of frame A fully covering it.
    frame_a = original_mask.copy()
    frame_b = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_b[10:20, 10:15] = 255

    output = Image.fromarray(image)
    client = FakeLamaClient(output=output)
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b], client, object_id="obj_1"
    )
    assert result is not None
    assert np.all(result.hole_mask[10:20, 15:20] == 255)  # the part only frame_a covers
    assert np.all(result.hole_mask[10:20, 10:15] == 0)  # both frames cover this part
    assert client.loaded is True
    assert client.unloaded is True


def test_hole_is_empty_when_every_frame_covers_the_same_region():
    """The complement of the buggy case above: if every frame really does cover the exact same

    (full) region, no pixel is ever left exposed at any frame, so no hole is needed -- this is
    the one case where "union" and "intersection" framings happen to agree.
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    frame_a = original_mask.copy()
    frame_b = original_mask.copy()

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b], client, object_id="obj_1"
    )
    assert result is None
    assert client.loaded is False


def test_hole_gap_never_covered_by_any_frame_is_still_detected():
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    frame_a = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_a[10:20, 10:15] = 255
    frame_b_smaller = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_b_smaller[10:20, 15:18] = 255

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b_smaller], client, object_id="obj_1"
    )
    assert result is not None
    assert np.any(result.hole_mask[10:20, 18:20] > 0)
    assert client.loaded is True
    assert client.unloaded is True


def test_raw_output_of_different_size_is_normalized_to_source_geometry():
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    transformed_masks = [np.zeros(PAGE_SHAPE, dtype=np.uint8)]  # never covered -> full hole

    # Simulate LaMa's real, documented behavior: raw output a few px larger than the input
    # (internal stride padding, per ADR 0005) — must be corrected before ReconstructionResult
    # is constructed.
    h, w = PAGE_SHAPE
    oversized_output = Image.fromarray(np.full((h + 6, w + 4, 3), 250, dtype=np.uint8))
    client = FakeLamaClient(output=oversized_output)

    result = reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, object_id="obj_1"
    )

    assert result is not None
    assert result.filled_pixels.shape[:2] == PAGE_SHAPE
    assert result.hole_mask.shape == PAGE_SHAPE
    assert result.object_id == "obj_1"
    assert result.model_id == "lama-large"


def test_client_load_failure_propagates():
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    transformed_masks = [np.zeros(PAGE_SHAPE, dtype=np.uint8)]

    client = FakeLamaClient(output=Image.fromarray(image), fail_on_load=True)
    with pytest.raises(RuntimeError, match="simulated load failure"):
        reconstruct_hidden_region(
            image, original_mask, transformed_masks, client, object_id="obj_1"
        )


# --- Phase 4 reconstruction-hardening: real failure-mode audit ----------------------------


def test_empty_original_mask_returns_none_without_crashing():
    """An empty (all-zero) `original_mask` -- e.g. a degenerate segmentation result -- must

    never be treated as a real hole; the object was never drawn anywhere, so nothing can be
    "revealed" by its motion.
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)  # nothing set
    transformed_masks = [np.zeros(PAGE_SHAPE, dtype=np.uint8) for _ in range(3)]

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, object_id="obj_1"
    )
    assert result is None
    assert client.loaded is False


def test_degenerate_single_pixel_mask_is_handled_without_crashing():
    """A 1x1-pixel mask (the smallest possible non-empty region) must not crash the hole

    computation or the reconstruction call -- no code here assumes a minimum mask size.
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[15, 20] = 255  # exactly one pixel
    transformed_masks = [np.zeros(PAGE_SHAPE, dtype=np.uint8)]  # that one pixel never re-covered

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, object_id="obj_1"
    )
    assert result is not None
    assert result.hole_mask[15, 20] == 255
    assert int(np.sum(result.hole_mask > 0)) == 1


def test_mask_touching_the_image_boundary_stays_within_bounds():
    """An object whose mask touches the page edge (e.g. hair starting at the top row, y=0)

    must not produce an out-of-bounds hole region -- the hole mask is always exactly the source
    image's own shape, so there is no separate geometry that could exceed it.
    """
    image = make_image()
    h, w = PAGE_SHAPE
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[0:10, 0:10] = 255  # flush against the top-left corner
    transformed_masks = [np.zeros(PAGE_SHAPE, dtype=np.uint8)]  # fully uncovered every frame

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, object_id="obj_1"
    )
    assert result is not None
    assert result.hole_mask.shape == (h, w)
    assert result.filled_pixels.shape == (h, w, 3)
    np.testing.assert_array_equal(result.hole_mask > 0, original_mask > 0)


def test_multiple_disconnected_holes_are_all_included_in_one_pass():
    """A single motion can reveal two geometrically separate regions (e.g. both sides of a

    rotating object) -- the hole mask must include every disconnected region, and
    `reconstruct_hidden_region` must still make exactly one inpainting call covering all of
    them (never assume or require one connected hole).
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 5:35] = 255  # one wide horizontal strip

    # Frame A leaves only the left edge uncovered; frame B leaves only the right edge
    # uncovered; the middle stays covered by both -> two disconnected holes, one on each side.
    frame_a = original_mask.copy()
    frame_a[10:20, 5:10] = 0
    frame_b = original_mask.copy()
    frame_b[10:20, 30:35] = 0

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b], client, object_id="obj_1"
    )
    assert result is not None
    assert np.all(result.hole_mask[10:20, 5:10] == 255)  # left disconnected region
    assert np.all(result.hole_mask[10:20, 30:35] == 255)  # right disconnected region
    assert np.all(result.hole_mask[10:20, 10:30] == 0)  # covered by both -> not part of the hole
    assert len(client.received_masks) == 1  # one inpainting call handles both regions at once


def test_thin_one_pixel_wide_hole_is_computed_correctly():
    """A genuinely thin (1px-wide) revealed sliver -- plausible at a mask's interpolated edge

    -- must be computed exactly, not rounded away or expanded by any morphological cleanup
    (none is applied, deliberately -- this project has no calibration evidence a cleanup step
    would help more than it would risk shrinking/growing a real hole boundary).
    """
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255
    frame_a = original_mask.copy()
    frame_a[10:20, 19] = 0  # exactly one column (10 pixels tall, 1 pixel wide) left uncovered

    client = FakeLamaClient(output=Image.fromarray(image))
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a], client, object_id="obj_1"
    )
    assert result is not None
    assert int(np.sum(result.hole_mask > 0)) == 10  # exactly the thin column, nothing more
