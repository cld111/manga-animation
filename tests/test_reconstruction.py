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


def test_hole_computed_as_union_of_revealed_region_across_frames():
    image = make_image()
    original_mask = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    original_mask[10:20, 10:20] = 255

    # Frame A covers the left half, frame B covers the right half -> together they cover
    # everything EXCEPT nothing, so no hole; but each individually reveals the other half at
    # different times, which is exactly what the union-of-revealed-region computation must
    # catch by looking across ALL frames, not just one.
    frame_a = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_a[10:20, 10:15] = 255
    frame_b = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_b[10:20, 15:20] = 255

    output = Image.fromarray(image)
    client = FakeLamaClient(output=output)
    result = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b], client, object_id="obj_1"
    )
    assert result is None  # union of frame_a|frame_b == original_mask, nothing left uncovered

    # Now make frame B smaller, leaving a real gap never covered by either frame.
    frame_b_smaller = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    frame_b_smaller[10:20, 15:18] = 255
    result2 = reconstruct_hidden_region(
        image, original_mask, [frame_a, frame_b_smaller], client, object_id="obj_1"
    )
    assert result2 is not None
    assert np.any(result2.hole_mask[10:20, 18:20] > 0)
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
