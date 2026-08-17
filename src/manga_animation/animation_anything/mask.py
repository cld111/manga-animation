"""Merging accepted SAM masks into the single AnimateAnything motion mask.

The upstream model accepts exactly ONE binary mask marking the region that is allowed to
move (everything outside stays frozen). The pipeline's segmentation stage produces one SAM
mask per accepted object; the generative animation stage therefore merges their union into a
single uint8 0/255 array. All masks are full-source-image-shaped (the project's
`SegmentationResult` contract), so the merge is a pure elementwise OR with no geometry work.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manga_animation.pipeline.types import MaskArray


def merge_motion_masks(masks: Sequence[MaskArray]) -> MaskArray:
    """Union of full-source-image-shaped SAM masks into one uint8 0/255 motion mask.

    Raises `ValueError` on empty input (fail closed) or on a shape mismatch with the first
    mask (the caller is responsible for passing masks of one image). The result is the
    elementwise OR of `mask > 0`, scaled to 0/255 -- matching the upstream worker's
    `np_mask[np_mask != 0] = 255` preprocessing.
    """
    if not masks:
        raise ValueError("cannot merge an empty mask list into a motion mask")
    merged = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape != merged.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match the first mask's {merged.shape}"
            )
        merged = np.maximum(merged, (mask > 0).astype(np.uint8))
    return (merged * 255).astype(np.uint8)
