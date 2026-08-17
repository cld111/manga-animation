"""Merging accepted SAM masks into a single uint8 motion mask.

Wan2.2 TI2V-5B does NOT use a motion mask for generation (it generates from image+prompt).
However, the pipeline still produces and persists SAM masks for provenance and debugging.
This module merges the accepted per-object SAM masks into a single uint8 0/255 array, matching
the contract from the AnimateAnything era.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from manga_animation.pipeline.types import MaskArray


def merge_motion_masks(masks: Sequence[MaskArray]) -> MaskArray:
    """Union of full-source-image-shaped SAM masks into one uint8 0/255 mask.

    Raises `ValueError` on empty input (fail closed) or on a shape mismatch with the first
    mask (the caller is responsible for passing masks of one image). The result is the
    elementwise OR of `mask > 0`, scaled to 0/255.
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
