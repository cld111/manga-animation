"""Alpha compositing of an animated layer back onto the untouched original artwork.

The single hard invariant this module exists to guarantee (see "Original Image Is the Source
of Truth" in docs/architecture.md, and `qa-agent`'s checks): for every output frame, every
pixel with zero contribution from the animated layer must equal the source image exactly.
"Outside the mask" means the transformed mask's actual `== 0` boundary, not an arbitrary
threshold — an earlier ad hoc check in `scripts/phase2_cv_feasibility.py` used `mask < 8` as a
proxy and reported false failures purely from interpolated mask-edge values in the 1-7 range
(see ADR 0005) — this module does not repeat that mistake.
"""

from __future__ import annotations

import numpy as np

from manga_animation.pipeline.types import ImageArray, MaskArray, ReconstructionResult


def composite_frame(
    original: ImageArray,
    layer: ImageArray,
    layer_mask: MaskArray,
    *,
    reconstruction: ReconstructionResult | None = None,
) -> ImageArray:
    """Alpha-blend `layer` (via `layer_mask`) over a FRESH copy of `original`.

    Building each frame from a fresh copy of `original` (never patching a running buffer) is
    what makes "untouched outside the mask" a structural guarantee rather than a best-effort
    one that could accumulate error frame-to-frame.

    If `reconstruction` is given, its `filled_pixels` replace `original`'s pixels wherever
    *this specific frame* reveals background the object used to cover but no longer does
    (`layer_mask == 0` AND `reconstruction.hole_mask != 0`) — `hole_mask` is the union of
    every frame's revealed region across the whole loop, so at any single frame only part of
    it may actually need patching; intersecting with this frame's `layer_mask` picks exactly
    that part. That substitution happens on the background plate BEFORE the layer is blended
    on top, so it composes correctly with the alpha blend below.
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
