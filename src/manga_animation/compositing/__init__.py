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

from manga_animation.pipeline.types import ImageArray, Layer, MaskArray, ReconstructionResult


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


def composite_frame_stack(
    original: ImageArray,
    layers: list[Layer],
    frame_index: int,
    *,
    reconstructions: dict[str, ReconstructionResult] | None = None,
) -> ImageArray:
    """`composite_frame`, generalized to N simultaneously-animated `Layer`s (Phase 4; see

    `pipeline.types.Layer` and ADR 0010) — alpha-composites every layer's frame `frame_index`
    onto a FRESH copy of `original`, in ascending `(z_order, object_id)` order (lower z_order
    further back; ties broken lexicographically by `object_id` so ordering never depends on
    input-list order or dict iteration order). `layers=[]` returns a copy of `original`
    unchanged.

    Reconstruction hole-filling is applied to the background plate BEFORE any layer is
    blended on top (matching `composite_frame`), but with one extra condition beyond the
    single-object case: object X's reconstructed hole pixels are only substituted where (a)
    X's own current-frame mask is 0 there, (b) `reconstructions[X].hole_mask` says X's
    original position covered it, AND (c) no *other* layer's current-frame mask covers that
    pixel either. (c) is the actual generalization this function adds — if a different
    animated object's layer is currently sitting on top of what would be X's revealed
    background, that other layer already paints the correct pixels there this frame, and X's
    reconstruction fill must not overwrite or race with it. Layers with no entry in
    `reconstructions` are simply never hole-filled (mirrors `reconstruct_hidden_region`
    returning `None` when an object's motion never reveals anything).

    With exactly one layer, this produces bit-identical output to `composite_frame` called
    with that layer's (image, mask) and matching reconstruction — the "other layers" union
    used in condition (c) is empty by construction, collapsing back to `composite_frame`'s
    plain `(layer_mask == 0) & (hole_mask != 0)` check.
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
                # No corresponding layer this frame -- shouldn't happen given the orchestrator
                # only reconstructs objects it also animates, but skip rather than guess.
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
