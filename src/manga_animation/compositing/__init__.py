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

from manga_animation.animation import bbox_of_mask
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

    Phase 6 local-rendering hardening: both the hole substitution and the alpha blend are
    restricted to their own relevant mask's bbox (`reconstruction.hole_mask`'s bbox and
    `layer_mask`'s bbox respectively) rather than computed over the whole page — outside a
    mask's own bbox that mask is `0` everywhere by construction, so the skipped region's
    result is trivially identical to `plate` either way (alpha=0 blends to exactly `plate`;
    "revealed" is exactly `False` where `hole_mask` is `0`). See `animation/transforms.py`'s
    docstring for the same reasoning applied to the transform stage.
    """
    plate = original.copy()

    if reconstruction is not None and np.any(reconstruction.hole_mask):
        hx0, hy0, hx1, hy1 = bbox_of_mask(reconstruction.hole_mask).as_xyxy()
        revealed_roi = (layer_mask[hy0:hy1, hx0:hx1] == 0) & (
            reconstruction.hole_mask[hy0:hy1, hx0:hx1] != 0
        )
        plate[hy0:hy1, hx0:hx1][revealed_roi] = reconstruction.filled_pixels[hy0:hy1, hx0:hx1][
            revealed_roi
        ]

    if not np.any(layer_mask):
        return plate

    x0, y0, x1, y1 = bbox_of_mask(layer_mask).as_xyxy()
    alpha = (layer_mask[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
    blended = layer[y0:y1, x0:x1].astype(np.float32) * alpha + plate[y0:y1, x0:x1].astype(
        np.float32
    ) * (1.0 - alpha)
    plate[y0:y1, x0:x1] = blended.astype(np.uint8)
    return plate


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

    Phase 6 local-rendering hardening: the reconstruction step is restricted to each object's
    own `hole_mask` bbox (identical reasoning to `composite_frame`'s localization — outside
    that bbox `hole_mask` is `0` everywhere, so "revealed" is trivially `False` there
    regardless of `other_covered`). The alpha-blend loop keeps its *exact* original
    semantics — every layer's contribution is accumulated in float32 and rounded to `uint8`
    only once, after the last (topmost) layer, never re-rounded between layers, since two
    overlapping partial-alpha layers would otherwise blend against a different (already
    uint8-rounded) value than the unrestricted implementation produces — but the float32
    accumulator itself is only allocated for the union of every active layer's own mask bbox
    this frame, not the whole page; outside that union, every layer's alpha is `0`, so the
    result is `plate` unchanged there either way.
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
            if not np.any(recon.hole_mask):
                continue
            own_mask = own_frame[1]

            hx0, hy0, hx1, hy1 = bbox_of_mask(recon.hole_mask).as_xyxy()
            other_covered = np.zeros((hy1 - hy0, hx1 - hx0), dtype=bool)
            for other_id, (_, other_mask) in current_frames.items():
                if other_id == object_id:
                    continue
                other_covered |= other_mask[hy0:hy1, hx0:hx1] > 0

            revealed_this_frame = (
                (own_mask[hy0:hy1, hx0:hx1] == 0)
                & (recon.hole_mask[hy0:hy1, hx0:hx1] != 0)
                & ~other_covered
            )
            plate[hy0:hy1, hx0:hx1][revealed_this_frame] = recon.filled_pixels[hy0:hy1, hx0:hx1][
                revealed_this_frame
            ]

    active_bboxes = [
        bbox_of_mask(layer_mask) for _, layer_mask in current_frames.values() if np.any(layer_mask)
    ]
    if not active_bboxes:
        return plate

    ux0 = min(b.x0 for b in active_bboxes)
    uy0 = min(b.y0 for b in active_bboxes)
    ux1 = max(b.x1 for b in active_bboxes)
    uy1 = max(b.y1 for b in active_bboxes)

    frame_roi = plate[uy0:uy1, ux0:ux1].astype(np.float32)
    for layer in ordered:
        layer_image, layer_mask = current_frames[layer.object_id]
        alpha = (layer_mask[uy0:uy1, ux0:ux1].astype(np.float32) / 255.0)[..., None]
        frame_roi = layer_image[uy0:uy1, ux0:ux1].astype(np.float32) * alpha + frame_roi * (
            1.0 - alpha
        )

    plate[uy0:uy1, ux0:ux1] = frame_roi.astype(np.uint8)
    return plate
