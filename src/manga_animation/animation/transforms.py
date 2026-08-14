"""Per-frame deterministic pixel transforms, driven by a `MotionSpec`.

Ported from and productionized over `scripts/phase2_cv_feasibility.py` (already executed for
real, locally, CPU-only — every `TransformKind` here confirmed bit-exact on static-region
preservation, per ADR 0005). The upgrade over that script: `amplitude` is now interpreted
exactly per the table in docs/animation-plan-schema.md (a real `MotionSpec`, not a placeholder
scale factor), and `pivot.reference` (`object_bbox` / `panel` / `page`) is now actually
resolved rather than assumed fixed.

Phase 6 local-rendering hardening: every transform below computes its actual pixel warp only
over the smallest page-space region that could possibly end up nonzero (the object's own bbox,
or — for the affine kinds — that bbox's transformed footprint), then places that small result
into a full-page-shaped, zero-initialized array before returning. The external contract
(`generate_transformed_layer` returns full-page `(ImageArray, MaskArray)`, exactly as before)
is unchanged; only the internal cost of computing it now scales with the animated region rather
than the page, per "Local Modification" in docs/architecture.md. This is safe *because*
downstream compositing (`src/manga_animation/compositing`) only ever reads a layer's pixels
where its own transformed mask is nonzero (`alpha > 0`) — whatever a full-page warp would have
put in the untouched-here region is compositing-irrelevant, so a zero-filled placeholder there
is observably identical.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from manga_animation.animation.curves import sample_motion_value
from manga_animation.pipeline.types import BBoxPx, ImageArray, MaskArray
from manga_animation.schemas.animation_plan import MotionSpec, PivotSpec, TransformKind

# Padding (px) added on top of the interpolation-kernel/rounding margin below, absorbing our
# own float-AABB rounding and INTER_LINEAR's ~1px kernel radius at the region boundary.
_ROI_SAFETY_MARGIN_PX = 2


def bbox_of_mask(mask: MaskArray) -> BBoxPx:
    """The tight pixel bbox of a mask's nonzero extent."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("mask is empty — cannot derive a bbox")
    return BBoxPx(x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()) + 1, y1=int(ys.max()) + 1)


def resolve_pivot_px(
    pivot: PivotSpec,
    object_bbox_px: BBoxPx,
    panel_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
) -> tuple[float, float]:
    """Resolve a normalized `PivotSpec` to actual pixel coordinates, per its `reference`."""
    if pivot.reference == "object_bbox":
        ref = object_bbox_px
    elif pivot.reference == "panel":
        ref = panel_bbox_px
    else:  # "page"
        h, w = page_shape
        ref = BBoxPx(x0=0, y0=0, x1=w, y1=h)
    x0, y0, x1, y1 = ref.as_xyxy()
    return (x0 + pivot.x * (x1 - x0), y0 + pivot.y * (y1 - y0))


def _affine_matrix(
    kind: TransformKind,
    value: float,
    motion: MotionSpec,
    pivot_px: tuple[float, float],
    panel_diag_px: float,
) -> np.ndarray:
    amplitude = motion.amplitude
    direction = motion.direction  # guaranteed non-None + unit for translate/shear by the schema

    if kind == TransformKind.TRANSLATE:
        assert direction is not None
        dx = value * amplitude * panel_diag_px * direction.x
        dy = value * amplitude * panel_diag_px * direction.y
        return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    if kind == TransformKind.ROTATE:
        angle_deg = value * amplitude
        return cv2.getRotationMatrix2D(pivot_px, angle_deg, 1.0)
    if kind == TransformKind.SCALE:
        scale = 1.0 + value * amplitude
        return cv2.getRotationMatrix2D(pivot_px, 0.0, scale)
    if kind == TransformKind.SHEAR:
        assert direction is not None
        shear = value * amplitude
        # Shear along the direction vector's axis, about the pivot — a straightforward
        # generalization of the phase2 script's fixed-horizontal-shear special case.
        linear = np.array(
            [
                [1 + shear * direction.x, shear * direction.y],
                [shear * direction.x, 1 + shear * direction.y],
            ],
            dtype=np.float32,
        )
        # Build the translation as p - A*p. Writing only one diagonal term here
        # does not keep a non-origin pivot fixed for a general direction.
        pivot = np.asarray(pivot_px, dtype=np.float32)
        translation = pivot - linear @ pivot
        return np.array(
            [
                [linear[0, 0], linear[0, 1], translation[0]],
                [linear[1, 0], linear[1, 1], translation[1]],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"{kind} is not an affine transform kind")


def _affine_operator_norm(matrix: np.ndarray) -> float:
    """Spectral norm of `matrix`'s 2x2 linear part: the largest factor by which this specific
    frame's affine transform can stretch a distance. Used to size the local-region margin so
    it scales with whatever the matrix actually does this frame (`amplitude`/`shear` have no
    schema-enforced upper bound) rather than assuming "modest" motion.
    """
    return float(np.linalg.norm(matrix[:, :2].astype(np.float64), ord=2))


def _affine_dest_roi(
    object_bbox_px: BBoxPx, matrix: np.ndarray, page_shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """The page-space region this frame's affine `matrix` could possibly leave a nonzero
    transformed mask in, given the mask is contained in `object_bbox_px`.

    Affine maps send a rectangle to a parallelogram and preserve convex containment, so the
    transformed mask is contained in the convex hull of the bbox's 4 transformed corners, which
    is in turn contained in that hull's AABB — computing the AABB of the transformed corners is
    therefore a valid (if not perfectly tight) bound, expanded by a margin covering our own
    float-to-int rounding plus INTER_LINEAR's ~1px kernel radius scaled by how much this matrix
    stretches distances, then clipped to the page (matching the old full-page `warpAffine`
    call's own `dsize=(w,h)` cropping). Returns `None` if the clipped region is empty — the
    transform moved the object's whole footprint off-page this frame.
    """
    h, w = page_shape
    x0, y0, x1, y1 = object_bbox_px.as_xyxy()
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
    transformed = corners @ matrix.T.astype(np.float64)
    margin = math.ceil(_affine_operator_norm(matrix)) + _ROI_SAFETY_MARGIN_PX

    rx0 = max(0, math.floor(transformed[:, 0].min()) - margin)
    ry0 = max(0, math.floor(transformed[:, 1].min()) - margin)
    rx1 = min(w, math.ceil(transformed[:, 0].max()) + margin)
    ry1 = min(h, math.ceil(transformed[:, 1].max()) + margin)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    return (int(rx0), int(ry0), int(rx1), int(ry1))


def _mesh_warp_frame(
    image: ImageArray,
    mask: MaskArray,
    motion: MotionSpec,
    value: float,
    object_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
) -> tuple[ImageArray, MaskArray]:
    """Cloth/hair-style ripple: smooth displacement, strongest away from the mask's

    direction-hinted anchor edge (mimics hair swaying from a fixed scalp attachment, or cloth
    hanging from a fixed pole) — see `scripts/phase2_cv_feasibility.py`'s proven version, now
    driven by the schema's real `amplitude`/`direction` fields instead of hardcoded constants.

    Phase 6: `map_x`/`map_y` are built directly in absolute page coordinates for only the local
    ROI (instead of the whole page), so the per-pixel `warped_map_x = map_x + strength*dir_x*
    local` formula below is evaluated at a subset of the exact same positions it always was —
    bit-exact vs. the old full-page computation by construction, not merely by testing. Every
    output position whose sampled source could fall inside `object_bbox_px` satisfies
    `position ∈ [object_bbox_px expanded by |strength|]` (the per-pixel displacement is bounded
    by `|strength| * max(|dir_x|, |dir_y|)` in each axis, since `local` ∈ [0, 1] and `direction`
    is not schema-normalized for mesh_warp) — the ROI below uses that bound, not the tighter
    (but direction-normalized-only) `|strength|` alone.

    Phase 10 fix (docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md):
    when `motion.direction` is unset (the real, common case — `analysis/plan_builder.py`'s
    `_MOTION_HEURISTICS` flag/cloth/cape entry never sets it), the anchor/flow axis now follows
    the object's own bbox elongation instead of a hardcoded `(1.0, 0.0)`. A real Phase 9 defect
    (`realworld_villainess_ending_scuffle`, a tall `cloth` SECONDARY object) was traced to
    exactly the old hardcoded default: since `local` below only varies along whichever axis
    `dir_x`/`dir_y` selects, a horizontal-only default applied the SAME horizontal displacement
    to every row of a tall object regardless of its own height — a rigid sideways shear
    uncorrelated with the object's actual vertical extent, producing a hard, page-aligned
    vertical discontinuity once `strength` (which scales with the object's own longest bbox
    dimension) grew large. Reproduced deterministically against this exact function with a real
    Phase 9 source image and a constructed tall mask (Phase 9's own GPU session and real
    SAM mask are gone — no live re-verification of the *exact* original instance was possible,
    only the mechanism): a tall mask's right-edge column shifted by the identical amount at
    every sampled row from `y0` to `y1` under the old default. The fix ties the fallback to the
    mask's own shape — taller-than-wide anchors at the top and sways downward (matching this
    heuristic's own `pivot=(0.5, 0.0, object_bbox)` convention, "hangs from a fixed point");
    wider-than-tall keeps the previous left-anchored horizontal sway unchanged (the real,
    already-validated flag/banner case is not touched, since `direction=None` widely-shaped
    masks take the same branch as before).
    """
    h, w = page_shape
    x0, y0, x1, y1 = object_bbox_px.as_xyxy()

    direction = motion.direction
    if direction is not None:
        dir_x, dir_y = direction.x, direction.y
    elif (y1 - y0) >= (x1 - x0):
        # No explicit flow hint and the mask is taller than (or as tall as) it is wide: default
        # to a downward sway anchored at the top, not the old fixed (1.0, 0.0) — see this
        # function's own docstring for the real defect this replaces.
        dir_x, dir_y = 0.0, 1.0
    else:
        dir_x, dir_y = 1.0, 0.0

    strength = value * motion.amplitude * max(x1 - x0, y1 - y0)
    margin = math.ceil(abs(strength) * max(abs(dir_x), abs(dir_y))) + _ROI_SAFETY_MARGIN_PX
    rx0 = max(0, x0 - margin)
    ry0 = max(0, y0 - margin)
    rx1 = min(w, x1 + margin)
    ry1 = min(h, y1 + margin)

    warped_layer = np.zeros_like(image)
    warped_mask = np.zeros_like(mask)

    map_x, map_y = np.meshgrid(
        np.arange(rx0, rx1, dtype=np.float32), np.arange(ry0, ry1, dtype=np.float32)
    )

    # Falloff anchored at the edge the flow direction points *away from* (e.g. a downward
    # sway anchors at the top/scalp; a rightward sway anchors at the left/pole).
    if abs(dir_y) >= abs(dir_x):
        local = np.clip((map_y - y0) / max(y1 - y0, 1), 0.0, 1.0)
    else:
        local = np.clip((map_x - x0) / max(x1 - x0, 1), 0.0, 1.0)

    warped_map_x = map_x + strength * dir_x * local
    warped_map_y = map_y + strength * dir_y * local

    warped_layer[ry0:ry1, rx0:rx1] = cv2.remap(
        image, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask[ry0:ry1, rx0:rx1] = cv2.remap(
        mask, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
    )
    return warped_layer, warped_mask


def _radial_expand_frame(
    image: ImageArray,
    mask: MaskArray,
    motion: MotionSpec,
    value: float,
    object_bbox_px: BBoxPx,
    pivot_px: tuple[float, float],
    page_shape: tuple[int, int],
) -> tuple[ImageArray, MaskArray]:
    """Impact/energy/glow pulse: a spatially-varying radial displacement about `pivot_px`.

    Unlike uniform `SCALE` (every pixel moves by the same factor, so the whole region reads
    as one rigid block), this applies a displacement whose magnitude grows with distance
    from the pivot — the center stays effectively fixed while the rim breathes outward on
    `value > 0` and inward on `value < 0`. That is the natural motion model for drawn
    impact bursts, radiating focus lines, energy fields, and glow-like effects (see the
    Drawn Effect Track in the phase brief): the effect's visual origin is anchored and the
    radiating artwork around it pulses.

    `amplitude` means the peak rim displacement as a fraction of the object bbox's longest
    side. The per-pixel displacement at radius `r` is
    `value * amplitude * max_dim * (r / r_max)^1.5`, where `r_max` is the pivot's distance
    to the farthest bbox corner — a smooth falloff that keeps the center quiet and
    concentrates motion at the rim (the exponent makes center-region motion negligible
    rather than linear, which is what reads as "anchored burst" instead of "zooming").

    Output position `p` samples source `p - disp`, so on the positive half of the cycle the
    rim pulls content from slightly nearer the center (visual expansion) and on the negative
    half it pushes content outward (visual contraction); at `value = 0` the map is identity.
    The ROI margin is `|strength|` because that bounds how far any output pixel's sampled
    source can fall from the object bbox (same bounding argument as `_mesh_warp_frame`).
    """
    h, w = page_shape
    px, py = pivot_px
    x0, y0, x1, y1 = object_bbox_px.as_xyxy()
    max_dim = max(x1 - x0, y1 - y0)
    strength = value * motion.amplitude * max_dim
    margin = math.ceil(abs(strength)) + _ROI_SAFETY_MARGIN_PX

    rx0 = max(0, x0 - margin)
    ry0 = max(0, y0 - margin)
    rx1 = min(w, x1 + margin)
    ry1 = min(h, y1 + margin)
    if rx1 <= rx0 or ry1 <= ry0:
        return np.zeros_like(image), np.zeros_like(mask)

    r_max = math.hypot(max(abs(x0 - px), abs(x1 - px)), max(abs(y0 - py), abs(y1 - py)))
    if r_max <= 1e-6:
        # Degenerate: the whole object sits on the pivot. Nothing meaningful to displace —
        # return the identity (rest) frame rather than dividing by ~zero.
        return image.copy(), mask.copy()

    grid_y, grid_x = np.mgrid[ry0:ry1, rx0:rx1].astype(np.float32)
    r = np.hypot(grid_x - px, grid_y - py)
    falloff = np.clip(r / r_max, 0.0, 1.0) ** 1.5
    disp = strength * falloff
    safe_r = np.where(r > 1e-6, r, 1.0)
    ux = np.where(r > 1e-6, (grid_x - px) / safe_r, 0.0)
    uy = np.where(r > 1e-6, (grid_y - py) / safe_r, 0.0)

    map_x = grid_x - disp * ux
    map_y = grid_y - disp * uy

    warped_layer = np.zeros_like(image)
    warped_mask = np.zeros_like(mask)
    warped_layer[ry0:ry1, rx0:rx1] = cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask[ry0:ry1, rx0:rx1] = cv2.remap(
        mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
    )
    return warped_layer, warped_mask


def _opacity_frame(
    image: ImageArray, mask: MaskArray, motion: MotionSpec, value: float, object_bbox_px: BBoxPx
) -> tuple[ImageArray, MaskArray]:
    alpha_scale = min(max(1.0 + value * motion.amplitude, 0.0), 1.0)
    x0, y0, x1, y1 = object_bbox_px.as_xyxy()
    scaled_mask = np.zeros_like(mask)
    # Opacity never moves pixels, so the only region that can differ from an all-zero mask is
    # `mask`'s own bbox — everywhere else `mask` (and therefore the scaled result) is already 0.
    scaled_mask[y0:y1, x0:x1] = np.clip(
        mask[y0:y1, x0:x1].astype(np.float32) * alpha_scale, 0, 255
    ).astype(np.uint8)
    return image, scaled_mask


def generate_transformed_layer(
    image: ImageArray,
    mask: MaskArray,
    motion: MotionSpec,
    panel_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
    t_frac: float,
    *,
    loop_duration_s: float,
    object_bbox_px: BBoxPx | None = None,
) -> tuple[ImageArray, MaskArray]:
    """The transformed (layer, mask) for one frame at `t_frac` (`[0, 1)` of the loop).

    `image`/`mask` are the object's ORIGINAL (untransformed) source-image-shape pixels/mask —
    every frame is generated from this same source, never from a previously-transformed frame,
    which is what makes the compositing stage's static-region guarantee structural rather than
    cumulative-error-prone (see `src/manga_animation/compositing`).

    `object_bbox_px`, if given, is used as `mask`'s tight bbox directly instead of recomputing it
    via `bbox_of_mask(mask)`. This is a per-frame-call cost, not a per-object one: a caller (e.g.
    the orchestrator's per-frame animation loop) that invokes this function once per frame for
    the SAME original `mask` would otherwise pay a full-page `np.where` scan on every single
    frame for a bbox that never changes across those calls, even though it was already computed
    once, correctly, by `segmentation-agent` (`SegmentationResult.bbox`,
    `src/manga_animation/segmentation/segment.py::_tight_bbox`, the same tight-bbox algorithm as
    `bbox_of_mask` — the two are guaranteed to agree for `mask == SegmentationResult.mask`). The
    caller is responsible for only passing a bbox that actually matches `mask`'s own tight
    extent; this function does not re-validate it (see the docstring's caller-responsibility note
    below and the accompanying test for why). Omit it (the default) for the old, self-contained
    behavior — still exactly correct, just recomputed every call.
    """
    t_s = t_frac * loop_duration_s
    value = sample_motion_value(motion, t_s, loop_duration_s)

    if object_bbox_px is None:
        object_bbox_px = bbox_of_mask(mask)
    kind = motion.transform_kind

    if kind == TransformKind.OPACITY:
        return _opacity_frame(image, mask, motion, value, object_bbox_px)
    if kind == TransformKind.MESH_WARP:
        return _mesh_warp_frame(image, mask, motion, value, object_bbox_px, page_shape)

    pivot_px = resolve_pivot_px(motion.pivot, object_bbox_px, panel_bbox_px, page_shape)
    if kind == TransformKind.RADIAL_EXPAND:
        return _radial_expand_frame(
            image, mask, motion, value, object_bbox_px, pivot_px, page_shape
        )
    panel_diag_px = math.hypot(panel_bbox_px.width, panel_bbox_px.height)
    matrix = _affine_matrix(kind, value, motion, pivot_px, panel_diag_px)

    warped_layer = np.zeros_like(image)
    warped_mask = np.zeros_like(mask)

    roi = _affine_dest_roi(object_bbox_px, matrix, page_shape)
    if roi is None:
        return warped_layer, warped_mask
    rx0, ry0, rx1, ry1 = roi

    # `matrix` maps absolute page coordinates; shifting only the translation column by the ROI's
    # own origin makes it map ROI-local output coordinates to the *same* absolute source
    # positions `matrix` itself would have (see this module's docstring / ADR-level Phase 6
    # write-up for the derivation) — `image`/`mask` are passed in full and unshifted, so those
    # absolute source positions are sampled from their real page location, not a cropped one.
    matrix_roi = matrix.copy()
    matrix_roi[:, 2] -= (rx0, ry0)

    roi_size = (rx1 - rx0, ry1 - ry0)
    warped_layer[ry0:ry1, rx0:rx1] = cv2.warpAffine(
        image, matrix_roi, roi_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask[ry0:ry1, rx0:rx1] = cv2.warpAffine(
        mask, matrix_roi, roi_size, flags=cv2.INTER_LINEAR, borderValue=0
    )
    return warped_layer, warped_mask
