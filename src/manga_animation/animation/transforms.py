"""Per-frame deterministic pixel transforms, driven by a `MotionSpec`.

Ported from and productionized over `scripts/phase2_cv_feasibility.py` (already executed for
real, locally, CPU-only — every `TransformKind` here confirmed bit-exact on static-region
preservation, per ADR 0005). The upgrade over that script: `amplitude` is now interpreted
exactly per the table in docs/animation-plan-schema.md (a real `MotionSpec`, not a placeholder
scale factor), and `pivot.reference` (`object_bbox` / `panel` / `page`) is now actually
resolved rather than assumed fixed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from manga_animation.animation.curves import sample_motion_value
from manga_animation.pipeline.types import BBoxPx, ImageArray, MaskArray
from manga_animation.schemas.animation_plan import MotionSpec, PivotSpec, TransformKind


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
        return np.array(
            [
                [1 + shear * direction.x, shear * direction.y, -pivot_px[0] * shear * direction.x],
                [shear * direction.x, 1 + shear * direction.y, -pivot_px[1] * shear * direction.y],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"{kind} is not an affine transform kind")


def _mesh_warp_frame(
    image: ImageArray, mask: MaskArray, motion: MotionSpec, value: float
) -> tuple[ImageArray, MaskArray]:
    """Cloth/hair-style ripple: smooth displacement, strongest away from the mask's

    direction-hinted anchor edge (mimics hair swaying from a fixed scalp attachment, or cloth
    hanging from a fixed pole) — see `scripts/phase2_cv_feasibility.py`'s proven version, now
    driven by the schema's real `amplitude`/`direction` fields instead of hardcoded constants.
    """
    h, w = mask.shape
    bbox = bbox_of_mask(mask)
    x0, y0, x1, y1 = bbox.as_xyxy()
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    direction = motion.direction
    dir_x = direction.x if direction is not None else 1.0
    dir_y = direction.y if direction is not None else 0.0

    # Falloff anchored at the edge the flow direction points *away from* (e.g. a downward
    # sway anchors at the top/scalp; a rightward sway anchors at the left/pole).
    if abs(dir_y) >= abs(dir_x):
        local = np.clip((map_y - y0) / max(y1 - y0, 1), 0.0, 1.0)
    else:
        local = np.clip((map_x - x0) / max(x1 - x0, 1), 0.0, 1.0)

    strength = value * motion.amplitude * max(x1 - x0, y1 - y0)
    warped_map_x = map_x + strength * dir_x * local
    warped_map_y = map_y + strength * dir_y * local

    warped_layer = cv2.remap(
        image, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.remap(
        mask, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
    )
    return warped_layer, warped_mask


def _opacity_frame(
    image: ImageArray, mask: MaskArray, motion: MotionSpec, value: float
) -> tuple[ImageArray, MaskArray]:
    alpha_scale = min(max(1.0 + value * motion.amplitude, 0.0), 1.0)
    scaled_mask = np.clip(mask.astype(np.float32) * alpha_scale, 0, 255).astype(np.uint8)
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
) -> tuple[ImageArray, MaskArray]:
    """The transformed (layer, mask) for one frame at `t_frac` (`[0, 1)` of the loop).

    `image`/`mask` are the object's ORIGINAL (untransformed) source-image-shape pixels/mask —
    every frame is generated from this same source, never from a previously-transformed frame,
    which is what makes the compositing stage's static-region guarantee structural rather than
    cumulative-error-prone (see `src/manga_animation/compositing`).
    """
    t_s = t_frac * loop_duration_s
    value = sample_motion_value(motion, t_s, loop_duration_s)

    object_bbox_px = bbox_of_mask(mask)
    kind = motion.transform_kind

    if kind == TransformKind.OPACITY:
        return _opacity_frame(image, mask, motion, value)
    if kind == TransformKind.MESH_WARP:
        return _mesh_warp_frame(image, mask, motion, value)

    pivot_px = resolve_pivot_px(motion.pivot, object_bbox_px, panel_bbox_px, page_shape)
    panel_diag_px = math.hypot(panel_bbox_px.width, panel_bbox_px.height)
    matrix = _affine_matrix(kind, value, motion, pivot_px, panel_diag_px)
    h, w = mask.shape
    warped_layer = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return warped_layer, warped_mask
