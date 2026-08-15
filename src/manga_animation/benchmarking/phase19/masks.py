"""Mask conversion and coordinate transforms for the OMG-LLaVA adapter.

Official preprocessing (`xtuner` `expand2square` + the app's crop): the page is padded to a
square canvas filled with the CLIP mean color, preprocessed to 1024x1024, and every produced
mask lives on that padded square. To map a mask back to the original page we must (1) upsample
from the 1024x1024 model output to the padded-square canvas size, and (2) crop out the padding
band. This module implements exactly that inverse mapping and verifies the result against the
original page geometry.

Pure numpy/PIL -- no model imports, fully testable locally. Boxes follow the project's
half-open `(x0, y0, x1, y1)` convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

ImageArray = np.ndarray
BBox = tuple[int, int, int, int]


@dataclass
class SquarePad:
    """The square-pad geometry for an original page of size `(H, W)`."""

    canvas_size: int  # max(H, W) -- the padded square
    sx: int  # x-offset of the original page within the canvas
    sy: int  # y-offset of the original page within the canvas
    ex: int  # x end (exclusive) of the original page within the canvas
    ey: int  # y end (exclusive) of the original page within the canvas
    h: int
    w: int

    @classmethod
    def from_page_size(cls, page_size: tuple[int, int]) -> SquarePad:
        h, w = page_size
        canvas = max(h, w)
        if w == h:
            sx, sy, ex, ey = 0, 0, w, h
        elif w > h:
            sy = (w - h) // 2
            sx, ex = 0, w
            ey = sy + h
        else:
            sx = (h - w) // 2
            sy, ey = 0, h
            ex = sx + w
        return cls(canvas_size=canvas, sx=sx, sy=sy, ex=ex, ey=ey, h=h, w=w)


def expand2square_pad_color(image_mean: tuple[float, float, float]) -> tuple[int, int, int]:
    """The exact fill color the official chat tool uses: CLIP image mean (0..1) scaled to 0..255
    and truncated to int (`tuple(int(x * 255) for x in image_processor.image_mean)`)."""
    return tuple(int(x * 255) for x in image_mean)  # type: ignore[return-value]


def page_to_square_canvas(
    image: ImageArray,
    pad: SquarePad,
    fill: tuple[int, int, int] = (123, 117, 104),
) -> ImageArray:
    """Pad `image` (H, W, 3) to a `(canvas_size, canvas_size, 3)` canvas, replicating
    `xtuner`'s `expand2square` (page pasted at `(sx, sy)`)."""
    h, w = image.shape[:2]
    if h != pad.h or w != pad.w:
        raise ValueError(
            f"image {image.shape[:2]} does not match pad geometry {pad.h}x{pad.w}"
        )
    canvas = np.full((pad.canvas_size, pad.canvas_size, 3), fill, dtype=np.uint8)
    canvas[pad.sy : pad.sy + h, pad.sx : pad.sx + w] = image
    return canvas


def upsample_to_canvas(mask1024: np.ndarray, pad: SquarePad) -> np.ndarray:
    """Bilinear-upsample a model mask from 1024x1024 to the padded-square canvas size, then
    threshold at 0.5 -- exactly what the official `show_mask_pred` does (it resizes logits with
    `F.interpolate(..., mode='bilinear')` and applies `sigmoid() > 0.5`; here the sigmoid is
    applied upstream and the input is already a [0,1] probability map)."""
    if mask1024.shape != (1024, 1024):
        # accept (1, 1024, 1024) or (1024, 1024)
        if mask1024.shape == (1, 1024, 1024):
            mask1024 = mask1024[0]
        else:
            raise ValueError(f"expected a 1024x1024 mask, got {mask1024.shape}")
    pil = Image.fromarray((np.clip(mask1024, 0.0, 1.0) * 255).astype(np.uint8))
    resized = pil.resize((pad.canvas_size, pad.canvas_size), Image.Resampling.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def mask_from_canvas(padded_mask: np.ndarray, pad: SquarePad) -> np.ndarray:
    """Crop the padding band out of a padded-canvas mask, returning a mask on the ORIGINAL page
    geometry (H, W). `padded_mask` is boolean or [0,1] at `pad.canvas_size`."""
    if padded_mask.shape[:2] != (pad.canvas_size, pad.canvas_size):
        raise ValueError(
            f"padded mask {padded_mask.shape[:2]} does not match canvas "
            f"{pad.canvas_size}x{pad.canvas_size}"
        )
    return padded_mask[pad.sy : pad.sy + pad.h, pad.sx : pad.sx + pad.w]


def model_mask_to_original(mask1024: np.ndarray, page_size: tuple[int, int]) -> np.ndarray:
    """Full inverse pipeline for one mask: 1024x1024 model output -> padded canvas -> crop to
    the original page geometry. Returns a boolean (H, W) mask."""
    pad = SquarePad.from_page_size(page_size)
    canvas_mask = upsample_to_canvas(mask1024, pad)
    cropped = mask_from_canvas(canvas_mask, pad)
    return cropped > 0.5


def tight_bbox_from_mask(mask: np.ndarray) -> BBox:
    """Tight half-open bbox of a non-empty boolean mask (project convention)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("cannot derive a tight bbox from an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def verify_mask_geometry(mask: np.ndarray, page_size: tuple[int, int]) -> bool:
    """True when a mask is a valid (H, W) boolean-compatible array matching the page."""
    if mask.ndim != 2 or mask.shape != page_size:
        return False
    values = set(np.unique(mask).tolist())
    return values.issubset({0, 1, 255})
