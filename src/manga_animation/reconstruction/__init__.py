"""Hidden-region reconstruction: fills the hole an object's motion reveals, and only that hole.

Owned by `cv-agent` (see `.claude/agents/cv-agent.md`'s "Hidden-region reconstruction" section)
alongside `animation`/`compositing` — the one deliberate exception to "Deterministic First" in
docs/architecture.md, since content that was never drawn can't come from a transform of
existing pixels. Candidate model: LaMa (`lama-large`, see ADR 0005) via `simple-lama-inpainting`.

ADR 0005's real finding (confirmed on a live Kaggle run): LaMa's raw output is NOT
pixel-aligned with its input (a 1778x1000 page came back 1784x1000 — internal stride padding).
This module's public function normalizes the raw output back to source geometry before
returning — never hand a raw, possibly-mis-sized inpainting result to the compositing stage.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from manga_animation.pipeline.types import ImageArray, MaskArray, ReconstructionResult


class ReconstructionClient(Protocol):
    """What `reconstruct_hidden_region` needs from an inpainting model wrapper."""

    def load(self) -> None: ...
    def inpaint(self, image: Image.Image, hole_mask: Image.Image) -> Image.Image: ...
    def unload(self) -> None: ...


class LamaClient:
    """Real `lama-large` client via `simple-lama-inpainting` — lazy-imported (not installed

    locally; remote-GPU-only per ADR 0003), matching `scripts/phase2_kaggle_benchmark.py`'s
    `LamaAdapter`.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model: object | None = None

    def load(self) -> None:
        # Import simple_lama_inpainting before any cv2/numpy use elsewhere in the process to
        # avoid a real, documented numpy/cv2 ABI conflict (ADR 0005) — callers running this in
        # a fresh process (the remote pipeline entry point) get this for free; this module
        # itself imports numpy/PIL at module load time, so this client is not safe to `load()`
        # in the same interpreter as a lot of prior cv2 activity — a known, documented limit.
        from simple_lama_inpainting import SimpleLama

        self._model = SimpleLama(device=self.device)

    def inpaint(self, image: Image.Image, hole_mask: Image.Image) -> Image.Image:
        if self._model is None:
            raise RuntimeError("LamaClient.load() must be called before inpaint()")
        result: Image.Image = self._model(image, hole_mask)  # type: ignore[operator]
        return result

    def unload(self) -> None:
        import torch

        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compute_hole_mask(original_mask: MaskArray, transformed_masks: list[MaskArray]) -> MaskArray:
    """The region covered by the object's original mask but never covered by ANY frame's

    transformed mask across the whole loop — i.e. background that is revealed at some point
    during playback and therefore needs real replacement pixels, not just the original's
    (never-drawn) content.
    """
    covered_ever = np.zeros(original_mask.shape, dtype=bool)
    for transformed in transformed_masks:
        covered_ever |= transformed > 0
    hole = (original_mask > 0) & ~covered_ever
    return (hole.astype(np.uint8) * 255)


def _normalize_to_source_geometry(raw: Image.Image, source_shape: tuple[int, int]) -> ImageArray:
    """Resize/align a raw inpainting-model output back to the source image's exact (H, W).

    Resizing (not cropping) is deliberate: the misalignment ADR 0005 found is internal stride
    padding, a near-1:1 size difference, not a meaningful crop region — resizing back to exact
    source geometry is the correct undo, matching the padding's own nature.
    """
    h, w = source_shape
    if raw.size != (w, h):
        raw = raw.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(raw.convert("RGB"), dtype=np.uint8)


def reconstruct_hidden_region(
    image: ImageArray,
    original_mask: MaskArray,
    transformed_masks: list[MaskArray],
    client: ReconstructionClient,
    *,
    object_id: str,
    model_id: str = "lama-large",
) -> ReconstructionResult | None:
    """Fill the hole `object_id`'s motion reveals across the loop, or `None` if it never

    reveals anything (e.g. a pure in-place rotation/scale whose footprint always covers the
    original region) — no reconstruction call is made in that case, per "Local Modification"
    in docs/architecture.md (don't do work outside the smallest necessary region).
    """
    hole_mask = _compute_hole_mask(original_mask, transformed_masks)
    if not np.any(hole_mask):
        return None

    client.load()
    try:
        raw_output = client.inpaint(Image.fromarray(image), Image.fromarray(hole_mask))
    finally:
        client.unload()

    filled_pixels = _normalize_to_source_geometry(raw_output, image.shape[:2])
    return ReconstructionResult(
        object_id=object_id,
        hole_mask=hole_mask,
        filled_pixels=filled_pixels,
        model_id=model_id,
    )
