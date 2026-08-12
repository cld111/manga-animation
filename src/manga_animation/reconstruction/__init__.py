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

from manga_animation.animation import bbox_of_mask
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
    """The region covered by the object's original mask that is left uncovered by AT LEAST ONE

    frame's transformed mask — i.e. every pixel that will show through to the background plate
    at some point during playback and therefore needs a real replacement value available,
    scoped per-frame by `compositing.composite_frame`/`composite_frame_stack`.

    Phase 4 reconstruction-hardening fix: the previous formula was `original & ~UNION(frames)`
    ("never covered by any frame") instead of the correct `original & ~INTERSECTION(frames)`
    ("not covered by every frame"), equivalently `UNION over frames of (original &
    ~transformed[i])` — the form used below. The old formula is mathematically guaranteed to
    return an empty (or near-empty) hole whenever ANY single sampled frame fully reproduces the
    original mask, which in practice is *always* true: frame index 0 (`t_frac=0`) is every
    `loop_mode="cycle"` motion's rest pose (the schema's seamless-loop convention with the
    default `phase=0`), so `transformed_masks[0]` is, up to interpolation rounding, bit-identical
    to `original_mask` — silently making the old computation vacuous for this project's actual
    motion model, confirmed empirically across every `TransformKind` except `OPACITY` (which
    legitimately never moves the mask and correctly has no hole either way). Compositing blends
    each frame from its OWN transformed mask independently — "the object eventually comes back
    to cover this pixel later in the loop" does not help the specific frame where it doesn't.

    Phase 6 local-rendering hardening: `original_mask > 0` can only ever be `True` within
    `original_mask`'s own tight bbox, so every OR-accumulation step outside that bbox is
    guaranteed a no-op — the loop over `transformed_masks` (one boolean full-page comparison
    per frame) is restricted to that bbox slice instead of the whole page, which is what makes
    this scale with the animated region rather than `frame_count * page_pixels`. An empty mask
    (no object ever drawn) has no bbox to compute and trivially has no hole either way.
    """
    hole = np.zeros(original_mask.shape, dtype=bool)
    if not np.any(original_mask):
        return hole.astype(np.uint8) * 255

    x0, y0, x1, y1 = bbox_of_mask(original_mask).as_xyxy()
    original_local = original_mask[y0:y1, x0:x1] > 0
    hole_local = np.zeros(original_local.shape, dtype=bool)
    for transformed in transformed_masks:
        hole_local |= original_local & (transformed[y0:y1, x0:x1] == 0)
    hole[y0:y1, x0:x1] = hole_local
    return hole.astype(np.uint8) * 255


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
