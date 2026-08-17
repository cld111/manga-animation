"""Wan2.2 TI2V-5B generative animation engine (replacing AnimateAnything, ADR 0024).

Wan2.2-TI2V-5B is a 5B-parameter Text-Image-to-Video diffusion model that generates
720P@24fps video from an input image + text prompt. It replaces the older AnimateAnything
512 v1.02 engine with a modern, actively maintained, higher-quality model.

Dependency isolation: Wan2.2 requires diffusers main branch (not the PyPI release) and
specific torch/transformers versions. Real inference runs as a SUBPROCESS in a dedicated
environment: `Wan2Client` serializes a spec, invokes the worker interpreter, and reads the
generated frames back. Local work stays code/tests/config; the heavy model and its
pinned environment are remote-GPU concerns (ADR 0003, ADR 0024).

Key differences from AnimateAnything:
- I2V mode: generates video conditioned on input image + text prompt (no mask required)
- 720P@24fps output (121 frames, 5s) vs 512x512@8fps (16 frames, 2s)
- ~23 GB peak VRAM on 4090 (FSDP + Ulysses for multi-GPU)
- Uses WanPipeline from diffusers, not vendored code
"""

from __future__ import annotations

from manga_animation.wan2.client import Wan2Client
from manga_animation.wan2.mask import merge_motion_masks
from manga_animation.wan2.prompt import build_animation_prompt, motion_phrase
from manga_animation.wan2.spec import Wan2Spec

__all__ = [
    "Wan2Client",
    "Wan2Spec",
    "build_animation_prompt",
    "merge_motion_masks",
    "motion_phrase",
]
