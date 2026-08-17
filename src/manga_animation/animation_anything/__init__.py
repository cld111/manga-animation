"""Generative animation engine: Alibaba AnimateAnything (mask-conditioned video diffusion).

The AnimateAnything client turns (original panel image, merged SAM motion mask, text prompt
built from the Qwen object descriptions) into a short generated video. It is THE animation
engine for this integration -- the deterministic OpenCV transforms remain in
`manga_animation.animation` for the legacy path, but a run that selects
`model_variants.animation = animate-anything-512-v1.02` drives its animation exclusively
through this package (see docs/decisions/0024-animate-anything-animation-engine.md).

Dependency isolation: AnimateAnything was released against a very old stack (diffusers
==0.24.0, transformers==4.36.2, torch==2.0.0 -- see its requirements.txt), which conflicts
with this project's `ml` extra (transformers>=5.0, torch>=2.3). Real inference therefore runs
as a SUBPROCESS in a dedicated environment on the remote worker: `AnimateAnythingClient`
serializes a spec (image, mask, prompt, hyper-parameters), invokes `worker.py` with the
isolated interpreter, and reads the generated frames back. Local work stays code/tests/config;
the heavy model and its pinned environment are remote-GPU concerns (ADR 0003).

The vendored model code under `vendored/` is the upstream AnimateAnything repository's
inference subset (Apache-2.0, see vendored/LICENSE and vendored/README) -- imported only by
`worker.py`, never by the main pipeline process.
"""

from __future__ import annotations

from manga_animation.animation_anything.client import AnimateAnythingClient
from manga_animation.animation_anything.mask import merge_motion_masks
from manga_animation.animation_anything.prompt import build_animation_prompt, motion_phrase
from manga_animation.animation_anything.spec import AnimateAnythingSpec

__all__ = [
    "AnimateAnythingClient",
    "AnimateAnythingSpec",
    "build_animation_prompt",
    "merge_motion_masks",
    "motion_phrase",
]
