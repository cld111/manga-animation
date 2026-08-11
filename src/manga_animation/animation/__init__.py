"""Deterministic/kinematic animation: turns a `MotionSpec` into per-frame pixel transforms.

See `curves.py` (motion-value sampling over time) and `transforms.py` (per-`TransformKind`
pixel warps). No generative models here — see "Deterministic First" in docs/architecture.md.
"""

from __future__ import annotations

from manga_animation.animation.curves import sample_motion_value
from manga_animation.animation.transforms import (
    bbox_of_mask,
    generate_transformed_layer,
    resolve_pivot_px,
)

__all__ = [
    "bbox_of_mask",
    "generate_transformed_layer",
    "resolve_pivot_px",
    "sample_motion_value",
]
