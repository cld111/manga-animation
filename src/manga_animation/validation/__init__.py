"""Explicit target validation between grounding and segmentation (Phase 3.2), extended with

transformation-aware geometric validation (Phase 3.3.1, see
docs/decisions/0008-transform-aware-target-validation.md).
"""

from manga_animation.validation.transform_geometry import (
    TransformGeometryProfile,
    check_transform_geometry,
)
from manga_animation.validation.validate import validate_target

__all__ = [
    "TransformGeometryProfile",
    "check_transform_geometry",
    "validate_target",
]
