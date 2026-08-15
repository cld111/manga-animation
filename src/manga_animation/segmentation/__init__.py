"""Precise object/part segmentation (SAM-family models)."""

from manga_animation.segmentation.client import MaskCandidate, Sam21Client, SegmentationClient
from manga_animation.segmentation.segment import _MAX_EFFECT_MASK_DENSITY, segment_object

__all__ = [
    "MaskCandidate",
    "SegmentationClient",
    "Sam21Client",
    "_MAX_EFFECT_MASK_DENSITY",
    "segment_object",
]
