"""Precise object/part segmentation (SAM-family models)."""

from manga_animation.segmentation.client import MaskCandidate, Sam21Client, SegmentationClient
from manga_animation.segmentation.segment import segment_object

__all__ = [
    "MaskCandidate",
    "SegmentationClient",
    "Sam21Client",
    "segment_object",
]
