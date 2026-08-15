"""Phase 18.2A deterministic error classification for each prediction.

Maps each target's outcome to the phase brief's minimum error categories (1-9). The
classification is mechanical and threshold-based (documented below), designed to be auditable
rather than clever; categories 5 (partially hidden object) and 6 (multiple similar objects)
cannot be decided reliably without GT context, so the classifier explicitly flags them as
`needs visual review` instead of guessing, and the visual packages are provided for that
human review.

Thresholds (documented heuristics for *diagnosis*, not calibrated production numbers):
- a bbox with IoU >= 0.75 against GT is GOOD (no error).
- IoU in [0.5, 0.75) -> category 1 (right instance, imprecise box).
- IoU < 0.5 -> the box is wrong in one of four mechanical ways, decided by area:
  - covers > 50% of the page (area_fraction) -> 7 (page/panel grab, "target outside panel");
  - area_ratio (pred/gt area) > 3 -> 3 (too large);
  - gt_coverage < 0.3 AND area_ratio < 0.6 -> 4 (too small);
  - otherwise -> 2 (a different instance/object was boxed).
- not found / conversion failure -> 8 / 9, decided before any IoU check.
"""

from __future__ import annotations

from dataclasses import dataclass

from manga_animation.benchmarking.phase17.metrics import bbox_area_ratio, bbox_gt_coverage, bbox_iou

BBox = tuple[int, int, int, int]

#: The phase brief's minimum error categories, keyed by their brief numbering.
ERROR_CATEGORY_NAMES: dict[int | str, str] = {
    0: "good",
    1: "correct_object_imprecise_bbox",
    2: "wrong_instance",
    3: "bbox_too_large",
    4: "bbox_too_small",
    5: "partially_hidden_object",
    6: "multiple_similar_objects",
    7: "target_outside_panel_or_page_grab",
    8: "not_found",
    9: "coordinate_conversion_failure",
}

#: Categories the mechanical classifier never assigns (they require human visual review).
NEEDS_VISUAL_REVIEW = (5, 6)

# Thresholds (diagnostic heuristics -- see module docstring).
_GOOD_IOU = 0.75
_IMPRECISE_IOU = 0.50
_PAGE_GRAB_AREA_FRACTION = 0.50
_TOO_LARGE_AREA_RATIO = 3.0
_TOO_SMALL_COVERAGE = 0.30
_TOO_SMALL_AREA_RATIO = 0.60


@dataclass(frozen=True, slots=True)
class Classification:
    category: int
    name: str
    note: str | None  # human-review hint, set for categories that need visual confirmation

    def as_dict(self) -> dict[str, str | int | None]:
        return {"category": self.category, "name": self.name, "note": self.note}


def _page_area_fraction(pixel_box: BBox, page_w: int, page_h: int) -> float:
    w = pixel_box[2] - pixel_box[0]
    h = pixel_box[3] - pixel_box[1]
    return (w * h) / (page_w * page_h) if page_w and page_h else 0.0


def classify(
    gt_bbox: BBox,
    pixel_box: BBox | None,
    found: bool,
    error: str | None,
    page_w: int,
    page_h: int,
) -> Classification:
    """Classify one prediction. Precedence: conversion failure (9) > not found (8) > geometry."""
    if error is not None:
        return Classification(9, ERROR_CATEGORY_NAMES[9], f"conversion detail: {error}")
    if not found or pixel_box is None:
        return Classification(8, ERROR_CATEGORY_NAMES[8], "VLM reported not found")
    if _page_area_fraction(pixel_box, page_w, page_h) > _PAGE_GRAB_AREA_FRACTION:
        return Classification(
            7,
            ERROR_CATEGORY_NAMES[7],
            "box covers >50% of the page (likely a whole-panel/whole-page grab)",
        )

    iou = bbox_iou(gt_bbox, pixel_box)
    if iou >= _GOOD_IOU:
        return Classification(0, ERROR_CATEGORY_NAMES[0], None)
    if iou >= _IMPRECISE_IOU:
        return Classification(
            1,
            ERROR_CATEGORY_NAMES[1],
            f"right instance but bbox imprecise (IoU {iou:.3f})",
        )

    ratio = bbox_area_ratio(gt_bbox, pixel_box)
    coverage = bbox_gt_coverage(gt_bbox, pixel_box)
    if ratio > _TOO_LARGE_AREA_RATIO:
        return Classification(
            3, ERROR_CATEGORY_NAMES[3], f"area ratio {ratio:.2f}x the GT box"
        )
    if coverage < _TOO_SMALL_COVERAGE and ratio < _TOO_SMALL_AREA_RATIO:
        return Classification(4, ERROR_CATEGORY_NAMES[4], f"covers {coverage:.2f} of the GT box")
    return Classification(
        2,
        ERROR_CATEGORY_NAMES[2],
        f"IoU {iou:.3f} -- a different instance/region was boxed (area ratio {ratio:.2f}x)",
    )
