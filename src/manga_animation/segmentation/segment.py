"""Turns a `GroundingResult` into a validated, pixel-accurate `SegmentationResult`.

Only `segment_object` is the public entry point — see
`.claude/agents/segmentation-agent.md` for why segmentation stays scoped to what grounding
already found (no re-detection, no re-deciding *whether* an object should move).
"""

from __future__ import annotations

import numpy as np

from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    ImageArray,
    PipelineStageError,
    SegmentationResult,
)
from manga_animation.segmentation.client import SegmentationClient

# Coverage-fraction bounds a mask must fall within to be accepted, as a fraction of the full
# source image's pixel count. Below MIN: almost certainly noise/an empty detection, not a real
# object. Above MAX: almost certainly a false-positive "select everything" mask, not a specific
# object — a real grounded object (hair, a hand, a banner) practically never covers this much of
# a full manga page. Both bounds are deliberately permissive rather than tuned per-object-class,
# since no real mask data exists yet to tune against (see ADR 0005's "no visual QA done yet").
_MIN_COVERAGE_FRACTION = 0.0001
_MAX_COVERAGE_FRACTION = 0.90


def segment_object(
    image: ImageArray, grounding: GroundingResult, client: SegmentationClient
) -> SegmentationResult:
    candidates = client.segment(image, grounding.bbox)
    if not candidates:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=grounding.object_id,
            detail=(
                f"segmentation model returned no mask candidates for box "
                f"{grounding.bbox.as_xyxy()}"
            ),
            root_cause="the segmentation model failed to propose any mask for the grounded box",
            architectural=False,
            proposed_fix=(
                "verify the grounding box is sane; consider a different segmentation candidate"
            ),
        )

    best = max(candidates, key=lambda c: c.iou_score)
    mask = best.mask
    _validate_mask(mask, object_id=grounding.object_id)
    bbox = _tight_bbox(mask)

    return SegmentationResult(
        object_id=grounding.object_id,
        mask=mask,
        bbox=bbox,
        model_id=client.model_id,
        iou_score=best.iou_score,
    )


def _validate_mask(mask: ImageArray, *, object_id: str) -> None:
    coverage = float(np.count_nonzero(mask)) / mask.size
    if coverage < _MIN_COVERAGE_FRACTION:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=object_id,
            detail=(
                f"mask coverage {coverage:.6f} is below the noise-floor threshold "
                f"({_MIN_COVERAGE_FRACTION})"
            ),
            root_cause="segmentation produced an effectively empty mask for the grounded box",
            architectural=False,
            proposed_fix="check the grounding box is correct; inspect the box crop visually",
        )
    if coverage > _MAX_COVERAGE_FRACTION:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=object_id,
            detail=(
                f"mask coverage {coverage:.4f} exceeds the full-page false-positive threshold "
                f"({_MAX_COVERAGE_FRACTION})"
            ),
            root_cause=(
                "segmentation likely selected the whole page/background rather than the "
                "specific object"
            ),
            architectural=False,
            proposed_fix="tighten the grounding box or use a more specific semantic_label prompt",
        )


def _tight_bbox(mask: ImageArray) -> BBoxPx:
    ys, xs = np.where(mask > 0)
    return BBoxPx(x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()) + 1, y1=int(ys.max()) + 1)
