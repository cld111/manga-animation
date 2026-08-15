"""Turns a `GroundingResult` into a validated, pixel-accurate `SegmentationResult`.

Only `segment_object` is the public entry point — see
`.claude/agents/segmentation-agent.md` for why segmentation stays scoped to what grounding
already found (no re-detection, no re-deciding *whether* an object should move).
"""

from __future__ import annotations

import numpy as np

from manga_animation.pipeline.types import (
    MAX_OBJECT_COVERAGE_FRACTION,
    MIN_OBJECT_COVERAGE_FRACTION,
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
# Shared with validation/validate.py's pre-segmentation bbox check — see pipeline/types.py.
_MIN_COVERAGE_FRACTION = MIN_OBJECT_COVERAGE_FRACTION
_MAX_COVERAGE_FRACTION = MAX_OBJECT_COVERAGE_FRACTION

# Phase 8.3 (docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md, "Defect B" -- the real
# "vertical seam" found in `phase3_action_page`): a real SAM 2.1 mask, downloaded from a live
# Kaggle run and independently re-verified, was found to include a large, roughly-rectangular
# strip of adjacent wall/panel background alongside the true hair silhouette -- its own tight
# bbox's LEFT edge was mask-covered for 45.5% of its height, vs. 2.2%-20.2% for five other real
# masks from the same investigation (a raised sword, two eyes, and a second real hair mask) that
# showed no visual defect. This is invisible at an object's rest pose (an untransformed layer's
# pixels are bit-identical to the plate, so mask shape cannot matter there -- confirmed by
# reproducing this exact real mask + real LaMa reconstruction through this exact production
# code) and produces a hard, duplicate-looking seam once TRANSLATE displaces it, dragging the
# erroneously-included background along with the object -- see the ADR for the full repro.
# `0.3` sits with real margin on both sides of the real evidence above (0.202 highest normal,
# 0.455 the one confirmed defect), the same "evidenced-but-not-statistically-calibrated" status
# as every other threshold in this module.
#
# Independent review caught a real gap in the first version of this check (which flagged ANY
# single edge above this bound): a genuinely rectangular real object -- e.g. the "cloth-banner-
# shaped region" this project's own dataset (`phase3_action_page`/`eval_weapon_effects`) names
# as an explicitly valid target -- would touch BOTH edges of an axis near 100% together, which
# is indistinguishable from a one-sided over-segmentation by magnitude alone. The real defect's
# own measured values disambiguate this: LEFT=45.5% but the OPPOSITE edge, RIGHT, was only
# 0.56% (TOP=0.2%, BOTTOM=0.4%) -- a real rectangle's opposite edge would ALSO be high, not
# near-zero. The check below therefore requires ASYMMETRY (one side of an axis high, the other
# comfortably low), not just one side being high -- a real banner/flag (high on both left+right
# or both top+bottom) is not flagged; the real, evidenced over-segmentation defect still is.
_MAX_BBOX_EDGE_TOUCH_FRACTION = 0.3

# Phase 16 (drawn-effect track): the maximum bbox density (mask area / tight-bbox area) a
# RADIAL_EXPAND effect mask may have. A drawn effect's mask is SPARSE by nature -- radiating
# lines, a burst, an energy field -- so animating it moves only those lines; the panel
# background inside the bbox but outside the mask is untouched by the transform and by
# compositing. Real evidence (scripts/run_phase16_effect_mask_diagnostic.py, real 2xT4
# worker): proper effect masks measured density 0.28-0.50, while "select everything"
# over-inclusive masks (the confirmed-defective wind_breaker_finish cloth_5 at 0.902 and
# character_hair_7 at 0.843, from docs/phase11-results.md) sit at 0.84-0.90. 0.70 sits with
# real margin inside that gap. This is the post-segmentation half of the radial_expand
# safety pair -- the pre-segmentation half (transform_geometry.py) deliberately uses a
# looser bbox-area bound because the bbox is a poor proxy for an effect's sparse mask.
_MAX_EFFECT_MASK_DENSITY = 0.70


def segment_object(
    image: ImageArray,
    grounding: GroundingResult,
    client: SegmentationClient,
    *,
    max_mask_density: float | None = None,
) -> SegmentationResult:
    candidates = client.segment(image, grounding.bbox)
    if not candidates:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=grounding.object_id,
            detail=(
                f"segmentation model returned no mask candidates for box {grounding.bbox.as_xyxy()}"
            ),
            root_cause="the segmentation model failed to propose any mask for the grounded box",
            architectural=False,
            proposed_fix=(
                "verify the grounding box is sane; consider a different segmentation candidate"
            ),
        )

    best = max(candidates, key=lambda c: c.iou_score)
    mask = best.mask
    _validate_mask(mask, expected_shape=image.shape[:2], object_id=grounding.object_id)
    bbox = _tight_bbox(mask)
    _validate_mask_shape(mask, bbox, object_id=grounding.object_id)
    if max_mask_density is not None:
        _validate_mask_density(
            mask, bbox, max_density=max_mask_density, object_id=grounding.object_id
        )

    return SegmentationResult(
        object_id=grounding.object_id,
        mask=mask,
        bbox=bbox,
        model_id=client.model_id,
        iou_score=best.iou_score,
    )


def _validate_mask_density(
    mask: ImageArray, bbox: BBoxPx, *, max_density: float, object_id: str
) -> None:
    """Reject an effect mask that is NOT sparse -- a dense "select everything" mask that would
    move a large filled region (including panel background) when RADIAL_EXPAND's rim breathes,
    instead of only the drawn effect's lines. See `_MAX_EFFECT_MASK_DENSITY` for the evidence
    behind the bound.
    """
    area = float(np.count_nonzero(mask))
    bbox_area = float(bbox.width * bbox.height)
    density = area / bbox_area if bbox_area > 0 else 1.0
    if density <= max_density:
        return
    raise PipelineStageError(
        stage="segmentation",
        input_ref=object_id,
        detail=(
            f"effect mask fills {density:.1%} of its own tight bbox -- above the "
            f"{max_density:.0%} bound for a drawn effect; a sparse radiating-line/energy "
            "effect mask should fill a small fraction of its box, while this density is "
            "consistent with a 'select everything' mask that would drag panel background "
            "along when the effect pulses"
        ),
        root_cause=(
            "the segmentation model produced a dense, filled mask for a drawn-effect target "
            "whose real artwork is sparse lines/rays"
        ),
        architectural=False,
        proposed_fix=(
            "verify the grounding box is scoped to the effect (not the whole panel); a "
            "denser mask means the box captured the object the effect is attached to, not "
            "the effect itself"
        ),
    )


def _validate_mask(mask: ImageArray, *, expected_shape: tuple[int, int], object_id: str) -> None:
    if mask.ndim != 2:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=object_id,
            detail=f"segmentation mask must be 2D, got shape {mask.shape}",
            root_cause="segmentation returned a mask with an unexpected channel dimension",
            architectural=False,
            proposed_fix="normalize the model output to a single full-page mask",
        )
    if mask.shape != expected_shape:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=object_id,
            detail=(
                f"segmentation mask shape {mask.shape} does not match source image shape "
                f"{expected_shape}"
            ),
            root_cause="segmentation returned crop-local or otherwise mis-sized mask data",
            architectural=False,
            proposed_fix="post-process the model mask to the exact source-image geometry",
        )
    if mask.dtype != np.uint8:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=object_id,
            detail=f"segmentation mask must use uint8 values, got dtype {mask.dtype}",
            root_cause="segmentation output violated the MaskCandidate dtype contract",
            architectural=False,
            proposed_fix="convert the binary mask to uint8 values in {0, 255}",
        )
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


def _bbox_edge_touch_fractions(mask: ImageArray, bbox: BBoxPx) -> tuple[float, float, float, float]:
    """`(left, right, top, bottom)`: the fraction of each of `bbox`'s 4 sides that `mask` covers,

    within `bbox` itself (`bbox` is assumed to be `mask`'s own tight bbox, so every side is
    touched at least once by construction) -- a SMALL fraction means that side is touched only
    near one point (e.g. a jagged hair strand grazing it); a LARGE fraction means a long straight
    run of mask hugging that edge -- see `_MAX_BBOX_EDGE_TOUCH_FRACTION`'s own comment for the
    real evidence behind this signal.
    """
    sub = mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] > 0
    if sub.shape[0] == 0 or sub.shape[1] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(sub[:, 0].mean()),
        float(sub[:, -1].mean()),
        float(sub[0, :].mean()),
        float(sub[-1, :].mean()),
    )


def _one_sided_axis(a: float, b: float) -> bool:
    """True if exactly one of an axis's two opposite edges (`a`, `b`) is hugged while the other

    is not -- the real, evidenced over-segmentation signature (LEFT=45.5% but RIGHT=0.56% for
    the confirmed real defect), as opposed to a genuinely rectangular object (e.g. a real
    banner/flag), whose opposite edges would BOTH be near 100% together.
    """
    hi, lo = max(a, b), min(a, b)
    return hi > _MAX_BBOX_EDGE_TOUCH_FRACTION and lo <= _MAX_BBOX_EDGE_TOUCH_FRACTION


def _validate_mask_shape(mask: ImageArray, bbox: BBoxPx, *, object_id: str) -> None:
    left, right, top, bottom = _bbox_edge_touch_fractions(mask, bbox)
    horizontal_offending = _one_sided_axis(left, right)
    vertical_offending = _one_sided_axis(top, bottom)
    if not (horizontal_offending or vertical_offending):
        return

    if horizontal_offending:
        side, fraction, opposite = ("left", left, right) if left > right else ("right", right, left)
    else:
        side, fraction, opposite = ("top", top, bottom) if top > bottom else ("bottom", bottom, top)

    raise PipelineStageError(
        stage="segmentation",
        input_ref=object_id,
        detail=(
            f"mask hugs its own tight bbox's {side} edge for {fraction:.1%} of that edge's "
            f"length while the opposite edge is only {opposite:.1%} -- exceeds the "
            f"{_MAX_BBOX_EDGE_TOUCH_FRACTION:.0%} bound on one side with no matching coverage "
            "on the other, unlike a genuinely rectangular object (whose opposite edges would "
            "both be high together); this asymmetry is consistent with the mask "
            "over-segmenting into adjacent background/panel content on just one side, rather "
            "than tightly following the object's own silhouette"
        ),
        root_cause=(
            "the segmentation model's mask extends well beyond the intended object's real "
            "silhouette along one side of its bbox -- animating it would move that "
            "erroneously-included background/panel content along with the object, "
            "invisible at rest but producing a visible hard-edged seam once displaced"
        ),
        architectural=False,
        proposed_fix=(
            "verify the grounding box is sane and has enough contrast against the "
            "background on the affected side; consider a different segmentation candidate "
            "or a more specific semantic_label prompt"
        ),
    )
