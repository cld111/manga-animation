"""Maps one `ObjectPlan` to a pixel-space region of the source image.

`ground_object` and `ground_object_candidates` are the public entry points other stages/the
orchestrator should call — everything else here is an implementation detail of turning a
semantic label into a Grounding-DINO-friendly text prompt, optionally cropping to a real panel
region first (docs/decisions/0011-panel-aware-grounding.md), and picking the best detection(s),
always translated back into full-page pixel coordinates before returning.
"""

from __future__ import annotations

from manga_animation.grounding.client import GroundingClient
from manga_animation.pipeline.types import BBoxPx, GroundingResult, ImageArray, PipelineStageError
from manga_animation.schemas.animation_plan import ObjectPlan


def _prompt_from_label(semantic_label: str) -> str:
    """"flag_cloth" -> "flag cloth." — Grounding DINO's phrase-prompt convention (see

    `GROUNDING_PROMPT` in `scripts/phase2_kaggle_benchmark.py`: period-separated phrases).
    """
    phrase = semantic_label.replace("_", " ").strip()
    return f"{phrase}."


def _grounding_region(image: ImageArray, panel_bbox_px: BBoxPx | None) -> BBoxPx:
    """The pixel-space region Grounding DINO should actually see: `panel_bbox_px` when given,

    else the whole `image` (see `ground_object_candidates`'s docstring for the full rationale).
    Deliberately the *only* place that "no region given" gets turned into a concrete box, so
    every caller below this point (the crop, the local->page translation) works off one region,
    never a special-cased `None`.
    """
    if panel_bbox_px is not None:
        return panel_bbox_px
    h, w = image.shape[0], image.shape[1]
    return BBoxPx(x0=0, y0=0, x1=w, y1=h)


def ground_object_candidates(
    image: ImageArray,
    object_plan: ObjectPlan,
    client: GroundingClient,
    *,
    max_candidates: int = 3,
    panel_bbox_px: BBoxPx | None = None,
) -> list[GroundingResult]:
    """Ground one non-STATIC object's semantic label into a ranked list of pixel bboxes.

    Returns every usable detection (highest score first, degenerate-after-clipping boxes
    dropped), up to `max_candidates` — not just the single best one. This exists so the
    Phase 3.2 validation stage (`src/manga_animation/validation`) can try the next-ranked
    grounding candidate when the top one fails validation ("attempt another ranked grounding
    candidate if available", per the Phase 3.2 brief's failure policy) without re-running the
    grounding model — `client.detect()` already returns every box above its own detection
    threshold in one call.

    `panel_bbox_px`: the object's real panel region in full-page pixel space, if known (see
    `pipeline.orchestrator._panel_bbox_px`) — mirrors the exact convention
    `validation.validate_target` already uses for its own `panel_bbox_px` parameter. When given,
    `client.detect()` runs against the CROP of `image` bounded by this region instead of the
    full page (docs/decisions/0011-panel-aware-grounding.md) — Grounding DINO never sees the
    unrelated rest of a large page. `None` (the default) preserves the exact pre-ADR-0011
    full-page behavior; this is also what a real panel-detection `fallback_full_page` region and
    page-level analysis's own synthetic `(0, 0, 1, 1)` panel already resolve to (both are, in
    pixel space, exactly `(0, 0, image_width, image_height)` — see ADR 0011's "Fallback
    behavior"), so passing either of those through here changes nothing; only a genuinely
    smaller, real detected panel actually shrinks what the model sees.

    Every returned `GroundingResult.bbox` is always in full-page pixel coordinates, never
    crop-local — `client.detect()`'s output is translated by the region's own `(x0, y0)` offset
    before this function does anything else with it (ranking, clipping, capping), so nothing
    downstream of this function ever has to know grounding ran on a crop at all.

    Callers are responsible for only invoking this on objects that actually need grounding
    (STATIC objects don't — see `.claude/agents/segmentation-agent.md`'s "Working from the
    Animation Plan"). Raises `PipelineStageError` when nothing clears the detection threshold
    at all, or when every detection's box lies (partially or fully) outside the image after
    clipping — never returns an empty list silently.
    """
    region = _grounding_region(image, panel_bbox_px)
    crop = image[region.y0 : region.y1, region.x0 : region.x1]

    prompt = _prompt_from_label(object_plan.semantic_label)
    detections = client.detect(crop, prompt)
    if not detections:
        raise PipelineStageError(
            stage="grounding",
            input_ref=object_plan.object_id,
            detail=(
                f"no detection above threshold for prompt {prompt!r} in region "
                f"{region.as_xyxy()} of a {image.shape[1]}x{image.shape[0]} page"
            ),
            root_cause=(
                "the grounding model found nothing matching the semantic label at the "
                "configured threshold — either the label doesn't describe anything visually "
                "present, or the threshold/phrasing needs tuning for this page"
            ),
            architectural=False,
            proposed_fix=(
                "check semantic_label against the actual artwork; consider a lower detection "
                "threshold or a rephrased prompt; do not substitute a guessed box"
            ),
        )

    page_h, page_w = image.shape[0], image.shape[1]
    ranked = sorted(detections, key=lambda d: d.score, reverse=True)
    candidates: list[GroundingResult] = []
    for det in ranked:
        lx0, ly0, lx1, ly1 = det.box  # crop-local pixel coordinates (region.x0/.y0 == 0 when
        # panel_bbox_px was None, so this is a no-op translation in the pre-ADR-0011 case)
        x0, y0 = lx0 + region.x0, ly0 + region.y0
        x1, y1 = lx1 + region.x0, ly1 + region.y0
        # Clip against the FULL PAGE, not just the crop's own bounds: a local box can still
        # overshoot the crop's edge in local coordinates (exactly like the pre-existing
        # full-page case already had to handle), and translating it can in principle also push
        # it past the page's far edge if the crop itself sat flush against that edge.
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(page_w, x1), min(page_h, y1)
        if x1 <= x0 or y1 <= y0:
            continue  # degenerate after clipping — not a usable candidate, skip it
        bbox = BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1, score=det.score)
        candidates.append(
            GroundingResult(object_id=object_plan.object_id, bbox=bbox, model_id=client.model_id)
        )
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        raise PipelineStageError(
            stage="grounding",
            input_ref=object_plan.object_id,
            detail=(
                f"every detection for prompt {prompt!r} ({len(detections)} candidate(s)) lies "
                f"entirely outside the {page_w}x{page_h} page after translating out of region "
                f"{region.as_xyxy()} and clipping to bounds"
            ),
            root_cause="grounding model returned only boxes outside the image's coordinate space",
            architectural=False,
            proposed_fix="verify the image resolution passed to the grounding client matches",
        )
    return candidates


def ground_object(
    image: ImageArray,
    object_plan: ObjectPlan,
    client: GroundingClient,
    *,
    panel_bbox_px: BBoxPx | None = None,
) -> GroundingResult:
    """Ground one non-STATIC object's semantic label into its single best pixel bbox.

    Convenience wrapper over `ground_object_candidates` for callers that only want the top
    candidate (e.g. existing callers/tests predating the Phase 3.2 validation stage) —
    identical detection/clipping/error/region behavior, just narrowed to one result.
    """
    return ground_object_candidates(
        image, object_plan, client, max_candidates=1, panel_bbox_px=panel_bbox_px
    )[0]
