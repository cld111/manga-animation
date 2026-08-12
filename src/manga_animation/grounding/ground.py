"""Maps one `ObjectPlan` to a pixel-space region of the source image.

Only `ground_object` is the public entry point other stages/the orchestrator should call —
everything else here is an implementation detail of turning a semantic label into a
Grounding-DINO-friendly text prompt and picking the best detection.
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


def ground_object_candidates(
    image: ImageArray,
    object_plan: ObjectPlan,
    client: GroundingClient,
    *,
    max_candidates: int = 3,
) -> list[GroundingResult]:
    """Ground one non-STATIC object's semantic label into a ranked list of pixel bboxes.

    Returns every usable detection (highest score first, degenerate-after-clipping boxes
    dropped), up to `max_candidates` — not just the single best one. This exists so the
    Phase 3.2 validation stage (`src/manga_animation/validation`) can try the next-ranked
    grounding candidate when the top one fails validation ("attempt another ranked grounding
    candidate if available", per the Phase 3.2 brief's failure policy) without re-running the
    grounding model — `client.detect()` already returns every box above its own detection
    threshold in one call.

    Callers are responsible for only invoking this on objects that actually need grounding
    (STATIC objects don't — see `.claude/agents/segmentation-agent.md`'s "Working from the
    Animation Plan"). Raises `PipelineStageError` when nothing clears the detection threshold
    at all, or when every detection's box lies (partially or fully) outside the image after
    clipping — never returns an empty list silently.
    """
    prompt = _prompt_from_label(object_plan.semantic_label)
    detections = client.detect(image, prompt)
    if not detections:
        raise PipelineStageError(
            stage="grounding",
            input_ref=object_plan.object_id,
            detail=f"no detection above threshold for prompt {prompt!r}",
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

    h, w = image.shape[0], image.shape[1]
    ranked = sorted(detections, key=lambda d: d.score, reverse=True)
    candidates: list[GroundingResult] = []
    for det in ranked:
        x0, y0, x1, y1 = det.box
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
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
                f"entirely outside the {w}x{h} image after clipping to bounds"
            ),
            root_cause="grounding model returned only boxes outside the image's coordinate space",
            architectural=False,
            proposed_fix="verify the image resolution passed to the grounding client matches",
        )
    return candidates


def ground_object(
    image: ImageArray, object_plan: ObjectPlan, client: GroundingClient
) -> GroundingResult:
    """Ground one non-STATIC object's semantic label into its single best pixel bbox.

    Convenience wrapper over `ground_object_candidates` for callers that only want the top
    candidate (e.g. existing callers/tests predating the Phase 3.2 validation stage) —
    identical detection/clipping/error behavior, just narrowed to one result.
    """
    return ground_object_candidates(image, object_plan, client, max_candidates=1)[0]
