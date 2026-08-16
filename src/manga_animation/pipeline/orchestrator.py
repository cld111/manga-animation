"""End-to-end pipeline orchestration (Phase 18.4 architecture): real manga page -> seamless MP4.

The Phase 18.4 architecture calls the VLM EXACTLY ONCE in the whole pipeline -- at the
per-candidate object-description stage, which now runs BEFORE segmentation. There is no
analysis stage (no Qwen-driven AnimationPlan up front), no crop-based VLM validation, and
no mask-semantics VLM gate:

    grounding (DINO, labels from the caller) -> object_description (Qwen2.5-VL: FULL image
    + bbox pixel coordinates -> structured description; fail-closed) -> segmentation (SAM2,
    ONLY for accepted bboxes, masks kept for animation) -> animation planning
    (deterministic ranking and MotionSpec mapping + transform-geometry gate) -> animation
    (SAM mask + MotionSpec) -> reconstruction (LaMa) -> compositing -> rendering (H.264 +
    loop metrics)

Candidate labels are supplied by the caller (or the documented default list): the pipeline
no longer invents them with a VLM. Every model family is loaded once per stage
(`ModelStage`, ADR 0020); Qwen is resident during ONE stage (object_description) and
processes every candidate of every panel there -- never again.

A candidate that fails a deterministic or model gate is dropped (logged); if NO candidate is
accepted, the run fails with stage="object_description" (fail closed, never an unvalidated
animation). Accepted candidates are ranked deterministically: highest description confidence
becomes PRIMARY, the rest SECONDARY.

This is orchestration code, not a stage itself -- it owns none of the stages' internal
decisions, only the wiring between their already-defined public entry points.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from manga_animation.analysis import Qwen25VLClient, VLMClient
from manga_animation.animation import generate_transformed_layer
from manga_animation.benchmarking.registry import load_candidates
from manga_animation.compositing import composite_frame_stack
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import StageTimer, get_logger
from manga_animation.core.seed import set_global_seed
from manga_animation.grounding import (
    GroundingClient,
    GroundingDinoClient,
    ground_object_candidates,
)
from manga_animation.object_description import CandidateBox, describe_objects
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.types import (
    BBoxPx,
    FrameSequence,
    GroundingResult,
    Layer,
    MaskArray,
    ObjectDescriptionResult,
    PipelineStageError,
    ReconstructionResult,
    RenderResult,
    SegmentationResult,
)
from manga_animation.reconstruction import (
    LamaClient,
    ReconstructionClient,
    reconstruct_hidden_region,
)
from manga_animation.rendering import render
from manga_animation.schemas.animation_plan import (
    AnimationPlan,
    BBox,
    LoopSpec,
    MotionType,
    ObjectPlan,
    PanelPlan,
    SourceImage,
)
from manga_animation.segmentation import Sam21Client, SegmentationClient, segment_object
from manga_animation.validation.transform_geometry import check_transform_geometry

logger = get_logger(__name__)

# Compositing z-order by MotionType -- higher composites on top. PRIMARY stays the reader's
# unoccluded focus; no real per-object depth evidence exists to base an ordering on instead.
_Z_ORDER_BY_MOTION_TYPE: dict[MotionType, int] = {
    MotionType.MICRO: 0,
    MotionType.SECONDARY: 1,
    MotionType.PRIMARY: 2,
}

# Phase 8.3 (Defect A): two independently-accepted objects whose masks substantially overlap
# each animated with its own MotionSpec visibly double-expose once their motions diverge
# (reproduced deterministically against this exact production code, see
# docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md). `0.25` is a containment-style
# overlap fraction (intersection / area of the SMALLER mask, not IoU), evidenced-but-not-
# statistically-calibrated, same status as this codebase's other deterministic thresholds.
_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION = 0.25

# The default candidate labels the pipeline grounds when the caller supplies none. The Phase
# 18.3 architecture deliberately does NOT invent labels with a VLM; these cover the common
# animatable categories and are overridable per call.
DEFAULT_ANIMATION_LABELS: tuple[str, ...] = (
    "character",
    "character_hair",
    "flag_banner",
    "weapon",
    "speed_lines",
    "impact_burst",
)

_PANEL_ID = "panel_1"


def _bbox_intersects(a: BBoxPx, b: BBoxPx) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _bbox_intersection_area(a: BBoxPx, b: BBoxPx) -> int:
    if not _bbox_intersects(a, b):
        return 0
    return max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(
        0, min(a.y1, b.y1) - max(a.y0, b.y0)
    )


def _cross_panel_conflict(
    bbox: BBoxPx,
    current_panel_bbox: BBoxPx,
    neighboring_panel_bboxes: tuple[BBoxPx, ...],
) -> BBoxPx | None:
    """Return a neighboring panel crossed by a materially ambiguous candidate bbox."""
    bbox_area = bbox.width * bbox.height
    if bbox_area <= 0:
        return None
    for neighbor in neighboring_panel_bboxes:
        if neighbor == current_panel_bbox:
            continue
        overlap_fraction = _bbox_intersection_area(bbox, neighbor) / bbox_area
        if overlap_fraction >= 0.10:
            return neighbor
    return None


def _mask_overlap_fraction(
    mask_a: MaskArray, bbox_a: BBoxPx, mask_b: MaskArray, bbox_b: BBoxPx
) -> float:
    """`intersection(mask_a, mask_b) / min(area(mask_a), area(mask_b))` -- 0.0 when the two
    masks' own tight bboxes don't even overlap (the common case, cheap to reject first)."""
    if not _bbox_intersects(bbox_a, bbox_b):
        return 0.0
    x0, y0 = max(bbox_a.x0, bbox_b.x0), max(bbox_a.y0, bbox_b.y0)
    x1, y1 = min(bbox_a.x1, bbox_b.x1), min(bbox_a.y1, bbox_b.y1)
    intersection = int(np.count_nonzero((mask_a[y0:y1, x0:x1] > 0) & (mask_b[y0:y1, x0:x1] > 0)))
    if intersection == 0:
        return 0.0
    area_a = int(np.count_nonzero(mask_a > 0))
    area_b = int(np.count_nonzero(mask_b > 0))
    return intersection / min(area_a, area_b)


def _drop_overlapping_secondary_objects(
    accepted: list[tuple[ObjectPlan, SegmentationResult]],
    primary_object_id: str,
) -> list[tuple[ObjectPlan, SegmentationResult]]:
    """Drop any non-PRIMARY candidate whose real segmentation mask overlaps an already-kept
    candidate's mask by more than `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION` -- the Phase 8.3
    Defect-A guard (see the constant's comment). PRIMARY is never a drop candidate: it is
    always considered first and kept."""
    kept: list[tuple[ObjectPlan, SegmentationResult]] = []
    for obj, seg in accepted:
        conflict_with: str | None = None
        max_overlap = 0.0
        for other_obj, other_seg in kept:
            overlap = _mask_overlap_fraction(
                seg.mask, seg.bbox, other_seg.mask, other_seg.bbox
            )
            if overlap > max_overlap:
                max_overlap = overlap
                conflict_with = other_obj.object_id
        if (
            obj.object_id != primary_object_id
            and max_overlap > _MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION
        ):
            logger.warning(
                "segmentation: object_id=%s semantic_label=%s mask overlaps already-accepted "
                "object_id=%s by %.1f%% of its own area (> %.0f%% bound) -- dropping it from "
                "this render to avoid a double-exposure ghost (see Defect A, "
                "docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md)",
                obj.object_id,
                obj.semantic_label,
                conflict_with,
                max_overlap * 100,
                _MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION * 100,
            )
            continue
        kept.append((obj, seg))
    return kept


@dataclass
class DroppedObjectResult:
    """A candidate that did not make it into the render, with the stage and reason."""

    object_plan: ObjectPlan
    failing_stage: Literal["grounding", "segmentation", "object_description"]
    reason: str


@dataclass
class ObjectRunResult:
    """Everything produced for one non-PRIMARY animated object during a run."""

    object_plan: ObjectPlan
    grounding: GroundingResult
    segmentation: SegmentationResult
    object_description: ObjectDescriptionResult
    reconstruction: ReconstructionResult | None


@dataclass
class PipelineRunResult:
    """Everything produced by one real end-to-end run.

    `primary_object`/`grounding`/`segmentation`/`reconstruction`/`object_description` describe
    the PRIMARY object (the highest-confidence accepted candidate). `secondary_objects` are
    the other accepted candidates that also made it into the render, in plan order.
    `dropped_objects` records every candidate rejected at some stage, with the reason.
    """

    image_path: Path
    plan: AnimationPlan
    primary_object: ObjectPlan
    grounding: GroundingResult
    segmentation: SegmentationResult
    object_description: ObjectDescriptionResult
    reconstruction: ReconstructionResult | None
    render: RenderResult
    secondary_objects: list[ObjectRunResult] = field(default_factory=list)
    dropped_objects: list[DroppedObjectResult] = field(default_factory=list)


def _candidate_source(stage: str, config: PipelineConfig) -> str:
    """Resolve a `PipelineConfig.model_variants[stage]` candidate id to its HF/GitHub source
    via the benchmark candidate manifest -- model identity is config-driven, never hardcoded."""
    candidate_id = config.model_variants.get(stage)
    if candidate_id is None:
        raise PipelineStageError(
            stage="analysis" if stage == "vlm" else stage,  # type: ignore[arg-type]
            input_ref=stage,
            detail=f"no model_variants[{stage!r}] configured",
            architectural=False,
            proposed_fix="set model_variants in configs/default.yaml (or the active env profile)",
        )
    for candidate in load_candidates().get(stage, []):
        if candidate.id == candidate_id:
            return candidate.source
    raise PipelineStageError(
        stage="analysis" if stage == "vlm" else stage,  # type: ignore[arg-type]
        input_ref=candidate_id,
        detail=(
            f"candidate {candidate_id!r} not found in configs/benchmark_candidates.yaml[{stage!r}]"
        ),
        architectural=False,
        proposed_fix="check the candidate id spelling against the manifest",
    )


_RUNTIME_CANDIDATES: dict[str, set[str]] = {
    # Manifest entries without a production client remain benchmark candidates.
    "vlm": {"qwen2.5-vl-7b-instruct"},
    "grounding": {"grounding-dino-swin-l"},
    "segmentation": {"sam2.1-hiera-base"},
    "inpainting": {"lama-large"},
}


def _runtime_candidate(stage: str, config: PipelineConfig) -> tuple[str, str]:
    candidate_id = config.model_variants.get(stage)
    source = _candidate_source(stage, config)
    if candidate_id not in _RUNTIME_CANDIDATES[stage]:
        error_stage = "analysis" if stage == "vlm" else stage
        raise PipelineStageError(
            stage=error_stage,  # type: ignore[arg-type]
            input_ref=candidate_id or stage,
            detail=(
                f"configured candidate {candidate_id!r} has no production client for stage "
                f"{stage!r}"
            ),
            architectural=False,
            proposed_fix=(
                f"use one of {sorted(_RUNTIME_CANDIDATES[stage])} or implement/register an "
                "adapter for the requested candidate"
            ),
        )
    assert candidate_id is not None
    return candidate_id, source


def build_default_clients(
    config: PipelineConfig,
) -> tuple[VLMClient, GroundingClient, SegmentationClient, ReconstructionClient]:
    """Construct the real (GPU-backed) clients for every model stage, from `config` alone.

    Heavy imports (torch/transformers) only happen inside each client's methods -- constructing
    them here is cheap and safe even without the `ml` extra installed.
    """
    device = config.resolve_device()
    _, vlm_source = _runtime_candidate("vlm", config)
    _, grounding_source = _runtime_candidate("grounding", config)
    _, segmentation_source = _runtime_candidate("segmentation", config)
    inpainting_id, _ = _runtime_candidate("inpainting", config)
    vlm_client = Qwen25VLClient(source=vlm_source, dtype=config.dtype)
    grounding_client = GroundingDinoClient(source=grounding_source, device=device, dtype="float32")
    segmentation_client = Sam21Client(source=segmentation_source, device=device, dtype="float32")
    reconstruction_client = LamaClient(device=device, model_id=inpainting_id)
    return vlm_client, grounding_client, segmentation_client, reconstruction_client


def _slugify(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"obj_{slug or 'object'}_{index}"


def _candidate_plan(label: str, index: int) -> ObjectPlan:
    """A placeholder `ObjectPlan` for grounding one label. STATIC with no motion -- only
    `semantic_label` is read by grounding/description; the real motion and motion_type are
    assigned later from the accepted description (the description is the source of truth)."""
    return ObjectPlan(
        object_id=_slugify(label, index),
        panel_id=_PANEL_ID,
        semantic_label=label,
        confidence=1.0,
        motion_type=MotionType.STATIC,
        motion=None,
    )


def _ground_labels(
    image: np.ndarray,
    labels: Sequence[str],
    grounding_client: GroundingClient,
    *,
    panel_bbox_px: BBoxPx | None,
    max_candidates: int = 3,
) -> tuple[list[ObjectPlan], dict[str, list[GroundingResult]], list[DroppedObjectResult]]:
    """Ground every label on this crop. A label with no detection above threshold is dropped
    (a normal, expected outcome -- not every label exists on every page)."""
    objects = [_candidate_plan(label, index) for index, label in enumerate(labels)]
    candidates_by_object: dict[str, list[GroundingResult]] = {}
    dropped_objects: list[DroppedObjectResult] = []
    for obj in objects:
        try:
            candidates_by_object[obj.object_id] = ground_object_candidates(
                image,
                obj,
                grounding_client,
                max_candidates=max_candidates,
                panel_bbox_px=panel_bbox_px,
            )
        except PipelineStageError as exc:
            logger.warning(
                "grounding found nothing for semantic_label=%s -- dropping it: %s",
                obj.semantic_label,
                exc.detail,
            )
            dropped_objects.append(
                DroppedObjectResult(object_plan=obj, failing_stage="grounding", reason=exc.detail)
            )
    return objects, candidates_by_object, dropped_objects


def _segment_candidates(
    image: np.ndarray,
    candidates_by_object: dict[str, list[GroundingResult]],
    plan_by_object: dict[str, ObjectPlan],
    segmentation_client: SegmentationClient,
    *,
    accepted_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], SegmentationResult], list[DroppedObjectResult]]:
    """Segment ONLY the candidates the object-description stage accepted (Phase 18.4
    ordering: DINO -> Qwen -> SAM, so SAM never runs on a bbox without an action
    description). A candidate whose mask fails the shape/coverage checks is dropped
    (fail closed); the object's other accepted candidates are unaffected."""
    segmentation_by_candidate: dict[tuple[str, int], SegmentationResult] = {}
    dropped_objects: list[DroppedObjectResult] = []
    for object_id, candidates in candidates_by_object.items():
        for rank, candidate in enumerate(candidates):
            if (object_id, rank) not in accepted_keys:
                continue  # rejected (or unparseable) at object description -- no SAM call
            try:
                segmentation_by_candidate[(object_id, rank)] = segment_object(
                    image, candidate, segmentation_client
                )
            except PipelineStageError as exc:
                logger.warning(
                    "segmentation failed for accepted candidate object_id=%s rank=%d -- "
                    "dropping it: %s",
                    object_id,
                    rank,
                    exc.detail,
                )
                dropped_objects.append(
                    DroppedObjectResult(
                        object_plan=plan_by_object[object_id],
                        failing_stage="segmentation",
                        reason=exc.detail,
                    )
                )
    return segmentation_by_candidate, dropped_objects


def _describe_candidates(
    image: np.ndarray,
    candidates_by_object: dict[str, list[GroundingResult]],
    plan_by_object: dict[str, ObjectPlan],
    vlm_client: VLMClient,
    config: PipelineConfig,
    *,
    panel_bbox_px: BBoxPx | None,
) -> tuple[
    dict[tuple[str, int], ObjectDescriptionResult],
    list[DroppedObjectResult],
]:
    """THE pipeline's single VLM stage: ONE call per image with the image and ALL of its
    grounded candidates' bboxes as pixel coordinates (the model sees every candidate at
    once, never one crop per candidate). Runs BEFORE segmentation (Phase 18.4 ordering:
    DINO -> Qwen -> SAM); masks play no role here. Accepted descriptions additionally pass
    the deterministic transform-geometry gate (a semantically-good box can still be
    geometrically unsafe for its mapped motion kind). Fail-closed per candidate."""

    batch: list[CandidateBox] = []
    batch_keys: list[tuple[str, int]] = []
    for object_id, candidates in candidates_by_object.items():
        for rank, candidate in enumerate(candidates):
            batch.append(
                CandidateBox(
                    object_id=object_id,
                    semantic_label=plan_by_object[object_id].semantic_label,
                    bbox=candidate.bbox,
                )
            )
            batch_keys.append((object_id, rank))

    descriptions_by_candidate: dict[tuple[str, int], ObjectDescriptionResult] = {}
    dropped_objects: list[DroppedObjectResult] = []
    if not batch:
        return descriptions_by_candidate, dropped_objects

    try:
        results = describe_objects(
            image, batch, vlm_client, max_long_edge=config.resolution
        )
    except Exception as exc:  # noqa: BLE001 -- a failed batch call drops every candidate
        logger.warning(
            "object description batch call failed (%s) -- dropping %d candidate(s)",
            exc,
            len(batch),
        )
        for (object_id, _rank), _box in zip(batch_keys, batch, strict=True):
            dropped_objects.append(
                DroppedObjectResult(
                    object_plan=plan_by_object[object_id],
                    failing_stage="object_description",
                    reason=f"VLM call failed: {type(exc).__name__}: {exc}",
                )
            )
        return descriptions_by_candidate, dropped_objects

    for (object_id, rank), description in zip(batch_keys, results, strict=True):
        descriptions_by_candidate[(object_id, rank)] = description
        if not description.accepted:
            logger.info(
                "object description REJECT for object_id=%s rank=%d "
                "(rejection_reason=%s): %s",
                object_id,
                rank,
                description.rejection_reason,
                description.reason,
            )
            dropped_objects.append(
                DroppedObjectResult(
                    object_plan=plan_by_object[object_id],
                    failing_stage="object_description",
                    reason=(
                        f"{description.rejection_reason or 'unparseable'}: "
                        f"{description.reason}"
                    ),
                )
            )
            continue
        # Deterministic transform-geometry gate (no VLM): the mapped MotionSpec must be
        # geometrically safe for this bbox (the Phase 3.3.1 protection, kept from the old
        # validation stage).
        assert description.motion_spec is not None
        candidate = candidates_by_object[object_id][rank]
        compatible, geometry_reason = check_transform_geometry(
            candidate.bbox,
            description.motion_spec.transform_kind,
            panel_bbox_px=panel_bbox_px,
            image_shape=(image.shape[0], image.shape[1]),
        )
        if not compatible:
            logger.info(
                "object description ACCEPT but geometry REJECT for object_id=%s rank=%d: %s",
                object_id,
                rank,
                geometry_reason,
            )
            dropped_objects.append(
                DroppedObjectResult(
                    object_plan=plan_by_object[object_id],
                    failing_stage="object_description",
                    reason=geometry_reason,
                )
            )
            continue
        logger.info(
            "object description ACCEPT for object_id=%s rank=%d confidence=%.2f "
            "motion_kind=%s",
            object_id,
            rank,
            description.confidence or 0.0,
            description.motion_spec.transform_kind.value,
        )
    return descriptions_by_candidate, dropped_objects


def _build_plan(
    image_path: Path,
    image_size: tuple[int, int],
    config: PipelineConfig,
    *,
    accepted: list[tuple[str, int, ObjectPlan, GroundingResult, SegmentationResult,
                         ObjectDescriptionResult]],
    global_origin: tuple[int, int] = (0, 0),
    logical_panel_bbox_px: BBoxPx | None = None,
    neighboring_panel_bboxes: tuple[BBoxPx, ...] = (),
) -> tuple[AnimationPlan, ObjectPlan, list[tuple[ObjectPlan, GroundingResult, SegmentationResult,
                                                 ObjectDescriptionResult]]]:
    """Assemble the final `AnimationPlan` from accepted descriptions and rank them:
    highest description confidence -> PRIMARY, the rest SECONDARY (deterministic tiebreak by
    object_id). Applies the cross-panel ambiguity rejection to each accepted bbox.

    Raises `PipelineStageError` (stage="object_description") when no candidate was accepted --
    the fail-closed outcome of the pipeline's only semantic stage.
    """
    if not accepted:
        raise PipelineStageError(
            stage="object_description",
            input_ref=str(image_path),
            detail="no grounded candidate was accepted by the object-description stage",
            root_cause=(
                "every candidate failed the VLM's bbox assessment, was unparseable, or failed "
                "the deterministic transform-geometry gate"
            ),
            architectural=False,
            proposed_fix=(
                "choose labels that actually appear on the page, or accept a REJECTED/STATIC "
                "outcome as the honest result"
            ),
        )

    # Cross-panel ambiguity rejection (the old validation stage's geometric guard, now applied
    # to accepted candidates): a bbox materially crossing a neighboring logical panel is a
    # conservative reject.
    kept: list[tuple[str, int, ObjectPlan, GroundingResult, SegmentationResult,
                     ObjectDescriptionResult]] = []
    for object_id, rank, obj, grounding, seg, description in accepted:
        if logical_panel_bbox_px is not None:
            ox, oy = global_origin
            global_bbox = BBoxPx(
                x0=grounding.bbox.x0 + ox,
                y0=grounding.bbox.y0 + oy,
                x1=grounding.bbox.x1 + ox,
                y1=grounding.bbox.y1 + oy,
            )
            conflict = _cross_panel_conflict(
                global_bbox, logical_panel_bbox_px, neighboring_panel_bboxes
            )
            if conflict is not None:
                logger.warning(
                    "grounded bbox %s materially crosses logical neighbor panel %s -- "
                    "rejecting this candidate conservatively",
                    global_bbox.as_xyxy(),
                    conflict.as_xyxy(),
                )
                continue
        kept.append((object_id, rank, obj, grounding, seg, description))

    if not kept:
        raise PipelineStageError(
            stage="object_description",
            input_ref=str(image_path),
            detail="every accepted candidate crossed a neighboring logical panel",
            root_cause="object ownership cannot be safely attributed to one panel",
            architectural=False,
            proposed_fix="leave the object static or provide a panel-local target",
        )

    kept.sort(
        key=lambda item: (item[5].confidence or 0.0, item[0]), reverse=True
    )
    primary_description = kept[0][5]

    width, height = image_size
    source = SourceImage(path=str(image_path), width=width, height=height)
    panel = PanelPlan(panel_id=_PANEL_ID, bbox=BBox(x=0.0, y=0.0, width=1.0, height=1.0))
    loop = LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True)

    # Rebuild each accepted item with the final ObjectPlan carrying the description's motion
    # and the PRIMARY/SECONDARY motion_type. object_id is unique per (label, rank) so several
    # accepted candidates of one label coexist in the plan.
    finalized: list[tuple[ObjectPlan, GroundingResult, SegmentationResult,
                          ObjectDescriptionResult]] = []
    for index, (object_id, rank, obj, grounding, seg, description) in enumerate(kept):
        assert description.motion_spec is not None
        final_plan = ObjectPlan(
            object_id=f"{object_id}_{rank}",
            panel_id=_PANEL_ID,
            semantic_label=obj.semantic_label,
            confidence=description.confidence or 0.0,
            motion_type=MotionType.PRIMARY if index == 0 else MotionType.SECONDARY,
            motion=description.motion_spec,
        )
        finalized.append((final_plan, grounding, seg, description))

    plan = AnimationPlan(
        source=source,
        panels=[panel],
        objects=[item[0] for item in finalized],
        loop=loop,
    )
    primary_obj = finalized[0][0]
    logger.info(
        "animation planning: PRIMARY=%s (confidence=%.2f), %d secondary accepted",
        primary_obj.semantic_label,
        primary_description.confidence or 0.0,
        len(finalized) - 1,
    )
    return plan, primary_obj, finalized


def _animate_objects(
    image: np.ndarray,
    animated_objects: list[ObjectPlan],
    segmentation_by_object: dict[str, SegmentationResult],
    panel_bbox_px_by_object: dict[str, BBoxPx],
    plan: AnimationPlan,
) -> tuple[list[Layer], dict[str, Layer]]:
    """Deterministic per-object frame generation (CPU). Returns the `Layer` list (compositing
    z-order contract, `_Z_ORDER_BY_MOTION_TYPE`) and the object-id -> Layer map."""
    frame_count = plan.loop.frame_count
    layers: list[Layer] = []
    for obj in animated_objects:
        assert obj.motion is not None  # plan construction guarantees a spec for non-STATIC
        seg = segmentation_by_object[obj.object_id]
        transformed = tuple(
            generate_transformed_layer(
                image,
                seg.mask,
                obj.motion,
                panel_bbox_px_by_object[obj.object_id],
                (image.shape[0], image.shape[1]),
                i / frame_count,
                loop_duration_s=plan.loop.duration_s,
                object_bbox_px=seg.bbox,
            )
            for i in range(frame_count)
        )
        layers.append(
            Layer(
                object_id=obj.object_id,
                frames=transformed,
                z_order=_Z_ORDER_BY_MOTION_TYPE[obj.motion_type],
            )
        )
    return layers, {layer.object_id: layer for layer in layers}


def _reconstruct_objects(
    image: np.ndarray,
    animated_objects: list[ObjectPlan],
    segmentation_by_object: dict[str, SegmentationResult],
    layers_by_object: dict[str, Layer],
    reconstruction_client: ReconstructionClient,
    config: PipelineConfig,
) -> dict[str, ReconstructionResult]:
    """Fill every object's motion-revealed hole (no LaMa call for objects that never reveal
    one). Assumes `reconstruction_client` is already loaded and passes `managed_loaded=True`
    so the whole reconstruction stage keeps LaMa resident once (Phase 14 stage-level lifecycle)."""
    reconstructions: dict[str, ReconstructionResult] = {}
    for obj in animated_objects:
        seg = segmentation_by_object[obj.object_id]
        layer = layers_by_object[obj.object_id]
        recon = reconstruct_hidden_region(
            image,
            seg.mask,
            [mask for _, mask in layer.frames],
            reconstruction_client,
            object_id=obj.object_id,
            model_id=str(
                getattr(
                    reconstruction_client,
                    "model_id",
                    config.model_variants.get("inpainting", "unknown"),
                )
            ),
            managed_loaded=True,
        )
        if recon is not None:
            reconstructions[obj.object_id] = recon
    return reconstructions


def _composite_and_render(
    image: np.ndarray,
    layers: list[Layer],
    reconstructions: dict[str, ReconstructionResult],
    plan: AnimationPlan,
    out_dir: Path,
    config: PipelineConfig,
    *,
    video_filename: str,
    frames_dir: Path | None,
) -> RenderResult:
    frame_count = plan.loop.frame_count
    with StageTimer("compositing", logger, device="cpu", model=None):
        frames = [
            composite_frame_stack(image, layers, i, reconstructions=reconstructions)
            for i in range(frame_count)
        ]
    with StageTimer("rendering", logger, device="cpu", model="ffmpeg/libx264"):
        frame_sequence = FrameSequence(frames=frames, fps=plan.loop.fps)
        return render(
            frame_sequence,
            out_dir / video_filename,
            codec=config.output_codec,
            keep_frames=True,
            frames_dir=frames_dir,
        )


def run_pipeline(
    image_path: Path,
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    labels: Sequence[str] | None = None,
    panel_bbox_px: BBoxPx | None = None,
    global_origin: tuple[int, int] = (0, 0),
    logical_panel_bbox_px: BBoxPx | None = None,
    neighboring_panel_bboxes: tuple[BBoxPx, ...] = (),
    video_filename: str = "output.mp4",
    frames_dir: Path | None = None,
) -> PipelineRunResult:
    """Run the Phase 18.4 pipeline on one image: grounding -> object description -> the
    pipeline's single VLM stage -> segmentation (only accepted bboxes) -> animation
    planning -> animation -> reconstruction -> compositing -> rendering.

    `labels`: the candidate semantic labels to ground (DINO prompts). Defaults to
    `DEFAULT_ANIMATION_LABELS`. The pipeline never invents labels with a VLM.

    Raises `PipelineStageError` (never a silent partial/false success) the moment a stage
    fails. `out_dir` receives the rendered MP4 and the intermediate frame sequence -- both
    git-ignored artifacts (ADR 0002).
    """
    set_global_seed(config.seed)
    device = config.resolve_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    active_labels = list(labels or DEFAULT_ANIMATION_LABELS)

    # Stage 1: grounding -- DINO once for every label on this canvas, then released.
    with (
        ModelStage(grounding_client, name="grounding"),
        StageTimer(
            "grounding", logger, device=device, model=config.model_variants.get("grounding")
        ),
    ):
        plans_by_object, candidates_by_object, dropped = _ground_labels(
            image, active_labels, grounding_client, panel_bbox_px=panel_bbox_px
        )
    plan_by_object = {plan.object_id: plan for plan in plans_by_object}
    dropped_objects: list[DroppedObjectResult] = list(dropped)

    # Stage 2: object description -- THE pipeline's single VLM stage. Qwen loads once,
    # processes every grounded candidate of this canvas (ONE call with the image + ALL its
    # bboxes), then is released. Runs BEFORE segmentation (Phase 18.4 ordering: DINO ->
    # Qwen -> SAM): SAM only sees bboxes that earned an action description.
    with (
        ModelStage(vlm_client, name="object_description"),
        StageTimer(
            "object_description",
            logger,
            device=device,
            model=config.model_variants.get("vlm"),
        ),
    ):
        descriptions_by_candidate, dropped = _describe_candidates(
            image,
            candidates_by_object,
            plan_by_object,
            vlm_client,
            config,
            panel_bbox_px=panel_bbox_px,
        )
    dropped_objects.extend(dropped)

    # Stage 3: segmentation -- SAM2 once, ONLY for candidates accepted at object
    # description, then released. Masks are kept for the animation stage; they were never
    # an input to the VLM.
    accepted_keys = {
        key for key, description in descriptions_by_candidate.items() if description.accepted
    }
    with (
        ModelStage(segmentation_client, name="segmentation"),
        StageTimer(
            "segmentation", logger, device=device, model=config.model_variants.get("segmentation")
        ),
    ):
        segmentation_by_candidate, dropped = _segment_candidates(
            image,
            candidates_by_object,
            plan_by_object,
            segmentation_client,
            accepted_keys=accepted_keys,
        )
    dropped_objects.extend(dropped)

    # Stage 4: animation planning (deterministic, no model) -- rank accepted candidates,
    # build the schema-valid AnimationPlan with the description-mapped MotionSpecs.
    accepted: list[
        tuple[str, int, ObjectPlan, GroundingResult, SegmentationResult, ObjectDescriptionResult]
    ] = []
    for (object_id, rank), description in descriptions_by_candidate.items():
        if not description.accepted:
            continue
        if (object_id, rank) not in segmentation_by_candidate:
            continue  # accepted by the VLM but dropped at segmentation (mask shape gate)
        accepted.append(
            (
                object_id,
                rank,
                plan_by_object[object_id],
                candidates_by_object[object_id][rank],
                segmentation_by_candidate[(object_id, rank)],
                description,
            )
        )
    plan, primary, kept = _build_plan(
        image_path,
        (image.shape[0], image.shape[1]),
        config,
        accepted=accepted,
        global_origin=global_origin,
        logical_panel_bbox_px=logical_panel_bbox_px,
        neighboring_panel_bboxes=neighboring_panel_bboxes,
    )

    animated_objects = [item[0] for item in kept]
    segmentation_by_object = {
        item[0].object_id: item[2] for item in kept
    }
    descriptions_by_object = {
        item[0].object_id: item[3] for item in kept
    }
    grounding_by_object = {item[0].object_id: item[1] for item in kept}
    panel_bbox_px_by_object = {
        obj.object_id: (panel_bbox_px or BBoxPx(x0=0, y0=0, x1=image.shape[1], y1=image.shape[0]))
        for obj in animated_objects
    }

    with StageTimer("animation", logger, device="cpu", model=None):
        layers, layers_by_object = _animate_objects(
            image,
            animated_objects,
            segmentation_by_object,
            panel_bbox_px_by_object,
            plan,
        )

    with (
        ModelStage(reconstruction_client, name="reconstruction"),
        StageTimer(
            "reconstruction", logger, device=device, model=config.model_variants.get("inpainting")
        ),
    ):
        reconstructions = _reconstruct_objects(
            image,
            animated_objects,
            segmentation_by_object,
            layers_by_object,
            reconstruction_client,
            config,
        )

    render_result = _composite_and_render(
        image,
        layers,
        reconstructions,
        plan,
        out_dir,
        config,
        video_filename=video_filename,
        frames_dir=frames_dir or out_dir / "frames",
    )

    primary_grounding = grounding_by_object[primary.object_id]
    primary_segmentation = segmentation_by_object[primary.object_id]
    primary_description = descriptions_by_object[primary.object_id]
    secondary_results = [
        ObjectRunResult(
            object_plan=obj,
            grounding=grounding_by_object[obj.object_id],
            segmentation=segmentation_by_object[obj.object_id],
            object_description=descriptions_by_object[obj.object_id],
            reconstruction=reconstructions.get(obj.object_id),
        )
        for obj in animated_objects
        if obj.object_id != primary.object_id
    ]

    return PipelineRunResult(
        image_path=image_path,
        plan=plan,
        primary_object=primary,
        grounding=primary_grounding,
        segmentation=primary_segmentation,
        object_description=primary_description,
        reconstruction=reconstructions.get(primary.object_id),
        render=render_result,
        secondary_objects=secondary_results,
        dropped_objects=dropped_objects,
    )
