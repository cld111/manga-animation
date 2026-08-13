"""End-to-end pipeline orchestration: real manga page -> playable seamless-loop MP4.

Wires the stage packages together in this actual order (pre-existing doc/code drift fixed
here: `docs/pipeline.md`'s diagram lists reconstruction before animation, but reconstruction
needs animation's transformed masks to know what a motion reveals, so it has always run after):

    analysis -> grounding -> validation -> segmentation -> mask_semantics -> animation
    -> reconstruction -> compositing -> rendering

`validation` (Phase 3.2, `src/manga_animation/validation`) sits between grounding and
segmentation: a grounding candidate that clears the grounding model's own detection threshold
is not automatically trusted as semantically correct — see
`docs/decisions/0006-grounding-target-validation.md`. If every ranked grounding candidate for
the plan's object fails validation, the run fails outright (`PipelineStageError`, stage=
`"validation"`) rather than animating an unvalidated best guess. `mask_semantics` (Phase 12,
`validation.mask_semantics`) sits between segmentation and animation: a segmented mask that
passes every existing geometric check is not automatically trusted as semantically correct
either — see `docs/decisions/0018-semantic-mask-validation.md`. The two "validation" stages are
deliberately distinct: `validation` operates on a grounding bbox before any mask exists;
`mask_semantics` operates on the real, segmented mask's own pixel content.

This orchestrates the plan's PRIMARY object end to end, plus (Phase 4, see
docs/decisions/0010-multi-object-layer-decomposition.md) any SECONDARY/MICRO objects the VLM
also proposed real motion for — `STATIC` objects the analysis stage identified are recorded in
the plan but never grounded, segmented, or animated (see
`.claude/agents/segmentation-agent.md`: "STATIC objects generally don't need
grounding/segmentation at all"). Every stage function this module calls already raises
`PipelineStageError` on failure; this module does not swallow or convert those into a false
success for the PRIMARY object — a failed run surfaces exactly which stage failed and why (see
the Phase 3.1 brief's "Failure policy", preserved unchanged in Phase 3.2). A SECONDARY/MICRO
object failing at grounding/validation/segmentation/mask_semantics does NOT fail the run — it is
dropped from the render (logged) while the PRIMARY object's own success/failure policy is
unaffected; see `_Z_ORDER_BY_MOTION_TYPE` and `ObjectRunResult` below.

This is orchestration code, not a stage itself — it owns none of the stages' internal
decisions, only the wiring between their already-defined public entry points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from manga_animation.analysis import Qwen25VLClient, VLMClient, analyze_page, analyze_page_panels
from manga_animation.animation import generate_transformed_layer
from manga_animation.benchmarking.registry import load_candidates
from manga_animation.compositing import composite_frame_stack
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import StageTimer, get_logger
from manga_animation.grounding import (
    GroundingClient,
    GroundingDinoClient,
    ground_object_candidates,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    FrameSequence,
    GroundingResult,
    Layer,
    MaskArray,
    MaskSemanticResult,
    PipelineStageError,
    ReconstructionResult,
    RenderResult,
    SegmentationResult,
    ValidationResult,
    normalized_bbox_to_px,
)
from manga_animation.reconstruction import (
    LamaClient,
    ReconstructionClient,
    reconstruct_hidden_region,
)
from manga_animation.rendering import render
from manga_animation.schemas.animation_plan import AnimationPlan, MotionType, ObjectPlan
from manga_animation.segmentation import Sam21Client, SegmentationClient, segment_object
from manga_animation.validation import validate_target, verify_mask_semantics

logger = get_logger(__name__)

# Phase 4 (docs/decisions/0010-multi-object-layer-decomposition.md): compositing z-order by
# MotionType -- higher composites on top. PRIMARY stays the reader's unoccluded focus (per the
# analysis prompt's own definition of "primary"); this project has no real per-object depth
# evidence to base an ordering on instead.
_Z_ORDER_BY_MOTION_TYPE: dict[MotionType, int] = {
    MotionType.MICRO: 0,
    MotionType.SECONDARY: 1,
    MotionType.PRIMARY: 2,
}

# Phase 8.3 (docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md, "Defect A" -- the real
# "duplicate silhouette" ghost found in `verified_action_1`): `compositing.composite_frame_stack`
# alpha-blends every animated layer independently, with no awareness of whether two layers'
# masks represent the same physical region. Two independently-accepted SECONDARY/MICRO objects
# (e.g. two separate `character_hair` grounding candidates) whose masks substantially overlap,
# each animated with its own MotionSpec, are invisible at rest (an untransformed layer's pixels
# are bit-identical to the plate, so mask shape/overlap cannot matter there) but visibly
# double-expose once their motions diverge -- reproduced deterministically against this exact
# production code (generate_transformed_layer + composite_frame_stack) using real source-image
# pixels, see the ADR for the repro script and evidence. `0.25` (a containment-style overlap
# fraction: intersection / area of the SMALLER mask, not IoU, so a small object fully nested
# inside a larger one's mask is still caught) is a documented, evidenced-but-not-statistically-
# calibrated choice, same status as this codebase's other deterministic thresholds -- the real
# defect's masks overlapped far above this (evidenced ~0.68 in the ADR's repro); genuinely
# distinct, only lightly-touching objects (e.g. two hair locks meeting at the scalp) are not
# expected to reach it.
_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION = 0.25


def _bbox_intersects(a: BBoxPx, b: BBoxPx) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _mask_overlap_fraction(
    mask_a: MaskArray, bbox_a: BBoxPx, mask_b: MaskArray, bbox_b: BBoxPx
) -> float:
    """`intersection(mask_a, mask_b) / min(area(mask_a), area(mask_b))` -- 0.0 when the two

    masks' own tight bboxes don't even overlap (the common case for genuinely distinct objects,
    cheap to reject before touching pixel data).
    """
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
    animated_objects: list[ObjectPlan],
    segmentation_by_object: dict[str, SegmentationResult],
    primary_object_id: str,
    dropped_objects: list[DroppedObjectResult],
) -> tuple[list[ObjectPlan], dict[str, SegmentationResult]]:
    """Drop any SECONDARY/MICRO object whose real segmentation mask overlaps an already-

    accepted object's mask by more than `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION` -- see this
    module's Defect-A comment above `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION` for why. Processes
    `animated_objects` in its existing order (PRIMARY always first, per `objects_to_animate`'s
    own construction) and keeps the FIRST of any conflicting pair -- deterministic, no ranking
    signal beyond plan order exists at this point. PRIMARY is never a candidate for being
    dropped here: it is always the first object considered, so it is trivially compared against
    an empty `kept` list and always kept -- its deliberate top z-order (`_Z_ORDER_BY_MOTION_TYPE`)
    is this codebase's existing, intentional way of letting a SECONDARY/MICRO object legitimately
    sit under/behind PRIMARY, which this check must not interfere with.
    """
    kept: list[ObjectPlan] = []
    kept_masks: list[MaskArray] = []
    kept_bboxes: list[BBoxPx] = []
    for obj in animated_objects:
        seg = segmentation_by_object[obj.object_id]
        conflict_with: str | None = None
        max_overlap = 0.0
        for other_obj, other_mask, other_bbox in zip(kept, kept_masks, kept_bboxes, strict=True):
            overlap = _mask_overlap_fraction(seg.mask, seg.bbox, other_mask, other_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                conflict_with = other_obj.object_id
        is_conflicting = (
            obj.object_id != primary_object_id
            and max_overlap > _MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION
        )
        if is_conflicting:
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
            dropped_objects.append(
                DroppedObjectResult(
                    object_plan=obj,
                    failing_stage="segmentation",
                    reason=(
                        f"mask overlaps already-accepted object_id={conflict_with!r} by "
                        f"{max_overlap:.1%} of its own area, exceeding the "
                        f"{_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION:.0%} bound -- animating both "
                        "independently would risk a double-exposure ghost once their motions "
                        "diverge"
                    ),
                )
            )
            continue
        kept.append(obj)
        kept_masks.append(seg.mask)
        kept_bboxes.append(seg.bbox)

    kept_ids = {obj.object_id for obj in kept}
    kept_segmentation = {
        object_id: seg for object_id, seg in segmentation_by_object.items() if object_id in kept_ids
    }
    return kept, kept_segmentation


@dataclass
class DroppedObjectResult:
    """A SECONDARY/MICRO object the plan proposed but that did not make it into the render
    (Phase 7.2.1, closing an evaluation-visibility gap ADR 0010 explicitly deferred to Phase 7:
    "extending evaluation to report on secondary/micro objects too is real future work").
    ADR 0010's failure policy already drops these without failing the whole run (see
    `run_pipeline`'s grounding/validation/segmentation stages below) -- this type is the
    additive record of WHICH object was dropped, at which stage, and why, so a caller (e.g.
    `evaluation/schemas.py::ObjectAttemptOutcome`) can see past `secondary_objects`, which only
    ever contained the objects that succeeded. `failing_stage="segmentation"` (Phase 8.3) covers
    two distinct real reasons: a genuine `PipelineStageError` from `segment_object` itself (e.g.
    the mask-shape check `segmentation/segment.py::_validate_mask_shape` added), or a mask that
    segmented successfully but conflicts with an already-accepted object's mask closely enough
    to risk a double-exposure ghost if both were animated independently (see
    `_drop_overlapping_secondary_objects` below) -- `reason` always distinguishes which.
    `failing_stage="mask_semantics"` (Phase 12) is distinct from both: the mask passed every
    geometric check but `validation.mask_semantics.verify_mask_semantics` REJECTed or ABSTAINed
    on its actual pixel content -- see `docs/decisions/0018-semantic-mask-validation.md`.
    """

    object_plan: ObjectPlan
    failing_stage: Literal["grounding", "validation", "segmentation", "mask_semantics"]
    reason: str


@dataclass
class ObjectRunResult:
    """Everything produced for one non-PRIMARY animated object during a run (Phase 4) -- a

    SECONDARY/MICRO object that was successfully grounded, validated, segmented, and passed
    semantic mask validation, and is therefore part of the render. An object that failed at any
    of those stages is simply absent from `PipelineRunResult.secondary_objects` (logged, not
    silently invented as a result) -- see `docs/decisions/0010-multi-object-layer-decomposition.md`.
    """

    object_plan: ObjectPlan
    grounding: GroundingResult
    validation_attempts: list[ValidationResult]
    segmentation: SegmentationResult
    mask_semantics: MaskSemanticResult | None
    """`None` only when `PipelineConfig.enable_semantic_mask_validation` is `False` -- the gate
    never ran, not that it produced no opinion (see Phase 12's config toggle)."""
    reconstruction: ReconstructionResult | None


@dataclass
class PipelineRunResult:
    """Everything the Phase 3.1 final report needs, from one real end-to-end run.

    `primary_object`/`grounding`/`validation_attempts`/`segmentation`/`reconstruction` describe
    ONLY the PRIMARY object, exactly as every phase through 3.3.x already had them -- unchanged
    in meaning, so no existing consumer of this type needs to change. `secondary_objects` is the
    new, additive Phase 4 field: zero or more SECONDARY/MICRO objects that also made it into the
    render, in the same order they appear in `plan.objects`.
    """

    image_path: Path
    plan: AnimationPlan
    primary_object: ObjectPlan
    grounding: GroundingResult
    validation_attempts: list[ValidationResult]
    segmentation: SegmentationResult
    mask_semantics: MaskSemanticResult | None
    """Phase 12, PRIMARY object only -- see `ObjectRunResult.mask_semantics`'s docstring for the
    `None` case."""
    reconstruction: ReconstructionResult | None
    render: RenderResult
    secondary_objects: list[ObjectRunResult] = field(default_factory=list)
    dropped_objects: list[DroppedObjectResult] = field(default_factory=list)
    """Phase 7.2.1: every SECONDARY/MICRO object the plan proposed that did NOT make it into
    the render (grounding or validation failure) -- additive, does not change the meaning of
    `secondary_objects` (still only the successful ones). Always empty for a single-object
    plan or a plan whose non-PRIMARY objects all succeeded."""


def _candidate_source(stage: str, config: PipelineConfig) -> str:
    """Resolve a `PipelineConfig.model_variants[stage]` candidate id to its HF/GitHub source,

    via the same shortlist `scripts/phase2_kaggle_benchmark.py` already validates against
    (`configs/benchmark_candidates.yaml`) -- model identity is config-driven, never hardcoded
    in stage code (see "Model Abstraction" in docs/architecture.md).
    """
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


def build_default_clients(
    config: PipelineConfig,
) -> tuple[VLMClient, GroundingClient, SegmentationClient, ReconstructionClient]:
    """Construct the real (GPU-backed) clients for every model stage, from `config` alone.

    Heavy imports (torch/transformers) only happen inside each client's `load()`/`generate()`/
    `detect()`/`segment()`/`inpaint()` methods (see each client module) -- constructing them
    here is cheap and safe even without the `ml` extra installed.
    """
    device = config.resolve_device()
    vlm_client = Qwen25VLClient(source=_candidate_source("vlm", config), dtype=config.dtype)
    # Real finding (Phase 3.1's first Kaggle run): Grounding DINO's processor produces
    # float32 pixel_values regardless of config.dtype, which raises "Input type (float) and
    # bias type (c10::Half) should be the same" against a float16-loaded model on this
    # transformers version. ADR 0005's actual successful benchmark runs for both Grounding
    # DINO and SAM 2.1 (see docs/phase2-benchmark-results.md) used float32 explicitly, never
    # float16 -- config.dtype's float16 default (configs/kaggle.yaml) was only ever proven
    # for the VLM stage. Hardcoding float32 here for these two stages reflects that real,
    # tested evidence rather than the single global config default; kaggle.yaml's own comment
    # already flagged dtype as something to "revisit per-model in Phase 2 if quality suffers".
    grounding_client = GroundingDinoClient(
        source=_candidate_source("grounding", config), device=device, dtype="float32"
    )
    segmentation_client = Sam21Client(
        source=_candidate_source("segmentation", config), device=device, dtype="float32"
    )
    reconstruction_client = LamaClient(device=device)
    return vlm_client, grounding_client, segmentation_client, reconstruction_client


def _select_primary(plan: AnimationPlan, image_path: str) -> ObjectPlan:
    primaries = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    if not primaries:
        # analyze_page() already guarantees exactly one PRIMARY object for a successfully
        # built plan (see plan_builder._rank_candidates) -- reaching this means a plan was
        # constructed by something else (e.g. a test fixture), not that analysis is buggy.
        raise PipelineStageError(
            stage="analysis",
            input_ref=image_path,
            detail="AnimationPlan has no PRIMARY object to animate",
            architectural=False,
            proposed_fix="Phase 3.1's pipeline requires a plan with exactly one PRIMARY object",
        )
    return primaries[0]


def _panel_bbox_px(plan: AnimationPlan, panel_id: str, page_shape: tuple[int, int]) -> BBoxPx:
    h, w = page_shape
    panel = next(p for p in plan.panels if p.panel_id == panel_id)
    return normalized_bbox_to_px(panel.bbox, page_width=w, page_height=h)


def run_pipeline(
    image_path: Path,
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    plan: AnimationPlan | None = None,
    analysis_mode: Literal["page", "panel"] = "panel",
) -> PipelineRunResult:
    """Run the complete pipeline (analysis through rendering, with Phase 3.2's grounding-
    validation gate) on one real manga page.

    Raises `PipelineStageError` (never a silent partial/false success) the moment any stage
    fails. `out_dir` receives the rendered MP4 and the intermediate frame sequence (kept, per
    the Phase 3.1 brief's "keep the frame sequence available as an ignored output artifact for
    debugging") -- both are generated, git-ignored artifacts (see ADR 0002), not canonical.

    `plan`: the controlled-fallback escape hatch the Phase 3.1 brief explicitly allows ("If
    the VLM produces an unusable or ambiguous plan: ... use a controlled fallback/test fixture
    if necessary; clearly distinguish the fallback from fully automatic operation"). Leave this
    `None` for real automatic operation (the default, and what every real run should use first).
    Pass a pre-built `AnimationPlan` ONLY when `analyze_page` has already been run for real and
    genuinely returned an unusable result (e.g. a defensible all-STATIC read on every candidate
    object) -- this skips the analysis stage's VLM call entirely, so the caller is responsible
    for recording that substitution honestly wherever this run's results are reported.

    `analysis_mode`: `"panel"` (default since Phase 10, see
    docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md -- deterministic
    panel detection followed by one VLM call per detected panel, `analyze_page_panels`; see
    docs/decisions/0007-panel-aware-analysis.md) or `"page"` (one whole-page VLM call,
    `analyze_page` -- the default through Phase 9). Real Phase 9 evidence
    (docs/phase9-results.md section 5.3) found panel mode dramatically more reliable on a
    10-sample real-world set (`end_to_end_completion_rate` 20%->60%, `grounding_success_rate`
    50%->100%, ERROR-classified outcomes 5->0) and, per Phase 10's own forensics
    (docs/phase10-results.md), page mode's single whole-page VLM call was the proximate cause of
    a real mid-cycle visual defect (`realworld_marika_love_meter`) that panel mode's independent,
    per-panel grounding correctly rejected instead of silently rendering. Ignored when `plan` is
    already supplied (there is no analysis stage to mode-switch on the controlled-fallback path).
    Everything downstream of analysis (grounding, validation, segmentation, animation,
    compositing, rendering) is identical either way -- this switch only changes how the
    `AnimationPlan` was produced, per ADR 0007's explicit decoupling from
    grounding/segmentation/animation. Pass `analysis_mode="page"` explicitly for the old default.
    """
    device = config.resolve_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    if plan is None:
        with StageTimer("analysis", logger, device=device, model=config.model_variants.get("vlm")):
            plan = (
                analyze_page_panels(image_path, vlm_client, config=config)
                if analysis_mode == "panel"
                else analyze_page(image_path, vlm_client, config=config)
            )
    else:
        logger.warning(
            "run_pipeline: using an externally supplied AnimationPlan -- the analysis stage's "
            "VLM call did NOT run for this invocation (controlled-fallback path, see the "
            "Phase 3.1 brief's failure policy)"
        )
    primary = _select_primary(plan, str(image_path))
    logger.info(
        "analysis selected PRIMARY object object_id=%s semantic_label=%s transform_kind=%s",
        primary.object_id,
        primary.semantic_label,
        primary.motion.transform_kind if primary.motion else None,
    )

    image = np.asarray(Image.open(image_path).convert("RGB"))
    page_shape = (image.shape[0], image.shape[1])

    # Phase 4: animate every non-STATIC object, not just the PRIMARY (see
    # docs/decisions/0010-multi-object-layer-decomposition.md). `objects_to_animate` always
    # starts with `primary` so every "primary vs. the rest" ordering below stays deterministic
    # and matches `plan.objects`' own order for the rest.
    objects_to_animate = [primary] + [
        obj
        for obj in plan.objects
        if obj.motion_type != MotionType.STATIC and obj.object_id != primary.object_id
    ]
    # Computed here (not just before the animation stage, as in Phase 3.1/3.2) so it's ready for
    # both consumers that need an object's real panel region: grounding itself (Phase 5.1, see
    # docs/decisions/0011-panel-aware-grounding.md -- Grounding DINO runs on this crop instead
    # of the full page) and validation's transform-geometry check (Phase 3.3.1, see
    # validation/transform_geometry.py), which still uses the same region as its reference
    # region. Each object may live in a different panel (panel-aware analysis mode), so this is
    # per-object.
    panel_bbox_px_by_object = {
        obj.object_id: _panel_bbox_px(plan, obj.panel_id, page_shape) for obj in objects_to_animate
    }

    def _is_primary(object_id: str) -> bool:
        return object_id == primary.object_id

    dropped_objects: list[DroppedObjectResult] = []

    with StageTimer(
        "grounding", logger, device=device, model=config.model_variants.get("grounding")
    ):
        grounding_client.load()
        candidates_by_object: dict[str, list[GroundingResult]] = {}
        try:
            for obj in objects_to_animate:
                try:
                    candidates_by_object[obj.object_id] = ground_object_candidates(
                        image,
                        obj,
                        grounding_client,
                        panel_bbox_px=panel_bbox_px_by_object[obj.object_id],
                    )
                except PipelineStageError as exc:
                    if _is_primary(obj.object_id):
                        raise
                    logger.warning(
                        "grounding found nothing for %s object_id=%s semantic_label=%s -- "
                        "dropping it from this render (PRIMARY is unaffected)",
                        obj.motion_type.value,
                        obj.object_id,
                        obj.semantic_label,
                    )
                    dropped_objects.append(
                        DroppedObjectResult(
                            object_plan=obj, failing_stage="grounding", reason=exc.detail
                        )
                    )
        finally:
            grounding_client.unload()

    with StageTimer("validation", logger, device=device, model=config.model_variants.get("vlm")):
        validation_attempts_by_object: dict[str, list[ValidationResult]] = {}
        accepted_by_object: dict[str, GroundingResult] = {}
        for obj in objects_to_animate:
            if obj.object_id not in candidates_by_object:
                continue  # already dropped at grounding
            attempts: list[ValidationResult] = []
            accepted: GroundingResult | None = None
            for rank, candidate in enumerate(candidates_by_object[obj.object_id]):
                result = validate_target(
                    image,
                    obj,
                    candidate,
                    vlm_client,
                    candidate_rank=rank,
                    panel_bbox_px=panel_bbox_px_by_object[obj.object_id],
                )
                attempts.append(result)
                if result.accepted:
                    accepted = candidate
                    break
            validation_attempts_by_object[obj.object_id] = attempts

            if accepted is not None:
                accepted_by_object[obj.object_id] = accepted
            elif _is_primary(obj.object_id):
                # Per the Phase 3.2 failure policy: a candidate that clears grounding's own
                # detection threshold is NOT the same as one that's semantically correct (see
                # docs/decisions/0006-grounding-target-validation.md) -- every ranked grounding
                # candidate was tried and none was accepted, so this run fails outright rather
                # than silently animating the best-scoring-but-unvalidated one. Phase 4 keeps
                # this exact policy for PRIMARY; SECONDARY/MICRO objects instead just drop out
                # of the render (see the `else` branch below).
                raise PipelineStageError(
                    stage="validation",
                    input_ref=obj.object_id,
                    detail=(
                        f"all {len(attempts)} grounding candidate(s) for "
                        f"semantic_label={obj.semantic_label!r} failed target validation: "
                        + "; ".join(f"rank={r.candidate_rank} {r.reason}" for r in attempts)
                    ),
                    root_cause=(
                        "no grounding candidate plausibly matched the intended semantic "
                        "target -- a technically valid detection is not the same as a "
                        "correct one"
                    ),
                    architectural=False,
                    proposed_fix=(
                        "retry with a different page/object, or supply a controlled-fallback "
                        "AnimationPlan (run_pipeline(..., plan=...)) for a human-verified target"
                    ),
                )
            else:
                logger.warning(
                    "every grounding candidate for %s object_id=%s semantic_label=%s failed "
                    "validation -- dropping it from this render (PRIMARY is unaffected)",
                    obj.motion_type.value,
                    obj.object_id,
                    obj.semantic_label,
                )
                dropped_objects.append(
                    DroppedObjectResult(
                        object_plan=obj,
                        failing_stage="validation",
                        reason="all "
                        + str(len(attempts))
                        + " grounding candidate(s) failed target validation: "
                        + "; ".join(f"rank={r.candidate_rank} {r.reason}" for r in attempts),
                    )
                )

    with StageTimer(
        "segmentation", logger, device=device, model=config.model_variants.get("segmentation")
    ):
        segmentation_client.load()
        segmentation_by_object: dict[str, SegmentationResult] = {}
        try:
            for obj in objects_to_animate:
                if obj.object_id not in accepted_by_object:
                    continue  # already dropped at grounding or validation
                try:
                    segmentation_by_object[obj.object_id] = segment_object(
                        image, accepted_by_object[obj.object_id], segmentation_client
                    )
                except PipelineStageError as exc:
                    # Phase 8.3 fix: this per-object try/except was missing entirely -- a
                    # SECONDARY/MICRO object's segmentation failure (e.g. this module's own
                    # docstring's mask-shape check) used to fail the WHOLE run, contradicting
                    # this file's own documented policy (see the module docstring's "A
                    # SECONDARY/MICRO object failing at grounding/validation/segmentation does
                    # NOT fail the run") -- grounding and validation above already implemented
                    # this correctly, segmentation alone did not.
                    if _is_primary(obj.object_id):
                        raise
                    logger.warning(
                        "segmentation failed for %s object_id=%s semantic_label=%s -- dropping "
                        "it from this render (PRIMARY is unaffected): %s",
                        obj.motion_type.value,
                        obj.object_id,
                        obj.semantic_label,
                        exc.detail,
                    )
                    dropped_objects.append(
                        DroppedObjectResult(
                            object_plan=obj, failing_stage="segmentation", reason=exc.detail
                        )
                    )
        finally:
            segmentation_client.unload()

    # Everything that survived grounding + validation + segmentation -- always includes
    # `primary` (its own failure at any prior stage already raised), plus zero or more
    # successful SECONDARY/MICRO objects, in `plan.objects` order.
    animated_objects = [
        obj for obj in objects_to_animate if obj.object_id in segmentation_by_object
    ]
    # Phase 8.3 (Defect A): drop any SECONDARY/MICRO object whose real mask overlaps an
    # already-accepted object's mask enough to risk a double-exposure ghost once both are
    # animated independently -- see `_drop_overlapping_secondary_objects`.
    animated_objects, segmentation_by_object = _drop_overlapping_secondary_objects(
        animated_objects, segmentation_by_object, primary.object_id, dropped_objects
    )

    # Phase 12 (docs/decisions/0018-semantic-mask-validation.md): does each real, ACCEPTED mask's
    # own pixel content actually match its semantic_label -- not just "is this mask's shape
    # unremarkable" (segmentation's own geometric checks above), but "does the mask's content
    # match its label" (docs/phase11-results.md section 6.4's confirmed, unresolved finding: a
    # SAM mask can pass every geometric check while covering substantially more/different real
    # content than its label). PRIMARY REJECT/ABSTAIN fails the run (same fail-closed policy as
    # grounding/validation above); SECONDARY/MICRO REJECT/ABSTAIN drops the object.
    mask_semantics_by_object: dict[str, MaskSemanticResult] = {}
    if config.enable_semantic_mask_validation:
        with StageTimer(
            "mask_semantics", logger, device=device, model=config.model_variants.get("vlm")
        ):
            kept_after_semantics: list[ObjectPlan] = []
            for obj in animated_objects:
                seg = segmentation_by_object[obj.object_id]
                mask_result = verify_mask_semantics(image, obj, seg.mask, seg.bbox, vlm_client)
                mask_semantics_by_object[obj.object_id] = mask_result
                if mask_result.accepted:
                    kept_after_semantics.append(obj)
                    continue
                if _is_primary(obj.object_id):
                    raise PipelineStageError(
                        stage="mask_semantics",
                        input_ref=obj.object_id,
                        detail=(
                            f"semantic mask validation {mask_result.verdict.upper()} for "
                            f"semantic_label={obj.semantic_label!r}: {mask_result.reason}"
                        ),
                        root_cause=(
                            "the segmented mask passed every geometric check but its actual "
                            "pixel content does not match the intended semantic target (or the "
                            "evidence was too weak to tell) -- see "
                            "docs/decisions/0018-semantic-mask-validation.md"
                        ),
                        architectural=False,
                        proposed_fix=(
                            "retry with a different page/object, or supply a "
                            "controlled-fallback AnimationPlan for a human-verified target"
                        ),
                    )
                logger.warning(
                    "semantic mask validation %s for %s object_id=%s semantic_label=%s -- "
                    "dropping it from this render (PRIMARY is unaffected): %s",
                    mask_result.verdict.upper(),
                    obj.motion_type.value,
                    obj.object_id,
                    obj.semantic_label,
                    mask_result.reason,
                )
                dropped_objects.append(
                    DroppedObjectResult(
                        object_plan=obj,
                        failing_stage="mask_semantics",
                        reason=f"{mask_result.verdict.upper()}: {mask_result.reason}",
                    )
                )
            animated_objects = kept_after_semantics
            kept_ids = {obj.object_id for obj in animated_objects}
            segmentation_by_object = {
                oid: seg for oid, seg in segmentation_by_object.items() if oid in kept_ids
            }

    with StageTimer("animation", logger, device="cpu", model=None):
        frame_count = plan.loop.frame_count
        layers: list[Layer] = []
        for obj in animated_objects:
            assert obj.motion is not None  # schema guarantees this for a non-STATIC ObjectPlan
            seg = segmentation_by_object[obj.object_id]
            transformed = tuple(
                generate_transformed_layer(
                    image,
                    seg.mask,
                    obj.motion,
                    panel_bbox_px_by_object[obj.object_id],
                    page_shape,
                    i / frame_count,
                    loop_duration_s=plan.loop.duration_s,
                    # `seg.bbox` is the same tight bbox `generate_transformed_layer` would
                    # otherwise recompute from `seg.mask` via a full-page np.where scan on every
                    # single frame call -- segmentation already computed it once, correctly (see
                    # `generate_transformed_layer`'s own docstring for why this is safe).
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
    layers_by_object = {layer.object_id: layer for layer in layers}

    with StageTimer(
        "reconstruction", logger, device=device, model=config.model_variants.get("inpainting")
    ):
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
                model_id=config.model_variants.get("inpainting", "lama-large"),
            )
            if recon is not None:
                reconstructions[obj.object_id] = recon

    with StageTimer("compositing", logger, device="cpu", model=None):
        frames = [
            composite_frame_stack(image, layers, i, reconstructions=reconstructions)
            for i in range(frame_count)
        ]

    with StageTimer("rendering", logger, device="cpu", model="ffmpeg/libx264"):
        frame_sequence = FrameSequence(frames=frames, fps=plan.loop.fps)
        render_result = render(
            frame_sequence,
            out_dir / "output.mp4",
            codec=config.output_codec,
            keep_frames=True,
            frames_dir=out_dir / "frames",
        )

    secondary_results = [
        ObjectRunResult(
            object_plan=obj,
            grounding=accepted_by_object[obj.object_id],
            validation_attempts=validation_attempts_by_object[obj.object_id],
            segmentation=segmentation_by_object[obj.object_id],
            mask_semantics=mask_semantics_by_object.get(obj.object_id),
            reconstruction=reconstructions.get(obj.object_id),
        )
        for obj in animated_objects
        if not _is_primary(obj.object_id)
    ]

    return PipelineRunResult(
        image_path=image_path,
        plan=plan,
        primary_object=primary,
        grounding=accepted_by_object[primary.object_id],
        validation_attempts=validation_attempts_by_object[primary.object_id],
        segmentation=segmentation_by_object[primary.object_id],
        mask_semantics=mask_semantics_by_object.get(primary.object_id),
        reconstruction=reconstructions.get(primary.object_id),
        render=render_result,
        secondary_objects=secondary_results,
        dropped_objects=dropped_objects,
    )
