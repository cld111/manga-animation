"""End-to-end pipeline orchestration: real manga page -> playable seamless-loop MP4.

Wires the stage packages together in the order `docs/pipeline.md` documents:

    analysis -> grounding -> validation -> segmentation -> reconstruction -> animation
    -> compositing -> rendering

`validation` (Phase 3.2, `src/manga_animation/validation`) sits between grounding and
segmentation: a grounding candidate that clears the grounding model's own detection threshold
is not automatically trusted as semantically correct — see
`docs/decisions/0006-grounding-target-validation.md`. If every ranked grounding candidate for
the plan's object fails validation, the run fails outright (`PipelineStageError`, stage=
`"validation"`) rather than animating an unvalidated best guess.

This orchestrates exactly ONE animated (`PRIMARY`) object end to end — `STATIC` objects the
analysis stage identified are recorded in the plan but never grounded, segmented, or animated
(see `.claude/agents/segmentation-agent.md`: "STATIC objects generally don't need
grounding/segmentation at all"). Every stage function this module calls already raises
`PipelineStageError` on failure; this module does not swallow or convert those into a false
success — a failed run surfaces exactly which stage failed and why (see the Phase 3.1 brief's
"Failure policy", preserved unchanged in Phase 3.2).

This is orchestration code, not a stage itself — it owns none of the stages' internal
decisions, only the wiring between their already-defined public entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from manga_animation.analysis import Qwen25VLClient, VLMClient, analyze_page, analyze_page_panels
from manga_animation.animation import generate_transformed_layer
from manga_animation.benchmarking.registry import load_candidates
from manga_animation.compositing import composite_frame
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
from manga_animation.validation import validate_target

logger = get_logger(__name__)


@dataclass
class PipelineRunResult:
    """Everything the Phase 3.1 final report needs, from one real end-to-end run."""

    image_path: Path
    plan: AnimationPlan
    primary_object: ObjectPlan
    grounding: GroundingResult
    validation_attempts: list[ValidationResult]
    segmentation: SegmentationResult
    reconstruction: ReconstructionResult | None
    render: RenderResult


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
            f"candidate {candidate_id!r} not found in "
            f"configs/benchmark_candidates.yaml[{stage!r}]"
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
    analysis_mode: Literal["page", "panel"] = "page",
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

    `analysis_mode`: `"page"` (default, unchanged Phase 3.2 behavior -- one whole-page VLM
    call, `analyze_page`) or `"panel"` (Phase 3.3 -- deterministic panel detection followed by
    one VLM call per detected panel, `analyze_page_panels`; see
    docs/decisions/0007-panel-aware-analysis.md). Ignored when `plan` is already supplied
    (there is no analysis stage to mode-switch on the controlled-fallback path). Everything
    downstream of analysis (grounding, validation, segmentation, animation, compositing,
    rendering) is identical either way -- this switch only changes how the `AnimationPlan` was
    produced, per ADR 0007's explicit decoupling from grounding/segmentation/animation.
    """
    device = config.resolve_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    if plan is None:
        with StageTimer(
            "analysis", logger, device=device, model=config.model_variants.get("vlm")
        ):
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
    # Computed here (not just before the animation stage, as in Phase 3.1/3.2) so validation's
    # transform-geometry check (Phase 3.3.1) can measure a candidate bbox against its REAL
    # panel region, not just the full page -- see validation/transform_geometry.py.
    panel_bbox_px = _panel_bbox_px(plan, primary.panel_id, page_shape)

    with StageTimer(
        "grounding", logger, device=device, model=config.model_variants.get("grounding")
    ):
        grounding_client.load()
        try:
            grounding_candidates = ground_object_candidates(image, primary, grounding_client)
        finally:
            grounding_client.unload()

    with StageTimer("validation", logger, device=device, model=config.model_variants.get("vlm")):
        validation_attempts: list[ValidationResult] = []
        grounding: GroundingResult | None = None
        for rank, candidate in enumerate(grounding_candidates):
            result = validate_target(
                image,
                primary,
                candidate,
                vlm_client,
                candidate_rank=rank,
                panel_bbox_px=panel_bbox_px,
            )
            validation_attempts.append(result)
            if result.accepted:
                grounding = candidate
                break

    if grounding is None:
        # Per the Phase 3.2 failure policy: a candidate that clears grounding's own detection
        # threshold is NOT the same as one that's semantically correct (see
        # docs/decisions/0006-grounding-target-validation.md) -- every ranked grounding
        # candidate was tried and none was accepted, so this run fails outright rather than
        # silently animating the best-scoring-but-unvalidated one. The caller decides whether
        # to retry with a human-verified `plan=` override (the existing controlled fallback).
        raise PipelineStageError(
            stage="validation",
            input_ref=primary.object_id,
            detail=(
                f"all {len(validation_attempts)} grounding candidate(s) for "
                f"semantic_label={primary.semantic_label!r} failed target validation: "
                + "; ".join(f"rank={r.candidate_rank} {r.reason}" for r in validation_attempts)
            ),
            root_cause=(
                "no grounding candidate plausibly matched the intended semantic target -- a "
                "technically valid detection is not the same as a correct one"
            ),
            architectural=False,
            proposed_fix=(
                "retry with a different page/object, or supply a controlled-fallback "
                "AnimationPlan (run_pipeline(..., plan=...)) for a human-verified target"
            ),
        )

    with StageTimer(
        "segmentation", logger, device=device, model=config.model_variants.get("segmentation")
    ):
        segmentation_client.load()
        try:
            segmentation = segment_object(image, grounding, segmentation_client)
        finally:
            segmentation_client.unload()

    assert primary.motion is not None  # schema guarantees this for a PRIMARY object

    with StageTimer("animation", logger, device="cpu", model=None):
        frame_count = plan.loop.frame_count
        transformed = [
            generate_transformed_layer(
                image,
                segmentation.mask,
                primary.motion,
                panel_bbox_px,
                page_shape,
                i / frame_count,
                loop_duration_s=plan.loop.duration_s,
            )
            for i in range(frame_count)
        ]

    with StageTimer(
        "reconstruction", logger, device=device, model=config.model_variants.get("inpainting")
    ):
        reconstruction = reconstruct_hidden_region(
            image,
            segmentation.mask,
            [mask for _, mask in transformed],
            reconstruction_client,
            object_id=primary.object_id,
            model_id=config.model_variants.get("inpainting", "lama-large"),
        )

    with StageTimer("compositing", logger, device="cpu", model=None):
        frames = [
            composite_frame(image, layer, mask, reconstruction=reconstruction)
            for layer, mask in transformed
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

    return PipelineRunResult(
        image_path=image_path,
        plan=plan,
        primary_object=primary,
        grounding=grounding,
        validation_attempts=validation_attempts,
        segmentation=segmentation,
        reconstruction=reconstruction,
        render=render_result,
    )
