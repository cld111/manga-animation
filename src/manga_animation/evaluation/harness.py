"""The one real pipeline-invoking piece of the evaluation harness -- turns one `EvalSample`

plus one `analysis_mode` into a real `PageRunOutcome` by actually calling
`pipeline.orchestrator.run_pipeline` (or, for the nondeterminism check,
`analysis.plan_builder.analyze_page`). Phase 9: extracted from
`scripts/run_phase3_3_evaluation.py` (which duplicated this exact ~150 lines of logic) so a
second evaluation driver (`scripts/run_phase9_evaluation.py`) can reuse it instead of
re-implementing it -- CLAUDE.md's "do not introduce a second parallel implementation".

**Deliberately excluded from `evaluation/__init__.py`'s default export list.** Every other
module in this package (`dataset.py`/`schemas.py`/`metrics.py`/`nondeterminism.py`) imports
nothing torch/transformers-related, so `import manga_animation.evaluation` stays safely
importable without the `ml` extras -- a real property this project's test suite relies on
(`tests/` runs locally with no GPU/torch installed). `pipeline.orchestrator` itself is
import-safe without torch (every real model client lazily imports torch inside its own
methods, not at module load, per `docs/architecture.md`'s "GPU Awareness" principle), so this
module *could* be imported safely too -- but it is kept out of the package's default import
graph anyway, so that promise stays true by construction rather than by accident. Only
`scripts/run_phase3_3_evaluation.py`/`scripts/run_phase9_evaluation.py` (which already only run
on the remote GPU worker per CLAUDE.md's "pipeline is not run locally" policy) import this
module directly, e.g. `from manga_animation.evaluation.harness import run_one_sample`.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from manga_animation.analysis.panels import detect_panels
from manga_animation.analysis.plan_builder import analyze_page
from manga_animation.evaluation.artifacts import detect_seam_like_artifacts
from manga_animation.evaluation.dataset import EvalSample
from manga_animation.evaluation.nondeterminism import (
    NondeterminismSummary,
    RepeatedRunRecord,
    summarize_repeated_runs,
)
from manga_animation.evaluation.schemas import (
    LoopMetricsOutcome,
    MaskSemanticOutcome,
    ObjectAttemptOutcome,
    PageRunOutcome,
    RenderSummary,
    ValidationAttemptOutcome,
)
from manga_animation.pipeline.orchestrator import run_pipeline
from manga_animation.pipeline.types import MaskSemanticResult, PipelineStageError, RenderResult
from manga_animation.schemas.animation_plan import MotionType

CURRENT_SCHEMA_VERSION = 5
"""The `PageRunOutcome.schema_version` every producer using this harness writes -- see that

field's own docstring for what each version number means. Named here (not just inlined as a
literal `3` at every construction site) so a future harness change that bumps this has exactly
one place to edit."""


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_metadata(device: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        if device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            meta["gpu_count"] = gpu_count
            meta["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except ImportError:
        meta["torch_version"] = None
    return meta


def panel_detection_evidence(image_path: Path) -> tuple[int, list[str]]:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    panels = detect_panels(image)
    return len(panels), [p.source for p in panels]


def object_outcome_motion_type(motion_type: MotionType) -> Literal["secondary", "micro"]:
    """Narrows `MotionType` to `ObjectAttemptOutcome.motion_type`'s literal type -- every real

    call site here reads `motion_type` off an object drawn from `PipelineRunResult.
    secondary_objects`/`dropped_objects`, which by `pipeline/orchestrator.py`'s own construction
    (`objects_to_animate` minus `primary`, itself already filtered to non-STATIC) never contains
    a PRIMARY or STATIC object -- so this can't actually raise on any real input; the ValueError
    is a defensive backstop against a future orchestrator change silently breaking that
    invariant, not a case this harness expects to hit.
    """
    if motion_type == MotionType.SECONDARY:
        return "secondary"
    if motion_type == MotionType.MICRO:
        return "micro"
    raise ValueError(
        f"unexpected motion_type={motion_type!r} for a secondary/dropped object -- "
        "only SECONDARY/MICRO objects should ever reach this point"
    )


def _decode_frames(video_path: Path) -> list[np.ndarray]:
    """Real decode via `cv2.VideoCapture` (same approach `rendering.encode._validate` already

    uses to measure real encoded output rather than trusting an in-memory frame sequence) --
    every frame this harness's artifact check inspects is what a viewer would actually see.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        ok, frame = cap.read()
        while ok:
            frames.append(frame)
            ok, frame = cap.read()
    finally:
        cap.release()
    return frames


def mask_semantics_outcome_from_result(
    result: MaskSemanticResult | None,
) -> MaskSemanticOutcome | None:
    """`pipeline.types.MaskSemanticResult` -> `evaluation.schemas.MaskSemanticOutcome` -- `None`

    in, `None` out (the gate didn't run for this object, or was disabled entirely), same
    convention `render_summary_from_result` has no equivalent for since `RenderResult` always
    exists on a completed run.
    """
    if result is None:
        return None
    return MaskSemanticOutcome(
        verdict=result.verdict,
        vlm_matches=result.vlm_matches,
        vlm_confidence=result.vlm_confidence,
        reason=result.reason,
        method=result.method,
        unexpected_content=list(result.unexpected_content),
        geometric_signals=dict(result.geometric_signals),
    )


def render_summary_from_result(
    render: RenderResult, *, seam_artifact_suspected: bool | None = None
) -> RenderSummary:
    """`pipeline.types.RenderResult` (carries a `Path`, not JSON-serializable as-is) ->

    `evaluation.schemas.RenderSummary`. `seam_artifact_suspected` is computed by the caller
    (`run_one_sample`, from the real decoded output) and threaded through here rather than
    decoded again inside this function, so this stays a pure, easily-testable mapping.
    """
    loop_metrics = None
    if render.loop_metrics is not None:
        lm = render.loop_metrics
        loop_metrics = LoopMetricsOutcome(
            ordinary_adjacent_step_mean_abs_diff=lm.ordinary_adjacent_step_mean_abs_diff,
            wrap_step_mean_abs_diff=lm.wrap_step_mean_abs_diff,
            wrap_step_within_2x_ordinary=lm.wrap_step_within_2x_ordinary,
            ordinary_adjacent_step_ssim=lm.ordinary_adjacent_step_ssim,
            wrap_step_ssim=lm.wrap_step_ssim,
            wrap_ssim_within_tolerance=lm.wrap_ssim_within_tolerance,
        )
    return RenderSummary(
        frame_count=render.frame_count,
        fps=render.fps,
        resolution=render.resolution,
        duration_s=render.duration_s,
        codec=render.codec,
        pixel_format=render.pixel_format,
        seamless_loop_verified=render.seamless_loop_verified,
        loop_metrics=loop_metrics,
        seam_artifact_suspected=seam_artifact_suspected,
    )


def run_one_sample(
    sample: EvalSample,
    mode: Literal["page", "panel"],
    config: Any,
    clients: tuple[Any, Any, Any, Any],
    out_dir: Path,
    panel_count: int,
    panel_sources: list[str],
) -> PageRunOutcome:
    """Run the real production pipeline once, for one sample/mode, and record what happened as

    a `PageRunOutcome` -- never raises (a `PipelineStageError` or any other exception is
    recorded as a failed/errored outcome, not propagated), so one sample's real crash never
    stops the rest of an evaluation run.
    """
    vlm_client, grounding_client, segmentation_client, reconstruction_client = clients
    image_path = Path(sample.image_path)
    try:
        result = run_pipeline(
            image_path,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=out_dir,
            analysis_mode=mode,
        )
    except PipelineStageError as exc:
        # No PipelineRunResult exists on this path (run_pipeline raised before returning), so
        # there is no visibility into which SECONDARY/MICRO objects might have been attempted
        # -- object_outcomes/render_summary stay empty/None here, same as any other
        # schema_version=3 page that genuinely had none/nothing rendered.
        return PageRunOutcome(
            sample_id=sample.sample_id,
            analysis_mode=mode,
            status="failed",
            failing_stage=exc.stage,
            failure_detail=exc.detail,
            panel_count=panel_count,
            panel_sources=panel_sources,
            schema_version=CURRENT_SCHEMA_VERSION,
        )
    except Exception as exc:  # noqa: BLE001 -- one sample's unexpected crash must not stop
        # the rest of the evaluation run; recorded distinctly from a classified
        # PipelineStageError (failing_stage="unexpected").
        return PageRunOutcome(
            sample_id=sample.sample_id,
            analysis_mode=mode,
            status="failed",
            failing_stage="unexpected",
            failure_detail=f"{type(exc).__name__}: {exc}",
            panel_count=panel_count,
            panel_sources=panel_sources,
            schema_version=CURRENT_SCHEMA_VERSION,
        )

    object_outcomes = [
        ObjectAttemptOutcome(
            object_id=obj.object_plan.object_id,
            semantic_label=obj.object_plan.semantic_label,
            motion_type=object_outcome_motion_type(obj.object_plan.motion_type),
            status="rendered",
            validation_attempts=[
                ValidationAttemptOutcome(
                    candidate_rank=v.candidate_rank,
                    accepted=v.accepted,
                    grounding_score=v.grounding_score,
                    reason=v.reason,
                )
                for v in obj.validation_attempts
            ],
            mask_semantics=mask_semantics_outcome_from_result(obj.mask_semantics),
        )
        for obj in result.secondary_objects
    ] + [
        ObjectAttemptOutcome(
            object_id=dropped.object_plan.object_id,
            semantic_label=dropped.object_plan.semantic_label,
            motion_type=object_outcome_motion_type(dropped.object_plan.motion_type),
            status="dropped",
            # DroppedObjectResult.reason already carries a real, human-readable summary of why
            # this object was dropped -- surfacing it here means the saved JSON alone explains a
            # drop. Only meaningful when the drop happened AT validation (`dropped.reason` is
            # validation-attempt prose in that case); a grounding-stage drop never reached
            # validation, so there is nothing validation-shaped to report. A mask_semantics-stage
            # drop (Phase 12) is a real, disclosed gap of the same shape: `dropped.reason`
            # carries the real verdict/VLM reason as prose (see orchestrator.py's construction of
            # `DroppedObjectResult(failing_stage="mask_semantics", ...)`), but there is no
            # structured `MaskSemanticResult` retained for a dropped object to populate
            # `mask_semantics=` with (only kept/rendered objects carry the real result object
            # through to this point) -- left `mask_semantics=None`/empty `validation_attempts`
            # here, same as every other non-validation drop reason, not silently invented.
            validation_attempts=(
                [
                    ValidationAttemptOutcome(
                        candidate_rank=-1,
                        accepted=False,
                        grounding_score=None,
                        reason=dropped.reason,
                    )
                ]
                if dropped.failing_stage == "validation"
                else []
            ),
        )
        for dropped in result.dropped_objects
    ]

    return PageRunOutcome(
        sample_id=sample.sample_id,
        analysis_mode=mode,
        status="completed",
        panel_count=panel_count,
        panel_sources=panel_sources,
        primary_semantic_label=result.primary_object.semantic_label,
        primary_motion_type=result.primary_object.motion_type.value,
        validation_attempts=[
            ValidationAttemptOutcome(
                candidate_rank=v.candidate_rank,
                accepted=v.accepted,
                grounding_score=v.grounding_score,
                reason=v.reason,
            )
            for v in result.validation_attempts
        ],
        primary_mask_semantics=mask_semantics_outcome_from_result(result.mask_semantics),
        object_outcomes=object_outcomes,
        render_summary=render_summary_from_result(
            result.render, seam_artifact_suspected=_seam_artifact_suspected(result.render)
        ),
        schema_version=CURRENT_SCHEMA_VERSION,
    )


def _seam_artifact_suspected(render: RenderResult) -> bool | None:
    """Decode the real encoded output and run the Phase 9 seam-artifact check on it -- `None`

    (not `False`) when the file can't be decoded or has too few frames, so an honest "not
    computed" stays distinguishable from a real negative result. Failures here are swallowed
    (this is a QA signal, not a pipeline-correctness gate) -- a decode problem must never fail
    an otherwise-successful pipeline run.
    """
    try:
        frames = _decode_frames(render.output_path)
        report = detect_seam_like_artifacts(frames)
    except Exception:  # noqa: BLE001 -- a QA-signal decode failure must not affect the outcome
        return None
    return report.seam_suspected if report is not None else None


def run_nondeterminism_check(
    sample: EvalSample, config: Any, vlm_client: Any, run_count: int
) -> NondeterminismSummary:
    records: list[RepeatedRunRecord] = []
    for i in range(run_count):
        try:
            plan = analyze_page(Path(sample.image_path), vlm_client, config=config)
        except PipelineStageError:
            records.append(
                RepeatedRunRecord(
                    sample_id=sample.sample_id, run_index=i, outcome="static_or_unusable"
                )
            )
            continue
        primaries = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
        primary = primaries[0] if primaries else None
        records.append(
            RepeatedRunRecord(
                sample_id=sample.sample_id,
                run_index=i,
                outcome="usable",
                primary_semantic_label=primary.semantic_label if primary else None,
                primary_motion_type=primary.motion_type.value if primary else None,
                object_count=len(plan.objects),
            )
        )
    return summarize_repeated_runs(records)
