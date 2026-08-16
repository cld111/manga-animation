"""Independent panel extraction, processing, outputs and page manifests.

This module is intentionally an orchestration boundary. It detects panels and creates scene
crops, then delegates every crop's actual animation to the existing ``run_pipeline`` stages.

Phase 18.3 architecture: the pipeline has NO Qwen analysis stage. Every panel is processed
with the same candidate label list: grounding (DINO) -> the pipeline's single VLM stage
(object description: full image + bbox coordinates) -> segmentation (SAM, only for accepted
bboxes) -> animation planning -> animation -> reconstruction -> compositing -> rendering.

Phase 18.4 batch mode: `run_pages` processes MANY pages with stage-level model residency
ACROSS pages. Each model loads ONCE, processes every eligible panel of EVERY page, saves its
results, and only then is released and the next model loads. `run_page_panels` is the
single-page convenience wrapper over the same code path.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis import VLMClient, detect_panels
from manga_animation.analysis.panels import derive_scene_crop_bbox
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import get_logger
from manga_animation.grounding import GroundingClient
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.orchestrator import (
    DEFAULT_ANIMATION_LABELS,
    DroppedObjectResult,
    _animate_objects,
    _build_plan,
    _composite_and_render,
    _describe_candidates,
    _ground_labels,
    _reconstruct_objects,
    _segment_candidates,
)
from manga_animation.pipeline.persistence import (
    has_descriptions,
    has_grounding,
    has_segmentation,
    load_descriptions,
    load_grounding,
    load_segmentation,
    save_descriptions,
    save_grounding,
    save_segmentation,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    Layer,
    ObjectDescriptionResult,
    PanelStatus,
    PanelUnit,
    PipelineStageError,
    ReconstructionResult,
    SegmentationResult,
)
from manga_animation.reconstruction import ReconstructionClient
from manga_animation.schemas.animation_plan import AnimationPlan, ObjectPlan
from manga_animation.segmentation import SegmentationClient

logger = get_logger(__name__)


@dataclass
class PagePanelsResult:
    """Results for all detected panels, including panels that did not render."""

    page_id: str
    source_image: Path
    manifest_path: Path
    panels: list[PanelUnit]


_SAFE_REJECTION_STAGES = {"grounding", "segmentation", "object_description"}


def _failure_status(stage: str) -> PanelStatus:
    """Map a failing stage to the panel's status: safe model-gate rejections are REJECTED,
    everything else is ERROR."""
    return "REJECTED" if stage in _SAFE_REJECTION_STAGES else "ERROR"


def _write_manifest(
    manifest_path: Path,
    page_id: str,
    source_image: Path,
    panels: list[PanelUnit],
    *,
    started_at: float,
) -> None:
    payload = {
        "page_id": page_id,
        "source_image": str(source_image),
        "panels": [panel.as_manifest_dict() for panel in panels],
        "performance": {
            "detected_panel_count": len(panels),
            "scene_crop_pixels": sum(
                panel.scene_crop_bbox.width * panel.scene_crop_bbox.height for panel in panels
            ),
            "elapsed_s": round(time.perf_counter() - started_at, 6),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def _existing_resumable_panel(
    existing: dict[str, object], panel: PanelUnit
) -> PanelStatus | None:
    if existing.get("status") == "PASS":
        output = existing.get("output_video")
        if isinstance(output, str) and Path(output).exists():
            return "PASS"
    if existing.get("status") == "STATIC" and panel.scene_crop_path.exists():
        return "STATIC"
    return None


def _load_existing_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(item["panel_id"]): item
        for item in payload.get("panels", [])
        if isinstance(item, dict) and "panel_id" in item
    }


def _set_failure(panel: PanelUnit, status: PanelStatus, stage: str, reason: str) -> None:
    panel.status = status
    panel.failure_stage = stage
    panel.failure_reason = reason


@dataclass
class _PageRunState:
    """Everything one page accumulates across the batch's model stages.

    Stage outputs live here between stages: a model processes ALL pages (every eligible
    panel of each), saves its results into the corresponding state, and only then is
    released and the next model loads (Phase 18.4 batch residency).
    """

    image_path: Path
    page_id: str
    page_dir: Path
    manifest_path: Path
    existing: dict[str, dict[str, object]]
    panels: list[PanelUnit]
    crops: dict[str, np.ndarray]
    panel_started_at: dict[str, float]
    candidates_by_panel: dict[str, dict[str, list[GroundingResult]]] = field(default_factory=dict)
    plan_by_object_by_panel: dict[str, dict[str, ObjectPlan]] = field(default_factory=dict)
    dropped_by_panel: dict[str, list[DroppedObjectResult]] = field(default_factory=dict)
    descriptions_by_panel: dict[str, dict[tuple[str, int], ObjectDescriptionResult]] = field(
        default_factory=dict
    )
    segmentation_by_panel: dict[str, dict[tuple[str, int], SegmentationResult]] = field(
        default_factory=dict
    )
    plans_by_panel: dict[str, AnimationPlan] = field(default_factory=dict)
    animated_by_panel: dict[str, list[ObjectPlan]] = field(default_factory=dict)
    seg_by_object_by_panel: dict[str, dict[str, SegmentationResult]] = field(default_factory=dict)
    layers_by_panel: dict[str, list[Layer]] = field(default_factory=dict)
    layers_by_object_by_panel: dict[str, dict[str, Layer]] = field(default_factory=dict)
    reconstructions_by_panel: dict[str, dict[str, ReconstructionResult]] = field(
        default_factory=dict
    )


def _prepare_page_state(
    image_path: Path, out_dir: Path, config: PipelineConfig
) -> _PageRunState:
    """Detect panels, write scene crops and build the initial state for one page."""
    image_path = image_path.resolve()
    page_id = image_path.stem
    page_dir = out_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "page_manifest.json"
    existing = _load_existing_manifest(manifest_path)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    page_shape = image.shape[:2]
    candidates = detect_panels(image)
    if not candidates:
        raise PipelineStageError(
            stage="analysis",
            input_ref=str(image_path),
            detail="panel detector returned no usable candidates",
            root_cause="the source image is too small or degenerate for deterministic detection",
            architectural=False,
            proposed_fix="provide a larger source page or use an explicit panel annotation",
        )

    panel_bboxes = tuple(candidate.panel_bbox for candidate in candidates)
    panels: list[PanelUnit] = []
    crops: dict[str, np.ndarray] = {}
    for index, candidate in enumerate(candidates, start=1):
        panel_id = f"panel_{index:03d}"
        panel_bbox = candidate.panel_bbox
        scene_bbox = derive_scene_crop_bbox(
            panel_bbox,
            page_shape,
            neighboring_panel_bboxes=panel_bboxes,
        )
        crop_path = page_dir / "crops" / f"{panel_id}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop = image[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]
        crops[panel_id] = crop
        Image.fromarray(crop).save(crop_path)
        panel = PanelUnit(
            page_id=page_id,
            panel_id=panel_id,
            panel_order=index,
            panel_bbox=panel_bbox,
            scene_crop_bbox=scene_bbox,
            source_page=image_path,
            scene_crop_path=crop_path,
        )
        panel.metrics.update(
            {
                "scene_crop_width": scene_bbox.width,
                "scene_crop_height": scene_bbox.height,
                "scene_crop_pixels": scene_bbox.width * scene_bbox.height,
            }
        )
        panels.append(panel)

        resumed = _existing_resumable_panel(existing.get(panel_id, {}), panel)
        if resumed is not None:
            panel.status = resumed
            output = existing[panel_id].get("output_video")
            panel.output_video = Path(output) if isinstance(output, str) else None

    return _PageRunState(
        image_path=image_path,
        page_id=page_id,
        page_dir=page_dir,
        manifest_path=manifest_path,
        existing=existing,
        panels=panels,
        crops=crops,
        panel_started_at={},
    )


def _crop_local_panel_bbox(state: _PageRunState, panel: PanelUnit) -> BBoxPx:
    """The panel's logical bbox translated into its scene crop's local coordinates --
    grounding/description run on the crop canvas, so the region argument must be
    crop-local (the Phase 18.3 flow derives it from the crop geometry directly)."""
    crop = state.crops[panel.panel_id]
    ox, oy = panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0
    h, w = crop.shape[0], crop.shape[1]
    return BBoxPx(
        x0=max(0, panel.panel_bbox.x0 - ox),
        y0=max(0, panel.panel_bbox.y0 - oy),
        x1=min(w, panel.panel_bbox.x1 - ox),
        y1=min(h, panel.panel_bbox.y1 - oy),
    )


def _finalize(
    state: _PageRunState, panel: PanelUnit, status: PanelStatus, stage: str, reason: str
) -> None:
    _set_failure(panel, status, stage, reason)
    start = state.panel_started_at.get(panel.panel_id)
    if start is not None:
        panel.metrics["runtime_s"] = round(time.perf_counter() - start, 6)
    state.panel_started_at.pop(panel.panel_id, None)


def _write_all_manifests(states: list[_PageRunState]) -> None:
    for state in states:
        _write_manifest(
            state.manifest_path,
            state.page_id,
            state.image_path,
            state.panels,
            started_at=0.0,
        )


def run_pages(
    image_paths: Sequence[Path],
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    labels: Sequence[str] | None = None,
) -> list[PagePanelsResult]:
    """Process MANY pages with stage-level model residency ACROSS pages (Phase 18.4 batch).

    Each model-backed stage loads its client ONCE, processes every eligible panel of EVERY
    page (saving its outputs into that page's state), then deterministically releases it
    (ADR 0020). A model never loads per page. The VLM runs exactly ONCE per panel -- the
    object-description stage, before segmentation (DINO -> Qwen -> SAM): SAM segments only
    the bboxes that earned an action description. A panel failure is recorded and
    processing continues; manifests are written after each stage so completed PASS/STATIC
    panels are reusable on a later invocation.

    Stage outputs are ALSO persisted to disk after every model stage (Phase 18.4): each
    page dir receives `grounding.json`, `descriptions.json`, and `segmentation.json` +
    mask `.npz` files. On a later invocation the completed stages are loaded from disk and
    their models are NOT loaded at all -- a killed session resumes from the last completed
    stage instead of re-running DINO/Qwen/SAM from scratch.
    """
    active_labels = list(labels or DEFAULT_ANIMATION_LABELS)
    states = [_prepare_page_state(path, out_dir, config) for path in image_paths]
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Stage 1: grounding -- DINO processes every eligible panel of EVERY page, then releases.
    # A completed grounding stage is restored from disk (no model load at all).
    # -------------------------------------------------------------------------------------
    resume_grounding = [s for s in states if has_grounding(s.page_dir)]
    for state in resume_grounding:
        (
            state.candidates_by_panel,
            state.plan_by_object_by_panel,
            state.dropped_by_panel,
        ) = load_grounding(state.page_dir)
        logger.info(
            "grounding: restored %d panel(s) from %s (no DINO load)",
            len(state.candidates_by_panel),
            state.page_dir / "grounding.json",
        )
    pending_grounding = [s for s in states if not has_grounding(s.page_dir)]
    if pending_grounding:
        with ModelStage(grounding_client, name="grounding"):
            for state in pending_grounding:
                for panel in state.panels:
                    panel_id = panel.panel_id
                    if panel.status in ("PASS", "STATIC"):
                        continue  # resumed from an earlier manifest
                    state.panel_started_at[panel.panel_id] = time.perf_counter()
                    try:
                        plans, grounded, dropped = _ground_labels(
                            state.crops[panel_id],
                            active_labels,
                            grounding_client,
                            panel_bbox_px=_crop_local_panel_bbox(state, panel),
                        )
                    except PipelineStageError as exc:
                        _finalize(
                            state, panel, _failure_status(exc.stage), exc.stage, exc.detail
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
                        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
                        continue
                    state.candidates_by_panel[panel_id] = grounded
                    state.plan_by_object_by_panel[panel_id] = {p.object_id: p for p in plans}
                    state.dropped_by_panel[panel_id] = dropped
        for state in pending_grounding:
            save_grounding(
                state.page_dir,
                state.candidates_by_panel,
                state.plan_by_object_by_panel,
                state.dropped_by_panel,
            )
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Stage 2: object description -- THE single VLM stage. Qwen loads once and processes
    # every eligible panel's grounded candidates of EVERY page, then releases. Runs BEFORE
    # segmentation (Phase 18.4 ordering: DINO -> Qwen -> SAM). A completed description stage
    # is restored from disk (no Qwen load at all).
    # -------------------------------------------------------------------------------------
    resume_descriptions = [s for s in states if has_descriptions(s.page_dir)]
    for state in resume_descriptions:
        state.descriptions_by_panel = load_descriptions(state.page_dir)
        logger.info(
            "object description: restored %d panel(s) from %s (no Qwen load)",
            len(state.descriptions_by_panel),
            state.page_dir / "descriptions.json",
        )
    pending_descriptions = [s for s in states if not has_descriptions(s.page_dir)]
    if pending_descriptions:
        with ModelStage(vlm_client, name="object_description"):
            for state in pending_descriptions:
                for panel in state.panels:
                    panel_id = panel.panel_id
                    if panel_id not in state.candidates_by_panel:
                        continue  # failed at grounding
                    try:
                        desc, dropped = _describe_candidates(
                            state.crops[panel_id],
                            state.candidates_by_panel[panel_id],
                            state.plan_by_object_by_panel[panel_id],
                            vlm_client,
                            config,
                            panel_bbox_px=_crop_local_panel_bbox(state, panel),
                        )
                    except PipelineStageError as exc:
                        _finalize(
                            state, panel, _failure_status(exc.stage), exc.stage, exc.detail
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
                        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
                        continue
                    state.descriptions_by_panel[panel_id] = desc
                    state.dropped_by_panel[panel_id].extend(dropped)
        for state in pending_descriptions:
            save_descriptions(state.page_dir, state.descriptions_by_panel)
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Stage 3: segmentation -- SAM processes ONLY the accepted candidates of every eligible
    # panel of EVERY page, then releases (Phase 18.4 ordering: DINO -> Qwen -> SAM). A
    # completed segmentation stage is restored from disk (no SAM load at all).
    # -------------------------------------------------------------------------------------
    resume_segmentation = [s for s in states if has_segmentation(s.page_dir)]
    for state in resume_segmentation:
        state.segmentation_by_panel = load_segmentation(state.page_dir)
        logger.info(
            "segmentation: restored %d panel(s) from %s (no SAM load)",
            len(state.segmentation_by_panel),
            state.page_dir / "segmentation.json",
        )
    pending_segmentation = [s for s in states if not has_segmentation(s.page_dir)]
    if pending_segmentation:
        with ModelStage(segmentation_client, name="segmentation"):
            for state in pending_segmentation:
                for panel in state.panels:
                    panel_id = panel.panel_id
                    if panel_id not in state.descriptions_by_panel:
                        continue  # failed at grounding or object description
                    accepted_keys = {
                        key
                        for key, description in state.descriptions_by_panel[panel_id].items()
                        if description.accepted
                    }
                    try:
                        seg, dropped = _segment_candidates(
                            state.crops[panel_id],
                            state.candidates_by_panel[panel_id],
                            state.plan_by_object_by_panel[panel_id],
                            segmentation_client,
                            accepted_keys=accepted_keys,
                        )
                    except PipelineStageError as exc:
                        _finalize(
                            state, panel, _failure_status(exc.stage), exc.stage, exc.detail
                        )
                        continue
                    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
                        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
                        continue
                    state.segmentation_by_panel[panel_id] = seg
                    state.dropped_by_panel[panel_id].extend(dropped)
        for state in pending_segmentation:
            save_segmentation(state.page_dir, state.segmentation_by_panel)
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Stage 4: animation planning (deterministic) + animation + reconstruction (LaMa loaded
    # once for EVERY page) + compositing + rendering. CV work is CPU-only.
    # -------------------------------------------------------------------------------------
    with ModelStage(reconstruction_client, name="reconstruction"):
        for state in states:
            for panel in state.panels:
                panel_id = panel.panel_id
                if panel_id not in state.descriptions_by_panel:
                    continue  # failed earlier
                state.panel_started_at.setdefault(panel_id, time.perf_counter())
                try:
                    accepted = []
                    for (object_id, rank), description in state.descriptions_by_panel[
                        panel_id
                    ].items():
                        if not description.accepted:
                            continue
                        if (object_id, rank) not in state.segmentation_by_panel[panel_id]:
                            continue  # accepted by the VLM but dropped at segmentation
                        accepted.append(
                            (
                                object_id,
                                rank,
                                state.plan_by_object_by_panel[panel_id][object_id],
                                state.candidates_by_panel[panel_id][object_id][rank],
                                state.segmentation_by_panel[panel_id][(object_id, rank)],
                                description,
                            )
                        )
                    plan, primary, kept = _build_plan(
                        panel.scene_crop_path,
                        state.crops[panel_id].shape[:2],
                        config,
                        accepted=accepted,
                        global_origin=(panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0),
                        logical_panel_bbox_px=panel.panel_bbox,
                        neighboring_panel_bboxes=tuple(
                            p.panel_bbox for p in state.panels
                        ),
                    )
                    animated_objects = [item[0] for item in kept]
                    segmentation_by_object = {item[0].object_id: item[2] for item in kept}
                    panel_bbox_px_by_object = {
                        obj.object_id: panel.panel_bbox for obj in animated_objects
                    }
                    layers, layers_by_object = _animate_objects(
                        state.crops[panel_id],
                        animated_objects,
                        segmentation_by_object,
                        panel_bbox_px_by_object,
                        plan,
                    )
                    reconstructions = _reconstruct_objects(
                        state.crops[panel_id],
                        animated_objects,
                        segmentation_by_object,
                        layers_by_object,
                        reconstruction_client,
                        config,
                    )
                    state.plans_by_panel[panel_id] = plan
                    state.animated_by_panel[panel_id] = animated_objects
                    state.seg_by_object_by_panel[panel_id] = segmentation_by_object
                    state.layers_by_panel[panel_id] = layers
                    state.layers_by_object_by_panel[panel_id] = layers_by_object
                    state.reconstructions_by_panel[panel_id] = reconstructions
                except PipelineStageError as exc:
                    _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
                    continue
                except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                    _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
                    continue
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Stage 5: render every planned panel (CPU), update statuses, write manifests.
    # -------------------------------------------------------------------------------------
    for state in states:
        for panel in state.panels:
            panel_id = panel.panel_id
            if panel_id not in state.plans_by_panel:
                continue  # failed earlier
            try:
                render_result = _composite_and_render(
                    state.crops[panel_id],
                    state.layers_by_panel[panel_id],
                    state.reconstructions_by_panel[panel_id],
                    state.plans_by_panel[panel_id],
                    state.page_dir,
                    config,
                    video_filename=f"{panel_id}.mp4",
                    frames_dir=state.page_dir / "frames" / panel_id,
                )
                panel.status = "PASS"
                panel.output_video = render_result.output_path
                panel.metrics["frame_count"] = render_result.frame_count
                panel.metrics["runtime_s"] = round(
                    time.perf_counter() - state.panel_started_at.pop(panel_id), 6
                )
            except PipelineStageError as exc:
                _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
    _write_all_manifests(states)

    return [
        PagePanelsResult(
            page_id=state.page_id,
            source_image=state.image_path,
            manifest_path=state.manifest_path,
            panels=state.panels,
        )
        for state in states
    ]


def run_page_panels(
    image_path: Path,
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    labels: Sequence[str] | None = None,
) -> PagePanelsResult:
    """Single-page convenience wrapper over `run_pages` (Phase 18.4 batch residency:
    each model loads once per call -- here, once for this one page)."""
    results = run_pages(
        [image_path],
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=segmentation_client,
        reconstruction_client=reconstruction_client,
        out_dir=out_dir,
        labels=labels,
    )
    return results[0]
