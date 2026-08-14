"""Independent panel extraction, processing, outputs and page manifests.

This module is intentionally an orchestration boundary. It detects panels and creates scene
crops, then delegates every crop's actual animation to the existing ``run_pipeline`` stages.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis import VLMClient, analyze_page, detect_panels
from manga_animation.analysis.panels import derive_scene_crop_bbox
from manga_animation.core.config import PipelineConfig
from manga_animation.grounding import GroundingClient
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.orchestrator import (
    DroppedObjectResult,
    _animate_objects,
    _composite_and_render,
    _ground_objects,
    _mask_semantics_objects,
    _reconstruct_objects,
    _segment_objects,
    _select_objects,
    _select_primary,
    _validate_objects,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    MaskSemanticResult,
    PanelStatus,
    PanelUnit,
    PipelineStageError,
    SegmentationResult,
    ValidationResult,
)
from manga_animation.reconstruction import ReconstructionClient
from manga_animation.schemas.animation_plan import AnimationPlan, MotionType, ObjectPlan
from manga_animation.segmentation import SegmentationClient


@dataclass
class PagePanelsResult:
    """Results for all detected panels, including panels that did not render."""

    page_id: str
    source_image: Path
    manifest_path: Path
    panels: list[PanelUnit]


_SAFE_REJECTION_STAGES = {"grounding", "validation", "segmentation", "mask_semantics"}


def _failure_status(stage: str) -> PanelStatus:
    """Map a failing stage to the panel's status: safe model-gate rejections are REJECTED,
    everything else (including analysis) is ERROR -- same mapping the pre-Phase-14 panel
    runner used.
    """
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


def run_page_panels(
    image_path: Path,
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
) -> PagePanelsResult:
    """Detect and process every panel independently on its scene-crop canvas.

    Panel processing is **stage-level** (Phase 14, docs/decisions/0020-stage-level-model-
    lifecycle.md): each model-backed stage loads its client once, processes every eligible
    panel, then deterministically releases the client via `ModelStage` before the next stage
    loads its own model. The four model families never co-reside, and a panel failure in one
    stage cannot leave a model resident to poison later panels (the old per-panel load/unload
    path leaked Qwen's ~16 GiB until an opportunistic `gc.collect()`, racing the next load into
    a CUDA OOM -- reproduced on a real 2xT4 Kaggle run before this change).

    A panel failure is recorded and processing continues. The manifest is written after each
    stage, making completed PASS/STATIC panels reusable on a later invocation.
    """
    image_path = image_path.resolve()
    page_id = image_path.stem
    page_dir = out_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "page_manifest.json"
    existing = _load_existing_manifest(manifest_path)
    started_at = time.perf_counter()

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
    panel_started_at: dict[str, float] = {}
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

    def write_manifest() -> None:
        _write_manifest(manifest_path, page_id, image_path, panels, started_at=started_at)

    def finalize(panel: PanelUnit, status: PanelStatus, stage: str, reason: str) -> None:
        _set_failure(panel, status, stage, reason)
        start = panel_started_at.get(panel.panel_id)
        if start is not None:
            panel.metrics["runtime_s"] = round(time.perf_counter() - start, 6)
        panel_started_at.pop(panel.panel_id, None)

    # -------------------------------------------------------------------------------------
    # Stage 1: panel/scene analysis -- the VLM processes every eligible panel, then releases.
    # -------------------------------------------------------------------------------------
    plans: dict[str, AnimationPlan] = {}
    failed: set[str] = set()
    with ModelStage(vlm_client, name="analysis"):
        for panel in panels:
            if panel.status in ("PASS", "STATIC"):
                continue  # resumed from an earlier manifest
            panel_started_at[panel.panel_id] = time.perf_counter()
            try:
                plan = analyze_page(
                    panel.scene_crop_path,
                    vlm_client,
                    config=config,
                    allow_all_static=True,
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                failed.add(panel.panel_id)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                failed.add(panel.panel_id)
                continue
            if not any(obj.motion_type != MotionType.STATIC for obj in plan.objects):
                panel.status = "STATIC"
                panel.metrics["runtime_s"] = round(
                    time.perf_counter() - panel_started_at.pop(panel.panel_id), 6
                )
                continue
            plans[panel.panel_id] = plan
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 2: grounding -- Grounding DINO processes every eligible panel, then releases.
    # -------------------------------------------------------------------------------------
    objects_by_panel: dict[str, list[ObjectPlan]] = {}
    bbox_by_object_by_panel: dict[str, dict[str, BBoxPx]] = {}
    primary_by_panel: dict[str, ObjectPlan] = {}
    candidates_by_panel: dict[str, dict[str, list[GroundingResult]]] = {}
    dropped_by_panel: dict[str, list[DroppedObjectResult]] = {}
    with ModelStage(grounding_client, name="grounding"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in plans or panel_id in failed:
                continue  # resumed/STATIC, or already failed at analysis
            try:
                plan = plans[panel_id]
                primary = _select_primary(plan, str(panel.scene_crop_path))
                objects, bbox_by_object = _select_objects(
                    plan, primary, crops[panel_id].shape[:2]
                )
                grounded, dropped = _ground_objects(
                    crops[panel_id],
                    objects,
                    grounding_client,
                    bbox_by_object,
                    primary.object_id,
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                failed.add(panel_id)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                failed.add(panel_id)
                continue
            objects_by_panel[panel_id] = objects
            bbox_by_object_by_panel[panel_id] = bbox_by_object
            primary_by_panel[panel_id] = primary
            candidates_by_panel[panel_id] = grounded
            dropped_by_panel[panel_id] = dropped
    write_manifest()
    # -------------------------------------------------------------------------------------
    # Stage 3: target validation -- the VLM validates every eligible panel, then releases.
    # -------------------------------------------------------------------------------------
    validation_by_panel: dict[str, dict[str, list[ValidationResult]]] = {}
    accepted_by_panel: dict[str, dict[str, GroundingResult]] = {}
    with ModelStage(vlm_client, name="validation"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in candidates_by_panel or panel_id in failed:
                continue
            try:
                attempts, accepted, dropped = _validate_objects(
                    crops[panel_id],
                    objects_by_panel[panel_id],
                    candidates_by_panel[panel_id],
                    vlm_client,
                    bbox_by_object_by_panel[panel_id],
                    primary_by_panel[panel_id].object_id,
                    logical_panel_bbox_px=panel.panel_bbox,
                    neighboring_panel_bboxes=panel_bboxes,
                    global_origin=(panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0),
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                failed.add(panel_id)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                failed.add(panel_id)
                continue
            validation_by_panel[panel_id] = attempts
            accepted_by_panel[panel_id] = accepted
            dropped_by_panel[panel_id].extend(dropped)
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 4: segmentation -- SAM processes every eligible panel, then releases.
    # -------------------------------------------------------------------------------------
    animated_by_panel: dict[str, list[ObjectPlan]] = {}
    segmentation_by_panel: dict[str, dict[str, SegmentationResult]] = {}
    with ModelStage(segmentation_client, name="segmentation"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in accepted_by_panel or panel_id in failed:
                continue
            try:
                animated, seg, dropped = _segment_objects(
                    crops[panel_id],
                    objects_by_panel[panel_id],
                    accepted_by_panel[panel_id],
                    segmentation_client,
                    primary_by_panel[panel_id].object_id,
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                failed.add(panel_id)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                failed.add(panel_id)
                continue
            animated_by_panel[panel_id] = animated
            segmentation_by_panel[panel_id] = seg
            dropped_by_panel[panel_id].extend(dropped)
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 5: semantic mask validation -- the VLM processes every eligible panel, releases.
    # -------------------------------------------------------------------------------------
    mask_semantics_by_panel: dict[str, dict[str, MaskSemanticResult]] = {}
    if config.enable_semantic_mask_validation:
        with ModelStage(vlm_client, name="mask_semantics"):
            for panel in panels:
                panel_id = panel.panel_id
                if panel_id not in animated_by_panel or panel_id in failed:
                    continue
                try:
                    kept, seg, msv, dropped = _mask_semantics_objects(
                        crops[panel_id],
                        animated_by_panel[panel_id],
                        segmentation_by_panel[panel_id],
                        vlm_client,
                        config,
                        primary_by_panel[panel_id].object_id,
                    )
                except PipelineStageError as exc:
                    finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                    failed.add(panel_id)
                    continue
                except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                    finalize(panel, "ERROR", type(exc).__name__, str(exc))
                    failed.add(panel_id)
                    continue
                animated_by_panel[panel_id] = kept
                segmentation_by_panel[panel_id] = seg
                mask_semantics_by_panel[panel_id] = msv
                dropped_by_panel[panel_id].extend(dropped)
        write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 6: animation + reconstruction + compositing + rendering. The CV work is CPU-only;
    # LaMa is loaded once for the whole stage and kept resident only while eligible panels'
    # motion-revealed holes are filled (its own reconstruction stage), then released.
    # -------------------------------------------------------------------------------------
    with ModelStage(reconstruction_client, name="reconstruction"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in animated_by_panel or panel_id in failed:
                continue
            panel_started_at.setdefault(panel_id, time.perf_counter())
            try:
                layers, layers_by_object = _animate_objects(
                    crops[panel_id],
                    animated_by_panel[panel_id],
                    segmentation_by_panel[panel_id],
                    bbox_by_object_by_panel[panel_id],
                    plans[panel_id],
                )
                reconstructions = _reconstruct_objects(
                    crops[panel_id],
                    animated_by_panel[panel_id],
                    segmentation_by_panel[panel_id],
                    layers_by_object,
                    reconstruction_client,
                    config,
                )
                render_result = _composite_and_render(
                    crops[panel_id],
                    layers,
                    reconstructions,
                    plans[panel_id],
                    page_dir,
                    config,
                    video_filename=f"{panel_id}.mp4",
                    frames_dir=page_dir / "frames" / panel_id,
                )
                panel.status = "PASS"
                panel.output_video = render_result.output_path
                panel.metrics["frame_count"] = render_result.frame_count
                panel.metrics["runtime_s"] = round(
                    time.perf_counter() - panel_started_at.pop(panel_id), 6
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
    write_manifest()

    return PagePanelsResult(
        page_id=page_id,
        source_image=image_path,
        manifest_path=manifest_path,
        panels=panels,
    )
