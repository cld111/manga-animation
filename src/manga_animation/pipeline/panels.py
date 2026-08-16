"""Independent panel extraction, processing, outputs and page manifests.

This module is intentionally an orchestration boundary. It detects panels and creates scene
crops, then delegates every crop's actual animation to the existing ``run_pipeline`` stages.

Phase 18.3 architecture: the pipeline has NO Qwen analysis stage. Every panel is processed
with the same candidate label list: grounding (DINO) -> segmentation (SAM) -> the pipeline's
single VLM stage (object description: full image + bbox coordinates) -> animation planning ->
animation -> reconstruction -> compositing -> rendering. Qwen is loaded exactly once for the
whole page (one ModelStage) and processes every panel's candidates there.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis import VLMClient, detect_panels
from manga_animation.analysis.panels import derive_scene_crop_bbox
from manga_animation.core.config import PipelineConfig
from manga_animation.grounding import GroundingClient
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.orchestrator import (
    DEFAULT_ANIMATION_LABELS,
    DroppedObjectResult,
    _animate_objects,
    _composite_and_render,
    _describe_candidates,
    _ground_labels,
    _reconstruct_objects,
    _segment_candidates,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    ObjectDescriptionResult,
    PanelStatus,
    PanelUnit,
    PipelineStageError,
    SegmentationResult,
)
from manga_animation.reconstruction import ReconstructionClient
from manga_animation.schemas.animation_plan import ObjectPlan
from manga_animation.segmentation import SegmentationClient


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
    """Detect and process every panel independently on its scene-crop canvas.

    Stage-level model lifecycle (ADR 0020): each model-backed stage loads its client once,
    processes every eligible panel, then deterministically releases it. In the Phase 18.4
    architecture the VLM is loaded exactly ONCE for the whole page -- in the object-
    description stage, which runs BEFORE segmentation (DINO -> Qwen -> SAM): SAM segments
    only the bboxes that earned an action description -- and no analysis stage exists at
    all. A panel failure is recorded and processing continues; the manifest is written
    after each stage so completed PASS/STATIC panels are reusable on a later invocation.
    """
    image_path = image_path.resolve()
    page_id = image_path.stem
    page_dir = out_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "page_manifest.json"
    existing = _load_existing_manifest(manifest_path)
    started_at = time.perf_counter()
    active_labels = list(labels or DEFAULT_ANIMATION_LABELS)

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

    def crop_local_panel_bbox(panel: PanelUnit, crop: np.ndarray) -> BBoxPx:
        """The panel's logical bbox translated into its scene crop's local coordinates --
        grounding/description run on the crop canvas, so the region argument must be
        crop-local (the old analysis flow derived it from the plan; the Phase 18.3 flow
        derives it from the crop geometry directly)."""
        ox, oy = panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0
        h, w = crop.shape[0], crop.shape[1]
        return BBoxPx(
            x0=max(0, panel.panel_bbox.x0 - ox),
            y0=max(0, panel.panel_bbox.y0 - oy),
            x1=min(w, panel.panel_bbox.x1 - ox),
            y1=min(h, panel.panel_bbox.y1 - oy),
        )

    def finalize(panel: PanelUnit, status: PanelStatus, stage: str, reason: str) -> None:
        _set_failure(panel, status, stage, reason)
        start = panel_started_at.get(panel.panel_id)
        if start is not None:
            panel.metrics["runtime_s"] = round(time.perf_counter() - start, 6)
        panel_started_at.pop(panel.panel_id, None)

    # -------------------------------------------------------------------------------------
    # Stage 1: grounding -- DINO processes every eligible panel, then releases.
    # -------------------------------------------------------------------------------------
    candidates_by_panel: dict[str, dict[str, list[GroundingResult]]] = {}
    plan_by_object_by_panel: dict[str, dict[str, ObjectPlan]] = {}
    dropped_by_panel: dict[str, list[DroppedObjectResult]] = {}
    with ModelStage(grounding_client, name="grounding"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel.status in ("PASS", "STATIC"):
                continue  # resumed from an earlier manifest
            panel_started_at[panel.panel_id] = time.perf_counter()
            try:
                plans, grounded, dropped = _ground_labels(
                    crops[panel_id],
                    active_labels,
                    grounding_client,
                    panel_bbox_px=crop_local_panel_bbox(panel, crops[panel_id]),
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                continue
            candidates_by_panel[panel_id] = grounded
            plan_by_object_by_panel[panel_id] = {p.object_id: p for p in plans}
            dropped_by_panel[panel_id] = dropped
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 2: object description -- THE page's single VLM stage. Qwen loads once and
    # processes every eligible panel's grounded candidates (ONE call per panel with the
    # crop + ALL its bboxes), then releases. Runs BEFORE segmentation (Phase 18.4
    # ordering: DINO -> Qwen -> SAM).
    # -------------------------------------------------------------------------------------
    descriptions_by_panel: dict[str, dict[tuple[str, int], ObjectDescriptionResult]] = {}
    with ModelStage(vlm_client, name="object_description"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in candidates_by_panel:
                continue  # failed at grounding
            try:
                desc, dropped = _describe_candidates(
                    crops[panel_id],
                    candidates_by_panel[panel_id],
                    plan_by_object_by_panel[panel_id],
                    vlm_client,
                    config,
                    panel_bbox_px=crop_local_panel_bbox(panel, crops[panel_id]),
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                continue
            descriptions_by_panel[panel_id] = desc
            dropped_by_panel[panel_id].extend(dropped)
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 3: segmentation -- SAM processes ONLY the accepted candidates of every
    # eligible panel, then releases (Phase 18.4 ordering: DINO -> Qwen -> SAM).
    # -------------------------------------------------------------------------------------
    segmentation_by_panel: dict[str, dict[tuple[str, int], SegmentationResult]] = {}
    with ModelStage(segmentation_client, name="segmentation"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in descriptions_by_panel:
                continue  # failed at grounding or object description
            accepted_keys = {
                key
                for key, description in descriptions_by_panel[panel_id].items()
                if description.accepted
            }
            try:
                seg, dropped = _segment_candidates(
                    crops[panel_id],
                    candidates_by_panel[panel_id],
                    plan_by_object_by_panel[panel_id],
                    segmentation_client,
                    accepted_keys=accepted_keys,
                )
            except PipelineStageError as exc:
                finalize(panel, _failure_status(exc.stage), exc.stage, exc.detail)
                continue
            except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
                finalize(panel, "ERROR", type(exc).__name__, str(exc))
                continue
            segmentation_by_panel[panel_id] = seg
            dropped_by_panel[panel_id].extend(dropped)
    write_manifest()

    # -------------------------------------------------------------------------------------
    # Stage 4: animation planning (deterministic) + animation + reconstruction + compositing
    # + rendering. The CV work is CPU-only; LaMa is loaded once for the whole stage.
    # -------------------------------------------------------------------------------------
    with ModelStage(reconstruction_client, name="reconstruction"):
        for panel in panels:
            panel_id = panel.panel_id
            if panel_id not in descriptions_by_panel:
                continue  # failed earlier
            panel_started_at.setdefault(panel_id, time.perf_counter())
            try:
                from manga_animation.pipeline.orchestrator import _build_plan

                accepted = []
                for (object_id, rank), description in descriptions_by_panel[panel_id].items():
                    if not description.accepted:
                        continue
                    if (object_id, rank) not in segmentation_by_panel[panel_id]:
                        continue  # accepted by the VLM but dropped at segmentation
                    accepted.append(
                        (
                            object_id,
                            rank,
                            plan_by_object_by_panel[panel_id][object_id],
                            candidates_by_panel[panel_id][object_id][rank],
                            segmentation_by_panel[panel_id][(object_id, rank)],
                            description,
                        )
                    )
                plan, primary, kept = _build_plan(
                    panel.scene_crop_path,
                    crops[panel_id].shape[:2],
                    config,
                    accepted=accepted,
                    global_origin=(panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0),
                    logical_panel_bbox_px=panel.panel_bbox,
                    neighboring_panel_bboxes=panel_bboxes,
                )
                animated_objects = [item[0] for item in kept]
                segmentation_by_object = {
                    item[0].object_id: item[2] for item in kept
                }
                panel_bbox_px_by_object = {
                    obj.object_id: panel.panel_bbox for obj in animated_objects
                }
                layers, layers_by_object = _animate_objects(
                    crops[panel_id],
                    animated_objects,
                    segmentation_by_object,
                    panel_bbox_px_by_object,
                    plan,
                )
                reconstructions = _reconstruct_objects(
                    crops[panel_id],
                    animated_objects,
                    segmentation_by_object,
                    layers_by_object,
                    reconstruction_client,
                    config,
                )
                render_result = _composite_and_render(
                    crops[panel_id],
                    layers,
                    reconstructions,
                    plan,
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
