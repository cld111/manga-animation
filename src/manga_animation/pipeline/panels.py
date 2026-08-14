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
from manga_animation.pipeline.orchestrator import run_pipeline
from manga_animation.pipeline.types import (
    PanelStatus,
    PanelUnit,
    PipelineStageError,
)
from manga_animation.reconstruction import ReconstructionClient
from manga_animation.schemas.animation_plan import MotionType
from manga_animation.segmentation import SegmentationClient


@dataclass
class PagePanelsResult:
    """Results for all detected panels, including panels that did not render."""

    page_id: str
    source_image: Path
    manifest_path: Path
    panels: list[PanelUnit]


_SAFE_REJECTION_STAGES = {"grounding", "validation", "segmentation", "mask_semantics"}


def _unload(client: object) -> None:
    unload = getattr(client, "unload", None)
    if callable(unload):
        unload()


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

    A panel failure is recorded and processing continues. The manifest is written after each
    panel, making completed PASS/STATIC panels reusable on a later invocation.
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
        Image.fromarray(image[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]).save(
            crop_path
        )
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
            _write_manifest(
                manifest_path, page_id, image_path, panels, started_at=started_at
            )
            continue

        panel_started_at = time.perf_counter()
        try:
            try:
                plan = analyze_page(
                    crop_path,
                    vlm_client,
                    config=config,
                    allow_all_static=True,
                )
            finally:
                _unload(vlm_client)

            if not any(obj.motion_type != MotionType.STATIC for obj in plan.objects):
                panel.status = "STATIC"
                panel.metrics["runtime_s"] = round(time.perf_counter() - panel_started_at, 6)
                _write_manifest(
                    manifest_path, page_id, image_path, panels, started_at=started_at
                )
                continue

            run_result = run_pipeline(
                crop_path,
                config,
                vlm_client=vlm_client,
                grounding_client=grounding_client,
                segmentation_client=segmentation_client,
                reconstruction_client=reconstruction_client,
                out_dir=page_dir,
                plan=plan,
                global_origin=(scene_bbox.x0, scene_bbox.y0),
                logical_panel_bbox_px=panel_bbox,
                neighboring_panel_bboxes=panel_bboxes,
                video_filename=f"{panel_id}.mp4",
                frames_dir=page_dir / "frames" / panel_id,
            )
            panel.status = "PASS"
            panel.output_video = run_result.render.output_path
            panel.metrics["frame_count"] = run_result.render.frame_count
        except PipelineStageError as exc:
            status: PanelStatus = "REJECTED" if exc.stage in _SAFE_REJECTION_STAGES else "ERROR"
            _set_failure(panel, status, exc.stage, exc.detail)
        except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
            _set_failure(panel, "ERROR", type(exc).__name__, str(exc))

        panel.metrics["runtime_s"] = round(time.perf_counter() - panel_started_at, 6)
        _write_manifest(manifest_path, page_id, image_path, panels, started_at=started_at)

    _write_manifest(manifest_path, page_id, image_path, panels, started_at=started_at)
    return PagePanelsResult(
        page_id=page_id,
        source_image=image_path,
        manifest_path=manifest_path,
        panels=panels,
    )
