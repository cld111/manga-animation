"""Per-stage result persistence for the batch pipeline (Phase 18.4).

Each model-backed stage of `run_pages` saves its outputs to disk BEFORE the model is
released, and a later invocation of `run_pages` on the same pages loads the saved results
and skips the completed stage entirely -- so a killed session or a crashed run resumes
from the last completed stage instead of re-running DINO/Qwen/SAM from scratch (their
outputs previously lived only in process memory and were lost on any restart).

Storage layout, per page (`page_dir` is `out_dir / <page_id>`):

    page_dir/grounding.json        candidates + object plans + drops per panel
    page_dir/descriptions.json     per-candidate object descriptions per panel
    page_dir/segmentation.json     mask metadata (bbox, model_id, iou) per candidate
    page_dir/segmentation/<panel>_<key>.npz   the actual uint8 mask array per candidate

Serialization conventions:

- Panel-level maps use `panel_id` string keys, as in `_PageRunState`.
- Candidate keys `(object_id, rank)` are flattened to `"<object_id>|<rank>"`.
- Pydantic models (`ObjectPlan`, `MotionSpec`) dump/validate via their own
  `model_dump(mode="json")`/`model_validate`, keeping them byte-stable.
- Raw `BBoxPx`/dataclass fields are dumped as plain dicts.
- Masks are saved as `.npz` (uint8, uncompressed -- fast, no encode/decode surprises).

Stage files are authoritative checkpoints, not caches: once written, a stage is never
re-run for that page. A panel that failed at an earlier stage simply has no entry in the
stage file, matching `_PageRunState`'s "not in dict = failed earlier" convention.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from manga_animation.pipeline.orchestrator import DroppedObjectResult
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    ObjectDescriptionResult,
    SegmentationResult,
)
from manga_animation.schemas.animation_plan import ObjectPlan

_GROUNDING_FILE = "grounding.json"
_DESCRIPTIONS_FILE = "descriptions.json"
_SEGMENTATION_FILE = "segmentation.json"
_SEGMENTATION_MASK_DIR = "segmentation"


def _candidate_key(object_id: str, rank: int) -> str:
    return f"{object_id}|{rank}"


def _split_candidate_key(key: str) -> tuple[str, int]:
    object_id, _, rank = key.rpartition("|")
    return object_id, int(rank)


def _bbox_to_dict(bbox: BBoxPx) -> dict[str, Any]:
    return {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1, "score": bbox.score}


def _bbox_from_dict(data: dict[str, Any]) -> BBoxPx:
    return BBoxPx(
        x0=int(data["x0"]),
        y0=int(data["y0"]),
        x1=int(data["x1"]),
        y1=int(data["y1"]),
        score=data.get("score"),
    )


def _grounding_result_to_dict(result: GroundingResult) -> dict[str, Any]:
    return {
        "object_id": result.object_id,
        "bbox": _bbox_to_dict(result.bbox),
        "model_id": result.model_id,
    }


def _grounding_result_from_dict(data: dict[str, Any]) -> GroundingResult:
    return GroundingResult(
        object_id=str(data["object_id"]),
        bbox=_bbox_from_dict(data["bbox"]),
        model_id=str(data["model_id"]),
    )


def _dropped_to_dict(dropped: DroppedObjectResult) -> dict[str, Any]:
    return {
        "object_plan": dropped.object_plan.model_dump(mode="json"),
        "failing_stage": dropped.failing_stage,
        "reason": dropped.reason,
    }


def _dropped_from_dict(data: dict[str, Any]) -> DroppedObjectResult:
    return DroppedObjectResult(
        object_plan=ObjectPlan.model_validate(data["object_plan"]),
        failing_stage=data["failing_stage"],
        reason=data["reason"],
    )


def _description_to_dict(description: ObjectDescriptionResult) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field in fields(description):
        value = getattr(description, field.name)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if field.name == "motion_spec" and value is not None:
            value = value.model_dump(mode="json")
        data[field.name] = value
    return data


def _description_from_dict(data: dict[str, Any]) -> ObjectDescriptionResult:
    values: dict[str, Any] = {}
    for field in fields(ObjectDescriptionResult):
        if field.name not in data:
            continue
        value = data[field.name]
        if field.name == "motion_spec" and value is not None:
            from manga_animation.schemas.animation_plan import MotionSpec

            value = MotionSpec.model_validate(value)
        elif str(field.type).startswith("tuple"):
            value = tuple(value) if value is not None else ()
        values[field.name] = value
    return ObjectDescriptionResult(**values)


# -------------------------------------------------------------------------------------
# Grounding stage
# -------------------------------------------------------------------------------------


def save_grounding(
    page_dir: Path,
    candidates_by_panel: dict[str, dict[str, list[GroundingResult]]],
    plans_by_panel: dict[str, dict[str, ObjectPlan]],
    dropped_by_panel: dict[str, list[DroppedObjectResult]],
) -> None:
    payload: dict[str, Any] = {}
    for panel_id, candidates in candidates_by_panel.items():
        payload[panel_id] = {
            "candidates": {
                object_id: [_grounding_result_to_dict(r) for r in ranked]
                for object_id, ranked in candidates.items()
            },
            "plans": {
                object_id: plan.model_dump(mode="json")
                for object_id, plan in plans_by_panel.get(panel_id, {}).items()
            },
            "dropped": [_dropped_to_dict(d) for d in dropped_by_panel.get(panel_id, [])],
        }
    (page_dir / _GROUNDING_FILE).write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )


def load_grounding(
    page_dir: Path,
) -> tuple[
    dict[str, dict[str, list[GroundingResult]]],
    dict[str, dict[str, ObjectPlan]],
    dict[str, list[DroppedObjectResult]],
]:
    payload = json.loads((page_dir / _GROUNDING_FILE).read_text(encoding="utf-8"))
    candidates: dict[str, dict[str, list[GroundingResult]]] = {}
    plans: dict[str, dict[str, ObjectPlan]] = {}
    dropped: dict[str, list[DroppedObjectResult]] = {}
    for panel_id, panel_data in payload.items():
        candidates[panel_id] = {
            object_id: [_grounding_result_from_dict(d) for d in ranked]
            for object_id, ranked in panel_data.get("candidates", {}).items()
        }
        plans[panel_id] = {
            object_id: ObjectPlan.model_validate(d)
            for object_id, d in panel_data.get("plans", {}).items()
        }
        dropped[panel_id] = [
            _dropped_from_dict(d) for d in panel_data.get("dropped", [])
        ]
    return candidates, plans, dropped


# -------------------------------------------------------------------------------------
# Object description stage
# -------------------------------------------------------------------------------------


def save_descriptions(
    page_dir: Path,
    descriptions_by_panel: dict[str, dict[tuple[str, int], ObjectDescriptionResult]],
) -> None:
    payload: dict[str, Any] = {}
    for panel_id, descriptions in descriptions_by_panel.items():
        payload[panel_id] = {
            _candidate_key(object_id, rank): _description_to_dict(description)
            for (object_id, rank), description in descriptions.items()
        }
    (page_dir / _DESCRIPTIONS_FILE).write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )


def load_descriptions(
    page_dir: Path,
) -> dict[str, dict[tuple[str, int], ObjectDescriptionResult]]:
    payload = json.loads((page_dir / _DESCRIPTIONS_FILE).read_text(encoding="utf-8"))
    result: dict[str, dict[tuple[str, int], ObjectDescriptionResult]] = {}
    for panel_id, descriptions in payload.items():
        result[panel_id] = {
            _split_candidate_key(key): _description_from_dict(data)
            for key, data in descriptions.items()
        }
    return result


# -------------------------------------------------------------------------------------
# Segmentation stage
# -------------------------------------------------------------------------------------


def _mask_path(page_dir: Path, panel_id: str, key: str) -> Path:
    return page_dir / _SEGMENTATION_MASK_DIR / f"{panel_id}__{key}.npz"


def save_segmentation(
    page_dir: Path,
    segmentation_by_panel: dict[str, dict[tuple[str, int], SegmentationResult]],
) -> None:
    mask_dir = page_dir / _SEGMENTATION_MASK_DIR
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {}
    for panel_id, segmentation in segmentation_by_panel.items():
        for (object_id, rank), result in segmentation.items():
            key = _candidate_key(object_id, rank)
            np.savez_compressed(
                _mask_path(page_dir, panel_id, key),
                mask=result.mask.astype(np.uint8),
            )
            meta[panel_id] = meta.get(panel_id, {})
            meta[panel_id][key] = {
                "object_id": result.object_id,
                "bbox": _bbox_to_dict(result.bbox),
                "model_id": result.model_id,
                "iou_score": result.iou_score,
                "mask_shape": list(result.mask.shape),
            }
    (page_dir / _SEGMENTATION_FILE).write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8"
    )


def load_segmentation(
    page_dir: Path,
) -> dict[str, dict[tuple[str, int], SegmentationResult]]:
    meta = json.loads((page_dir / _SEGMENTATION_FILE).read_text(encoding="utf-8"))
    result: dict[str, dict[tuple[str, int], SegmentationResult]] = {}
    for panel_id, entries in meta.items():
        for key, data in entries.items():
            object_id, rank = _split_candidate_key(key)
            mask = np.load(_mask_path(page_dir, panel_id, key))["mask"]
            result.setdefault(panel_id, {})[(object_id, rank)] = SegmentationResult(
                object_id=str(data["object_id"]),
                mask=mask,
                bbox=_bbox_from_dict(data["bbox"]),
                model_id=str(data["model_id"]),
                iou_score=data.get("iou_score"),
            )
    return result


# -------------------------------------------------------------------------------------
# Stage presence helpers
# -------------------------------------------------------------------------------------


def has_grounding(page_dir: Path) -> bool:
    return (page_dir / _GROUNDING_FILE).exists()


def has_descriptions(page_dir: Path) -> bool:
    return (page_dir / _DESCRIPTIONS_FILE).exists()


def has_segmentation(page_dir: Path) -> bool:
    return (page_dir / _SEGMENTATION_FILE).exists()
