"""Phase 18.2A GPU runner: Qwen direct-localization bbox + Qwen-bbox->SAM downstream masks.

Two model stages, following the project's stage-level lifecycle discipline (never co-reside):

- **Stage 1 (VLM):** for each of the 64 targets, run Qwen2.5-VL on the FULL page (at source
  resolution -- Qwen's processor bounds the vision tokens internally, and source resolution
  keeps the pixel-coordinate reference unambiguous; see `coords.py` for the measured
  coordinate contract) with the production target description, parse/convert the returned
  bbox (`coords.py`), and compare against GT bbox. Results are cached per sample
  (`predictions_by_sample.json`) so a partial/failed run resumes without re-inferring.
- **Stage 2 (SAM):** for each target, run SAM 2.1 on the GT bbox (reference/upper bound) and,
  when the Qwen bbox is usable, on the Qwen bbox (the downstream experiment). Masks are saved
  as git-ignored npz artifacts and skipped when already present.

Only the VLM and SAM load. Grounding DINO is NOT touched (Phase 18.1 already measured its
candidate recall; this phase measures Qwen's independent localization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.analysis.client import VLMClient
from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase17.metrics import (
    bbox_area_ratio,
    bbox_gt_coverage,
    bbox_iou,
    mask_metrics,
)
from manga_animation.benchmarking.phase18a.classify import classify
from manga_animation.benchmarking.phase18a.coords import (
    QwenBboxPrediction,
    convert_prediction,
)
from manga_animation.benchmarking.phase18a.metrics import PerTargetMetrics
from manga_animation.benchmarking.phase18a.prompt import build_direct_prompt
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.types import BBoxPx
from manga_animation.segmentation.client import SegmentationClient

BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DirectLocalizationRecord:
    """One sample's VLM output plus its per-target measurements (IoU etc. vs GT)."""

    prediction: QwenBboxPrediction
    target_description: str
    gt_bbox: BBox
    bbox_iou: float | None
    gt_coverage: float | None
    area_ratio: float | None
    page_w: int
    page_h: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.prediction.sample_id,
            "target_description": self.target_description,
            "gt_bbox": list(self.gt_bbox),
            "prediction": self.prediction.as_dict(),
            "bbox_iou": self.bbox_iou,
            "gt_coverage": self.gt_coverage,
            "area_ratio": self.area_ratio,
            "page_size": [self.page_h, self.page_w],
        }


def _load_page(dataset_dir: Path, sample_id: str) -> np.ndarray:
    return np.asarray(Image.open(dataset_dir / f"{sample_id}.png").convert("RGB"))


def _query_one(
    sample_id: str,
    sample_prompt: str,
    gt_bbox: BBox,
    image: np.ndarray,
    vlm_client: VLMClient,
) -> DirectLocalizationRecord:
    h, w = image.shape[:2]
    prompt = build_direct_prompt(sample_prompt, w, h)
    raw_text = vlm_client.generate(Image.fromarray(image), prompt)
    prediction = convert_prediction(sample_id, raw_text, w, h)

    iou = coverage = ratio = None
    if prediction.usable and prediction.pixel_box is not None:
        iou = bbox_iou(gt_bbox, prediction.pixel_box)
        coverage = bbox_gt_coverage(gt_bbox, prediction.pixel_box)
        ratio = bbox_area_ratio(gt_bbox, prediction.pixel_box)

    return DirectLocalizationRecord(
        prediction=prediction,
        target_description=sample_prompt,
        gt_bbox=gt_bbox,
        bbox_iou=iou,
        gt_coverage=coverage,
        area_ratio=ratio,
        page_w=w,
        page_h=h,
    )


def collect_direct_predictions(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    vlm_client: VLMClient,
) -> list[DirectLocalizationRecord]:
    """Run the VLM stage over every sample. Resumable via `predictions_by_sample.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "predictions_by_sample.json"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"resuming from cached predictions: {len(cached)} samples")

    records: list[DirectLocalizationRecord] = []
    with ModelStage(vlm_client, name="vlm_direct_bbox"):
        for sample in manifest.samples:
            if sample.sample_id in cached:
                entry = cached[sample.sample_id]
                prediction = QwenBboxPrediction(
                    sample_id=entry["prediction"]["sample_id"],
                    found=bool(entry["prediction"]["found"]),
                    box_raw=(
                        tuple(entry["prediction"]["box_raw"])
                        if entry["prediction"].get("box_raw")
                        else None
                    ),
                    pixel_box=(
                        tuple(entry["prediction"]["pixel_box"])
                        if entry["prediction"].get("pixel_box")
                        else None
                    ),
                    raw_text=str(entry["prediction"].get("raw_text", "")),
                    error=entry["prediction"].get("error"),
                    convention_ok=bool(entry["prediction"].get("convention_ok", False)),
                    width=int(entry["prediction"]["width"]),
                    height=int(entry["prediction"]["height"]),
                    clamped=bool(entry["prediction"].get("clamped", False)),
                )
                records.append(
                    DirectLocalizationRecord(
                        prediction=prediction,
                        target_description=str(entry["target_description"]),
                        gt_bbox=(
                            int(entry["gt_bbox"][0]),
                            int(entry["gt_bbox"][1]),
                            int(entry["gt_bbox"][2]),
                            int(entry["gt_bbox"][3]),
                        ),
                        bbox_iou=entry.get("bbox_iou"),
                        gt_coverage=entry.get("gt_coverage"),
                        area_ratio=entry.get("area_ratio"),
                        page_w=int(entry["page_size"][1]),
                        page_h=int(entry["page_size"][0]),
                    )
                )
                continue
            image = _load_page(dataset_dir, sample.sample_id)
            record = _query_one(
                sample.sample_id, sample.prompt, sample.gt_bbox, image, vlm_client
            )
            records.append(record)
            print(
                f"  {sample.sample_id}: found={record.prediction.found} "
                f"iou={record.bbox_iou if record.bbox_iou is not None else 'n/a'}"
            )
            cached[sample.sample_id] = record.as_dict()
            cache_path.write_text(json.dumps(cached, indent=1), encoding="utf-8")
    return records


def _best_sam_candidate(client: SegmentationClient, image: np.ndarray, box: BBox) -> np.ndarray:
    """The mask SAM would produce for `box` under the production convention (best of the
    model's candidate masks by its own iou_score -- the exact selection `segment_object` uses)."""
    candidates = client.segment(image, BBoxPx(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))
    if not candidates:
        raise RuntimeError("segmentation model returned no mask candidates for the prompt box")
    return max(candidates, key=lambda c: c.iou_score).mask


def collect_sam_masks(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    segmentation_client: SegmentationClient,
    records: list[DirectLocalizationRecord],
) -> dict[str, dict[str, float]]:
    """SAM stage: mask on the GT bbox (reference) and, when usable, on the Qwen bbox.
    Returns `{sample_id: {"gs_iou": float, "gs_dice": float, ...(mask metrics),
    "qs_iou": float|None, ...}}`. Resumable via saved npz artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record_by_id = {r.prediction.sample_id: r for r in records}
    results: dict[str, dict[str, float]] = {}

    with ModelStage(segmentation_client, name="sam_downstream"):
        for sample in manifest.samples:
            record = record_by_id[sample.sample_id]
            gt_mask = (np.load(dataset_dir / f"{sample.sample_id}.mask.npz")["mask"] > 0).astype(
                np.uint8
            ) * 255
            image = _load_page(dataset_dir, sample.sample_id)
            entry: dict[str, float] = {}

            gs_path = out_dir / f"{sample.sample_id}.gs.mask.npz"
            if gs_path.exists():
                gs = np.load(gs_path)["mask"]
            else:
                gs = _best_sam_candidate(segmentation_client, image, sample.gt_bbox)
                np.savez_compressed(gs_path, mask=gs)
            gs_metrics = mask_metrics(gt_mask, gs)
            entry.update(
                gs_iou=gs_metrics.iou,
                gs_dice=gs_metrics.dice,
                gs_precision=gs_metrics.precision,
                gs_recall=gs_metrics.recall,
            )

            if record.prediction.usable and record.prediction.pixel_box is not None:
                qs_path = out_dir / f"{sample.sample_id}.qs.mask.npz"
                if qs_path.exists():
                    qs = np.load(qs_path)["mask"]
                else:
                    qs = _best_sam_candidate(
                        segmentation_client, image, record.prediction.pixel_box
                    )
                    np.savez_compressed(qs_path, mask=qs)
                qs_metrics = mask_metrics(gt_mask, qs)
                entry.update(
                    qs_iou=qs_metrics.iou,
                    qs_dice=qs_metrics.dice,
                    qs_precision=qs_metrics.precision,
                    qs_recall=qs_metrics.recall,
                )
            else:
                entry.update(qs_iou=0.0, qs_dice=0.0, qs_precision=0.0, qs_recall=0.0)
            results[sample.sample_id] = entry
            print(
                f"  {sample.sample_id}: gs_iou={entry['gs_iou']:.3f} "
                f"qs_iou={entry.get('qs_iou', 0.0):.3f}"
            )

    (out_dir / "sam_masks_by_sample.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    return results


def build_per_target_metrics(
    manifest: BenchmarkManifest,
    records: list[DirectLocalizationRecord],
    sam_results: dict[str, dict[str, float]] | None = None,
) -> list[PerTargetMetrics]:
    """Combine VLM records + SAM results + classification into the report's per-target table."""
    sam_results = sam_results or {}
    out: list[PerTargetMetrics] = []
    for record in records:
        sample = next(s for s in manifest.samples if s.sample_id == record.prediction.sample_id)
        classification = classify(
            sample.gt_bbox,
            record.prediction.pixel_box,
            record.prediction.found,
            record.prediction.error,
            record.page_w,
            record.page_h,
        )
        sam = sam_results.get(record.prediction.sample_id, {})
        out.append(
            PerTargetMetrics(
                sample_id=record.prediction.sample_id,
                gt_bbox=sample.gt_bbox,
                found=record.prediction.found,
                pixel_box=record.prediction.pixel_box,
                bbox_iou=record.bbox_iou,
                gt_coverage=record.gt_coverage,
                area_ratio=record.area_ratio,
                error=record.prediction.error,
                error_category=classification.name,
                gs_mask_iou=sam.get("gs_iou"),
                qs_mask_iou=sam.get("qs_iou"),
            )
        )
    return out
