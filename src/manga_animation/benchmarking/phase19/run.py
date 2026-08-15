"""Phase 19 GPU runner: single smoke, five-target smoke, and the full 64-target controlled
benchmark, running the real OMG-LLaVA adapter.

GPU discipline (phase brief section 15/17): the model is loaded ONCE and kept resident across
all targets (never reloaded per target), then released deterministically in a `finally`. A
single target's failure is recorded as `inference_error` (taxonomy K) and the loop continues --
difficult examples are never discarded. Latency and peak VRAM are measured per target.

The record-assembly logic (`assemble_controlled_record`, `masks_to_original`) is pure and
unit-tested with a fake adapter; only the GPU loop itself imports the adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.metrics import mask_metrics
from manga_animation.benchmarking.phase19.masks import (
    SquarePad,
    mask_from_canvas,
    tight_bbox_from_mask,
)
from manga_animation.benchmarking.phase19.metrics import measure_target_metrics
from manga_animation.benchmarking.phase19.parsing import no_target_text, target_context

BBox = tuple[int, int, int, int]


def masks_to_original(
    padded_masks: list[np.ndarray], page_size: tuple[int, int]
) -> list[np.ndarray]:
    """Convert adapter masks (padded-square canvas, bool) to original page geometry (H, W)."""
    pad = SquarePad.from_page_size(page_size)
    return [mask_from_canvas(m, pad) for m in padded_masks]


def masks_pairwise_overlap(masks: list[np.ndarray]) -> float | None:
    """IoU between the first two masks on the same geometry -- the "distinct instances"
    signal for the D (multiple instances) taxonomy category."""
    if len(masks) < 2:
        return None
    return mask_metrics(masks[0], masks[1]).iou


def select_five_targets(manifest) -> list:
    """Deterministic five-target smoke selection covering the brief's five difficulty axes.

    The five axes (multiple similar characters / target near text / small target / partial
    occlusion / target close to another object) are picked by documented proxies from the
    manifest's GT-derived features -- a curation heuristic, and the five are never used to
    tune the system:
      1. smallest target:      min `area_fraction`
      2. crowded page:         first sample on the page with the most manifest instances
      3. sparse/occluded:      min `silhouette_density`
      4. thin target:          min `aspect_ratio`
      5. close to another:     sample whose bbox is closest to another sample's bbox on the
                               same page
    """
    samples = list(manifest.samples)
    if len(samples) < 5:
        raise ValueError(f"need >= 5 manifest samples, got {len(samples)}")
    chosen: list = []

    def pick(key):
        return min(samples, key=key)

    chosen.append(pick(lambda s: s.features.get("area_fraction", 1.0)))
    by_page: dict[str, list] = {}
    for s in samples:
        by_page.setdefault(f"{s.book}_{s.page_index:03d}", []).append(s)
    crowded = max(by_page.values(), key=len)[0]
    if crowded.sample_id not in {c.sample_id for c in chosen}:
        chosen.append(crowded)
    chosen.append(pick(lambda s: s.features.get("silhouette_density", 1.0)))
    chosen.append(pick(lambda s: s.features.get("aspect_ratio", 1e9)))

    def min_pair_distance(s) -> float:
        best = float("inf")
        for other in samples:
            if other.sample_id == s.sample_id:
                continue
            if f"{other.book}_{other.page_index:03d}" != f"{s.book}_{s.page_index:03d}":
                continue
            d = _bbox_center_distance(s.gt_bbox, other.gt_bbox)
            best = min(best, d)
        return best

    chosen.append(min(samples, key=min_pair_distance))
    seen: list = []
    for s in chosen:
        if s.sample_id not in {c.sample_id for c in seen}:
            seen.append(s)
    for s in samples:  # guarantee five distinct samples deterministically
        if len(seen) >= 5:
            break
        if s.sample_id not in {c.sample_id for c in seen}:
            seen.append(s)
    return seen[:5]


def _bbox_center_distance(a: BBox, b: BBox) -> float:
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)


@dataclass
class ControlledSampleRecord:
    """One controlled-benchmark target's full result (GPU + CPU, raw + metrics)."""

    sample_id: str
    page_key: str
    condition: str
    prompt: str
    provenance: str
    page_size: tuple[int, int]
    gt_bbox: BBox
    status: str  # "ok" | "inference_error"
    error_detail: str | None = None
    n_masks: int = 0
    output_text: str = ""
    target_context: str | None = None
    target_not_found_text: bool = False
    multi_instance: bool = False
    mask_overlap: float | None = None
    pred_mask_path: str | None = None
    pred_bbox: list[int] | None = None
    latency_seconds: float = 0.0
    vram_peak_mb: float | None = None
    metrics: dict[str, Any] | None = None
    failure_category: str | None = None
    forbidden: dict[str, float] | None = None
    manual_category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "page_key": self.page_key,
            "condition": self.condition,
            "prompt": self.prompt,
            "provenance": self.provenance,
            "page_size": list(self.page_size),
            "gt_bbox": list(self.gt_bbox),
            "status": self.status,
            "error_detail": self.error_detail,
            "n_masks": self.n_masks,
            "output_text": self.output_text,
            "target_context": self.target_context,
            "target_not_found_text": self.target_not_found_text,
            "multi_instance": self.multi_instance,
            "mask_overlap": self.mask_overlap,
            "pred_mask_path": self.pred_mask_path,
            "pred_bbox": self.pred_bbox,
            "latency_seconds": self.latency_seconds,
            "vram_peak_mb": self.vram_peak_mb,
            "metrics": self.metrics,
            "failure_category": self.failure_category,
            "forbidden": self.forbidden,
            "manual_category": self.manual_category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlledSampleRecord:
        return cls(
            sample_id=data["sample_id"],
            page_key=data["page_key"],
            condition=data["condition"],
            prompt=data["prompt"],
            provenance=data["provenance"],
            page_size=tuple(data["page_size"]),
            gt_bbox=tuple(data["gt_bbox"]),
            status=data["status"],
            error_detail=data.get("error_detail"),
            n_masks=int(data.get("n_masks", 0)),
            output_text=data.get("output_text", ""),
            target_context=data.get("target_context"),
            target_not_found_text=bool(data.get("target_not_found_text", False)),
            multi_instance=bool(data.get("multi_instance", False)),
            mask_overlap=data.get("mask_overlap"),
            pred_mask_path=data.get("pred_mask_path"),
            pred_bbox=data.get("pred_bbox"),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            vram_peak_mb=data.get("vram_peak_mb"),
            metrics=data.get("metrics"),
            failure_category=data.get("failure_category"),
            forbidden=data.get("forbidden"),
            manual_category=data.get("manual_category"),
        )


def assemble_controlled_record(
    sample,
    image: np.ndarray,
    gt_mask: np.ndarray,
    out,
    *,
    condition: str,
    prompt: str,
    provenance: str,
    out_dir: Path,
    latency_seconds: float = 0.0,
    vram_peak_mb: float | None = None,
    error: Exception | None = None,
) -> ControlledSampleRecord:
    """Assemble the record for one target from the adapter output (or an inference error).

    Pure (no model imports) so it is unit-tested with a fake adapter object. Saves the
    predicted masks as `<sample_id>.<condition>.mask.npz` on original page geometry.
    """
    page_size = tuple(image.shape[:2])
    if error is not None:
        return ControlledSampleRecord(
            sample_id=sample.sample_id,
            page_key=f"{sample.book}_{sample.page_index:03d}",
            condition=condition,
            prompt=prompt,
            provenance=provenance,
            page_size=page_size,
            gt_bbox=sample.gt_bbox,
            status="inference_error",
            error_detail=f"{type(error).__name__}: {error}",
            latency_seconds=latency_seconds,
            vram_peak_mb=vram_peak_mb,
        )

    masks = masks_to_original(out.masks, page_size)
    multi = len(masks) > 1
    overlap = masks_pairwise_overlap(masks) if len(masks) >= 2 else None
    if multi and overlap is not None and overlap < 0.5:
        multi = True
    elif multi:
        multi = False  # two masks on the SAME object are not "multiple instances"

    pred_mask_path = None
    pred_bbox = None
    metrics_dict = None
    if masks:
        pred = masks[0]  # the primary mask: the first [SEG]
        pred_mask_path = str(out_dir / f"{sample.sample_id}.{condition}.mask.npz")
        np.savez_compressed(pred_mask_path, mask=pred.astype(np.uint8) * 255)
        try:
            pred_bbox = list(tight_bbox_from_mask(pred))
        except ValueError:
            pred_bbox = None
        metrics_dict = measure_target_metrics(sample.sample_id, pred, gt_mask).as_dict()

    return ControlledSampleRecord(
        sample_id=sample.sample_id,
        page_key=f"{sample.book}_{sample.page_index:03d}",
        condition=condition,
        prompt=prompt,
        provenance=provenance,
        page_size=page_size,
        gt_bbox=sample.gt_bbox,
        status="ok",
        n_masks=len(masks),
        output_text=out.text,
        target_context=target_context(out.text),
        target_not_found_text=no_target_text(out.text),
        multi_instance=multi,
        mask_overlap=overlap,
        pred_mask_path=pred_mask_path,
        pred_bbox=pred_bbox,
        latency_seconds=latency_seconds,
        vram_peak_mb=vram_peak_mb,
        metrics=metrics_dict,
    )


def _vram_peak_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024.0**2


def _reset_vram_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_controlled_benchmark(
    manifest,
    dataset_dir: Path,
    out_dir: Path,
    adapter,
    *,
    condition: str,
    provenance: str,
    prompt_for: Any,
) -> list[ControlledSampleRecord]:
    """Run every manifest target through the loaded adapter, saving raw outputs + masks.

    `prompt_for` is a callable `sample -> prompt string` so the CLI decides which condition's
    prompt is built (and `provenance` documents where that description came from). The model
    stays loaded for the whole run; a per-target inference error is recorded and never aborts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ControlledSampleRecord] = []
    for i, sample in enumerate(manifest.samples):
        image = np.asarray(Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB"))
        gt_mask = np.load(dataset_dir / f"{sample.sample_id}.mask.npz")["mask"]
        prompt = prompt_for(sample)
        latency = 0.0
        vram = None
        error: Exception | None = None
        out = None
        _reset_vram_peak()
        try:
            out = adapter.predict(image, prompt)
            latency = out.latency_seconds
            vram = _vram_peak_mb()
        except Exception as exc:  # noqa: BLE001 -- a failed target is recorded, never dropped
            error = exc
            latency = 0.0
            vram = _vram_peak_mb()
        records.append(
            assemble_controlled_record(
                sample,
                image,
                gt_mask,
                out,
                condition=condition,
                prompt=prompt,
                provenance=provenance,
                out_dir=out_dir,
                latency_seconds=latency,
                vram_peak_mb=vram,
                error=error,
            )
        )
        if (i + 1) % 10 == 0 or i == len(manifest.samples) - 1:
            print(f"[phase19] {i + 1}/{len(manifest.samples)} targets done")
    return records


def run_smoke(
    image: np.ndarray,
    prompt: str,
    adapter,
    out_dir: Path,
    *,
    label: str = "smoke",
) -> dict[str, Any]:
    """Single-inference smoke: latency + peak VRAM + whether [SEG] masks were emitted. Saves the
    page, the raw output text, and the masks under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _reset_vram_peak()
    try:
        out = adapter.predict(image, prompt)
    except Exception as exc:  # noqa: BLE001 -- a smoke failure is the measurement
        return {
            "label": label,
            "status": "inference_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    vram = _vram_peak_mb()
    Image.fromarray(image).save(out_dir / f"{label}.page.png")
    (out_dir / f"{label}.text.txt").write_text(out.text, encoding="utf-8")
    for i, mask in enumerate(out.masks):
        np.savez_compressed(out_dir / f"{label}.mask{i}.npz", mask=mask)
    return {
        "label": label,
        "status": "ok",
        "n_masks": out.n_masks,
        "latency_seconds": out.latency_seconds,
        "vram_peak_mb": vram,
        "has_seg": out.n_masks > 0,
        "target_context": target_context(out.text),
        "output_text": out.text,
    }


def save_records(records: list[ControlledSampleRecord], out_dir: Path) -> Path:
    path = out_dir / "per_sample_results.json"
    path.write_text(json.dumps([r.as_dict() for r in records], indent=2), encoding="utf-8")
    return path


def load_records(out_dir: Path) -> list[ControlledSampleRecord]:
    return [
        ControlledSampleRecord.from_dict(entry)
        for entry in json.loads((out_dir / "per_sample_results.json").read_text(encoding="utf-8"))
    ]
