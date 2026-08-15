"""Phase 18.2A visual diagnostic packages: page + GT bbox + Qwen bbox (+ SAM masks).

Purely CPU (reads the saved images/masks). Two kinds of montage, matching the phase brief:

- **bbox packages** for representative bbox outcomes: original page + GT bbox + Qwen bbox;
- **downstream packages** for the SAM experiment: original page + GT bbox + Qwen bbox +
  GT mask + SAM(GT bbox) mask + SAM(Qwen bbox) mask.

Selection is deterministic: worst bbox IoU, biggest area-ratio outliers, conversion failures,
not-found samples, and the best successes -- bounded by `max_cases`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase18a.metrics import PerTargetMetrics

_GRID_LABELS = (
    "bbox",
    "downstream",
)


def _rank_priority(targets: list[PerTargetMetrics], max_cases: int = 12) -> list[str]:
    """Deterministic priority list of sample_ids for visual review: the worst failures first,
    always including the best successes (so the review can see what a correct localization
    looks like, not only the failures)."""
    scored: list[tuple[float, str, str]] = []
    successes: list[tuple[float, str, str]] = []
    for t in targets:
        iou = t.bbox_iou
        scored.append(
            (iou if iou is not None else -1.0, f"worst_iou:{t.sample_id}", t.sample_id)
        )
        if t.error is not None:
            scored.append((1.0, f"conversion:{t.sample_id}", t.sample_id))
        if t.error_category == "not_found":
            scored.append((1.0, f"notfound:{t.sample_id}", t.sample_id))
        if t.qs_mask_iou is not None:
            scored.append((t.qs_mask_iou, f"qs_worst:{t.sample_id}", t.sample_id))
        if iou is not None and iou >= 0.5:
            successes.append((-iou, f"best_iou:{t.sample_id}", t.sample_id))
    scored.sort(key=lambda item: (item[0], item[1]))
    successes.sort(key=lambda item: (item[0], item[1]))
    n_success = min(3, len(successes))
    keep_failures = max_cases - n_success
    seen: list[str] = []
    for _, _reason, sample_id in scored[:keep_failures] + successes[:n_success]:
        if sample_id not in seen:
            seen.append(sample_id)
    return seen


def _draw_box(
    canvas: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]
) -> None:
    import cv2

    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)


def _overlay_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
        (canvas.shape[1], canvas.shape[0]), Image.Resampling.NEAREST
    )
    region = np.asarray(resized) > 0
    canvas[region] = (canvas[region] * 0.55 + np.asarray(color, dtype=np.uint8) * 0.45).astype(
        np.uint8
    )


def _montage(panels: list[np.ndarray]) -> np.ndarray:
    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    cell_h, cell_w = panels[0].shape[:2]
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = panel
    return canvas


def build_visual_packages(
    targets: list[PerTargetMetrics],
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    *,
    max_cases: int = 12,
) -> list[Path]:
    """Write one bbox montage + one downstream montage per selected sample. CPU only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {s.sample_id: s for s in manifest.samples}
    targets_by_id = {t.sample_id: t for t in targets}
    written: list[Path] = []
    for sample_id in _rank_priority(targets, max_cases):
        sample = samples_by_id.get(sample_id)
        target = targets_by_id.get(sample_id)
        if sample is None or target is None:
            continue
        image = np.asarray(Image.open(dataset_dir / f"{sample_id}.png").convert("RGB"))
        h, w = image.shape[:2]
        scale = min(1.0, 640.0 / max(h, w))
        small = Image.fromarray(image).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
        small_arr = np.asarray(small)

        gt_box = (
            int(target.gt_bbox[0] * scale),
            int(target.gt_bbox[1] * scale),
            int(target.gt_bbox[2] * scale),
            int(target.gt_bbox[3] * scale),
        )
        qwen_box = (
            (
                int(target.pixel_box[0] * scale),
                int(target.pixel_box[1] * scale),
                int(target.pixel_box[2] * scale),
                int(target.pixel_box[3] * scale),
            )
            if target.pixel_box is not None
            else None
        )

        bbox_panel = small_arr.copy()
        _draw_box(bbox_panel, gt_box, (0, 255, 0))
        if qwen_box is not None:
            _draw_box(bbox_panel, qwen_box, (255, 0, 0))

        bbox_path = out_dir / f"bbox_{sample_id}.png"
        Image.fromarray(bbox_panel).save(bbox_path)
        written.append(bbox_path)

        downstream_path = out_dir / f"downstream_{sample_id}.png"
        panels = [small_arr.copy(), bbox_panel.copy()]
        gt_mask = np.load(dataset_dir / f"{sample_id}.mask.npz")["mask"] > 0
        gt_panel = small_arr.copy()
        _overlay_mask(gt_panel, gt_mask, (0, 200, 255))
        panels.append(gt_panel)

        gs_path = out_dir.parent / f"{sample_id}.gs.mask.npz"
        qs_path = out_dir.parent / f"{sample_id}.qs.mask.npz"
        if gs_path.exists():
            panel = small_arr.copy()
            _overlay_mask(panel, np.load(gs_path)["mask"] > 0, (255, 120, 120))
            panels.append(panel)
        if qs_path.exists():
            panel = small_arr.copy()
            _overlay_mask(panel, np.load(qs_path)["mask"] > 0, (120, 120, 255))
            panels.append(panel)
        Image.fromarray(_montage(panels)).save(downstream_path)
        written.append(downstream_path)
    return written
