"""Phase 18.2 visual diagnostic packages: per-target montage of the VLM reranking.

For representative targets (VLM success, VLM failure, correct candidate ranked very low,
multiple similar candidates), renders one PNG per target showing: the original page (with GT
bbox, DINO candidates), the top-K candidates the VLM ranked, the VLM matches/confidence, and
the GT candidate -- so the reranker's behavior is inspectable, not just numeric.

Runs locally after the GPU results are pulled (reads per_target_rerank.json +
vlm_scores_by_page.json + the phase-17 dataset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.pipeline.types import BBoxPx
from manga_animation.validation.validate import _crop_with_margin


def _load_scores(scores_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return json.loads(scores_path.read_text(encoding="utf-8"))


def _pick_representative(per_target: list[dict[str, Any]], max_cases: int) -> list[str]:
    """Deterministic priority: VLM successes, VLM failures, correct-ranked-low, then the rest
    ordered by descending gap between DINO rank and VLM rank of the correct candidate."""
    chosen: list[str] = []
    buckets: dict[str, list[str]] = {"success": [], "failure": [], "correct_low": [], "other": []}
    for e in per_target:
        a = e["strategies"]["A"]
        if e["best_available_iou"] < 0.5:
            continue  # category C -- separate
        if a["selected_correct"]:
            buckets["success"].append(e["sample_id"])
        elif a["best_correct_rank"] is not None and a["best_correct_rank"] >= 10:
            buckets["correct_low"].append(e["sample_id"])
        else:
            buckets["failure"].append(e["sample_id"])
    for bucket in ("success", "failure", "correct_low", "other"):
        chosen.extend(buckets[bucket])
    return chosen[:max_cases]


def build_rerank_visuals(
    manifest: BenchmarkManifest,
    per_target: list[dict[str, Any]],
    scores_by_page: dict[str, dict[str, dict[str, Any]]],
    dataset_dir: Path,
    out_dir: Path,
    *,
    max_cases: int = 12,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {s.sample_id: s for s in manifest.samples}
    written: list[Path] = []
    for sample_id in _pick_representative(per_target, max_cases):
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        page_key = f"{sample.book}_{sample.page_index:03d}"
        image = np.asarray(Image.open(dataset_dir / f"{sample_id}.png").convert("RGB"))
        h, w = image.shape[:2]
        scale = min(1.0, 560.0 / max(h, w))
        small = Image.fromarray(image).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
        small_arr = np.asarray(small).copy()  # writable copy for cv2 drawing
        sh, sw = small_arr.shape[:2]
        page_scores = scores_by_page.get(page_key, {})

        def draw_box(box, color: tuple[int, int, int], thick: int = 2,
                     sc: float = scale, canvas: np.ndarray = small_arr):
            x0, y0, x1, y1 = (int(v * sc) for v in box)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thick)

        # Panel 1: page with GT (green) and all DINO candidates (thin gray).
        for s in page_scores.values():
            draw_box(s["box"], (160, 160, 160), 1)
        draw_box(sample.gt_bbox, (0, 255, 0), 3)

        # Panel 2: the VLM's top-5 ranked candidates, colored by rank, GT in green.
        def rank_key(s):
            m = s["matches"]
            grp = 0 if m is True else 1 if m is False else 2
            conf = s["confidence"] if s["confidence"] is not None else -1.0
            return (grp, -conf)

        ranked = sorted(page_scores.values(), key=rank_key)
        for s in ranked[:5]:
            draw_box(s["box"], (255, 128, 0), 2)

        # Panels 3..N: individual candidate crops (the exact production crops) with the VLM
        # verdict as text.
        crops: list[np.ndarray] = []
        for s in ranked[:6]:
            crop = _crop_with_margin(image, BBoxPx(x0=s["box"][0], y0=s["box"][1],
                                                   x1=s["box"][2], y1=s["box"][3]))
            cc = crop.copy()
            verdict = "MATCH" if s["matches"] is True else "NO" if s["matches"] is False else "?"
            conf = f" {s['confidence']:.2f}" if s["confidence"] is not None else ""
            cc_h, cc_w = cc.shape[:2]
            cc_scale = min(1.0, 180.0 / max(cc_h, cc_w))
            cc_small = np.asarray(Image.fromarray(cc).resize(
                (max(1, int(cc_w * cc_scale)), max(1, int(cc_h * cc_scale))),
                Image.Resampling.LANCZOS,
            )).copy()
            cv2.putText(cc_small, verdict + conf, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            crops.append(cc_small)

        panels = [small_arr]
        if crops:
            cw = max(c.shape[1] for c in crops)
            pad = 2
            stack = np.ones((sum(c.shape[0] for c in crops) + pad * (len(crops) - 1), cw, 3),
                            dtype=np.uint8) * 30
            y = 0
            for c in crops:
                stack[y : y + c.shape[0], : c.shape[1]] = c
                y += c.shape[0] + pad
            panels.append(stack)

        target_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < target_h:
                canvas = np.ones((target_h, p.shape[1], 3), dtype=np.uint8) * 20
                canvas[: p.shape[0], : p.shape[1]] = p
                padded.append(canvas)
            else:
                padded.append(p)

        out_path = out_dir / f"rerank_{sample_id}.png"
        Image.fromarray(np.concatenate(padded, axis=1)).save(out_path)
        written.append(out_path)
    return written
