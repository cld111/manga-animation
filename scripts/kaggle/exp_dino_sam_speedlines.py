"""Experiment: DINO -> SAM -> original image with overlaid masks.

Targets ONLY speed/motion lines (action lines). For each of the 10 realworld manga pages:

1. GroundingDINO detects speed/motion-line boxes with a prompt restricted to speed/motion-line
   concepts (deliberately NOT the full LABELS list — this experiment isolates action lines).
2. SAM 2.1 segments the mask for each accepted speed/motion-line box.
3. The masks are composited over the ORIGINAL image (colored overlay, original preserved).

Writes per page (into OUT_DIR):
  <name>_overlay.png      original image with speed-line masks drawn over it
  <name>_speedlines.json  DINO detections + SAM mask stats for that page

Model inference runs on the remote Kaggle GPU worker only (ADR 0003/CLAUDE.md).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PAGES_DIR = Path("/kaggle/working/manga-animation/examples/realworld")
OUT_DIR = Path("/kaggle/working/exp_speedlines")
DINO_REPO = "IDEA-Research/grounding-dino-base"
SAM_REPO = "facebook/sam2.1-hiera-base-plus"

# Speed/motion line concepts only. " . " join is GroundingDINO's prompt syntax.
SPEEDLINE_PROMPT = " . ".join([
    "speed lines", "motion lines", "action lines", "impact lines",
    "background speed lines", "radial lines", "concentration lines",
    "movement lines", "wind lines", "trajectory lines",
]) + "."

PAGES = [
    "wind_breaker_sprint.png",
    "wind_breaker_finish.png",
    "omniscient_reader_blade.png",
    "angels_of_war_fleet.png",
    "marika_love_meter.png",
    "reality_lie_office.png",
    "space_monster_creature.png",
    "space_monster_hypersenses.png",
    "sss_hunter_gladiator.png",
    "villainess_ending_scuffle.png",
]

OVERLAY_COLOR = (255, 80, 0, 120)  # RGBA: vivid orange, semi-transparent


def log(msg: str) -> None:
    print(f"[exp_speedlines] {msg}", flush=True)


def build_overlay(image_np: np.ndarray, masks: list[np.ndarray]) -> Image.Image:
    """Compose original image with colored semi-transparent masks overlaid."""
    base = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for mask in masks:
        if mask is None:
            continue
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        # polygon from the mask's convex hull (draw.polygon) or point-by-point fill
        # Simpler & robust: flood the mask pixels with the overlay color via numpy.
        color_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
        color_arr[mask > 0] = OVERLAY_COLOR
        overlay.alpha_composite(Image.fromarray(color_arr, "RGBA"))
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out


def main() -> None:
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    log(f"device={device}, torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    pages = [p for p in PAGES if (PAGES_DIR / p).exists()]
    missing = [p for p in PAGES if not (PAGES_DIR / p).exists()]
    if missing:
        log(f"WARNING missing pages: {missing}")
    if not pages:
        log("ERROR: no pages found; run fetch_phase9_realworld_pages.py first")
        sys.exit(1)
    log(f"processing {len(pages)} pages")

    # ---- Load DINO ----
    t0 = time.perf_counter()
    log(f"loading GroundingDINO from {DINO_REPO} ...")
    dino_processor = AutoProcessor.from_pretrained(DINO_REPO)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_REPO, device_map=device,
    )
    dino_model.eval()
    log(f"DINO loaded in {time.perf_counter() - t0:.1f}s")

    # ---- Load SAM 2.1 ----
    t0 = time.perf_counter()
    log(f"loading SAM 2.1 from {SAM_REPO} ...")
    sam_processor = Sam2Processor.from_pretrained(SAM_REPO)
    sam_model = Sam2Model.from_pretrained(SAM_REPO, device_map=device)
    sam_model.eval()
    log(f"SAM loaded in {time.perf_counter() - t0:.1f}s")

    summary = []
    for page_name in pages:
        page_path = PAGES_DIR / page_name
        image_pil = Image.open(page_path).convert("RGB")
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]
        log(f"\n=== {page_name} ({w}x{h}) ===")

        # ---- DINO: speed/motion lines only ----
        inputs = dino_processor(images=image_pil, text=SPEEDLINE_PROMPT, return_tensors="pt")
        inputs = {k: v.to(dino_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=0.2, text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        detections = []
        labels = results.get("labels", [])
        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            label = str(labels[i]) if i < len(labels) else "speed lines"
            box_list = [float(v) for v in box.tolist()]
            detections.append({
                "label": label,
                "score": round(float(score), 4),
                "bbox": [round(v, 1) for v in box_list],
            })
        log(f"  DINO found {len(detections)} speed/motion-line boxes")

        # ---- SAM: mask for each detection ----
        masks = []
        mask_records = []
        for det in detections:
            x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
            # clamp to image
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                log(f"    {det['label']}: degenerate bbox {det['bbox']}, skip")
                continue
            input_boxes = [[[x0, y0, x1, y1]]]
            s_inputs = sam_processor(
                image_pil, input_boxes=input_boxes, return_tensors="pt",
            ).to(sam_model.device)
            with torch.no_grad():
                s_outputs = sam_model(**s_inputs)
            pred = sam_processor.post_process_masks(
                s_outputs.pred_masks.cpu(), s_inputs["original_sizes"]
            )
            cands = pred[0][0]  # (num_candidates, H, W)
            scores = s_outputs.iou_scores.cpu()[0][0]
            # pick best candidate
            best_i = int(np.argmax(scores.numpy()))
            mask = (cands[best_i].numpy() > 0).astype(np.uint8) * 255
            masks.append(mask)
            mask_records.append({
                "label": det["label"],
                "dino_score": det["score"],
                "bbox": det["bbox"],
                "sam_iou": round(float(scores[best_i]), 4),
                "mask_area_px": int((mask > 0).sum()),
                "mask_area_frac": round(float((mask > 0).sum()) / (h * w), 5),
            })
            log(f"    {det['label']} iou={scores[best_i]:.3f} "
                f"mask_frac={mask_records[-1]['mask_area_frac']}")
            torch.cuda.empty_cache()

        # ---- Overlay on original ----
        overlay = build_overlay(image_np, masks)
        overlay_path = OUT_DIR / f"{Path(page_name).stem}_overlay.png"
        overlay.save(overlay_path)
        log(f"  saved overlay -> {overlay_path}")

        record = {
            "page": page_name,
            "size": [w, h],
            "dino_prompt": SPEEDLINE_PROMPT,
            "num_detections": len(detections),
            "num_masks": len(mask_records),
            "detections": detections,
            "masks": mask_records,
        }
        (OUT_DIR / f"{Path(page_name).stem}_speedlines.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        summary.append({
            "page": page_name,
            "detections": len(detections),
            "masks": len(mask_records),
            "overlay": str(overlay_path),
        })

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    log(f"\nDONE: {len(pages)} pages processed")
    for s in summary:
        log(f"  {s['page']}: {s['detections']} detections, {s['masks']} masks")


if __name__ == "__main__":
    main()
