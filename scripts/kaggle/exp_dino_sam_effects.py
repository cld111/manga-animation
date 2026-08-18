"""Experiment: DINO -> SAM -> original image with overlaid masks (ALL manga effects).

Expands the speed-lines-only experiment to find ANY visual effect on the page:
motion/impact lines, bursts/explosions, glow/flash/aura, action slashes, emotional marks,
particles/debris, stylized comic effects. For each of the 10 realworld manga pages:

1. GroundingDINO detects effect boxes with a broad, categorized prompt.
2. SAM 2.1 segments the mask for each accepted effect box.
3. Masks are composited over the ORIGINAL image, color-coded by effect category.

Writes per page (into OUT_DIR):
  <name>_effects_overlay.png  original image with effect masks drawn over it
  <name>_effects.json         DINO detections + category + SAM mask stats

Model inference runs on the remote Kaggle GPU worker only (ADR 0003/CLAUDE.md).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PAGES_DIR = Path("/kaggle/working/manga-animation/examples/realworld")
OUT_DIR = Path("/kaggle/working/exp_effects")
DINO_REPO = "IDEA-Research/grounding-dino-base"
SAM_REPO = "facebook/sam2.1-hiera-base-plus"

# Categorized manga-effect concepts. " . " join is GroundingDINO's prompt syntax.
EFFECT_CATEGORIES = {
    "motion_lines": [
        "speed lines", "motion lines", "action lines", "impact lines",
        "background speed lines", "radial lines", "concentration lines",
        "movement lines", "wind lines", "trajectory lines", "afterimage", "motion blur",
    ],
    "impact_burst": [
        "explosion", "impact burst", "blast", "shockwave", "impact stars",
        "burst effect", "flash star",
    ],
    "glow_flash": [
        "glow", "flash", "bright light", "halo", "aura", "shine", "sparkle",
        "radiance", "light rays",
    ],
    "action_slash": [
        "sword slash", "slash effect", "cut marks", "claw marks", "scratch marks",
        "attack swing", "punch effect",
    ],
    "emotional": [
        "anger mark", "cross vein", "sweat drop", "blush mark", "shock mark",
        "surprise mark", "heart mark",
    ],
    "particles": [
        "dust", "debris", "sparks", "smoke", "steam", "confetti", "shards",
        "water splash", "sand",
    ],
    "stylized": [
        "comic effect", "manga effect", "screen tone", "sound effect text",
        "action lines background",
    ],
}

# Build the full DINO prompt from all categories.
EFFECT_PROMPT = " . ".join(
    concept for concepts in EFFECT_CATEGORIES.values() for concept in concepts
) + "."

# category -> RGBA overlay color
CATEGORY_COLOR = {
    "motion_lines": (255, 80, 0, 120),    # orange
    "impact_burst": (255, 0, 0, 120),     # red
    "glow_flash": (255, 230, 0, 120),     # yellow
    "action_slash": (0, 200, 255, 120),   # cyan
    "emotional": (255, 0, 220, 120),      # magenta
    "particles": (120, 120, 120, 120),    # gray
    "stylized": (0, 180, 80, 120),        # green
}
DEFAULT_COLOR = (255, 255, 255, 120)  # white fallback

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


def log(msg: str) -> None:
    print(f"[exp_effects] {msg}", flush=True)


def category_of(label: str) -> str:
    """Map a DINO detection label back to its effect category."""
    lab = label.strip().lower()
    for cat, concepts in EFFECT_CATEGORIES.items():
        for concept in concepts:
            if concept in lab:
                return cat
    return "unknown"


def build_overlay(image_np: np.ndarray, masks: list[np.ndarray]) -> Image.Image:
    """Compose original image with colored semi-transparent masks overlaid."""
    base = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for mask, color in masks:
        if mask is None:
            continue
        if not np.any(mask > 0):
            continue
        color_arr = np.zeros((*mask.shape, 4), dtype=np.uint8)
        color_arr[mask > 0] = color
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
    log(f"prompt ({len(EFFECT_PROMPT.split(' . '))} concepts): {EFFECT_PROMPT}")

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

        # ---- DINO: all effects ----
        inputs = dino_processor(images=image_pil, text=EFFECT_PROMPT, return_tensors="pt")
        inputs = {k: v.to(dino_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = dino_model(**inputs)
        results = dino_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=0.2, text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        # transformers 5.0: use text_labels when present (labels may be integer ids)
        text_labels = results.get("text_labels", [])
        labels = results.get("labels", [])
        detections = []
        for i, (box, score) in enumerate(zip(results["boxes"], results["scores"], strict=True)):
            if i < len(text_labels):
                label = str(text_labels[i])
            elif i < len(labels):
                label = str(labels[i])
            else:
                label = "effect"
            box_list = [float(v) for v in box.tolist()]
            detections.append({
                "label": label,
                "category": category_of(label),
                "score": round(float(score), 4),
                "bbox": [round(v, 1) for v in box_list],
            })
        log(f"  DINO found {len(detections)} effect boxes")

        # ---- SAM: mask for each detection ----
        masks = []  # (mask, color)
        mask_records = []
        for det in detections:
            x0, y0, x1, y1 = [int(v) for v in det["bbox"]]
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
            best_i = int(np.argmax(scores.numpy()))
            mask = (cands[best_i].numpy() > 0).astype(np.uint8) * 255
            color = CATEGORY_COLOR.get(det["category"], DEFAULT_COLOR)
            masks.append((mask, color))
            mask_records.append({
                "label": det["label"],
                "category": det["category"],
                "dino_score": det["score"],
                "bbox": det["bbox"],
                "sam_iou": round(float(scores[best_i]), 4),
                "mask_area_px": int((mask > 0).sum()),
                "mask_area_frac": round(float((mask > 0).sum()) / (h * w), 5),
            })
            log(f"    [{det['category']}] {det['label']} "
                f"iou={scores[best_i]:.3f} frac={mask_records[-1]['mask_area_frac']}")
            torch.cuda.empty_cache()

        # ---- Overlay on original ----
        overlay = build_overlay(image_np, masks)
        overlay_path = OUT_DIR / f"{Path(page_name).stem}_effects_overlay.png"
        overlay.save(overlay_path)
        log(f"  saved overlay -> {overlay_path}")

        # ---- Category counts ----
        cat_counts: dict[str, int] = {}
        for m in mask_records:
            cat_counts[m["category"]] = cat_counts.get(m["category"], 0) + 1

        record = {
            "page": page_name,
            "size": [w, h],
            "dino_prompt": EFFECT_PROMPT,
            "num_detections": len(detections),
            "num_masks": len(mask_records),
            "category_counts": cat_counts,
            "detections": detections,
            "masks": mask_records,
        }
        (OUT_DIR / f"{Path(page_name).stem}_effects.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        summary.append({
            "page": page_name,
            "detections": len(detections),
            "masks": len(mask_records),
            "category_counts": cat_counts,
            "overlay": str(overlay_path),
        })

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    log(f"\nDONE: {len(pages)} pages processed")
    for s in summary:
        log(f"  {s['page']}: {s['detections']} detections, {s['masks']} masks "
            f"{s['category_counts']}")


if __name__ == "__main__":
    main()
