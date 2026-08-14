"""Phase 16 effect-mask diagnostic: are real drawn-effect masks sparse?

The radial_expand geometry profile (35% area bound, 2% edge margin) fail-closed every real
impact/energy burst tried this phase, because bursts legitimately cover a large fraction of
their panel. The hypothesis this diagnostic tests is that the 35% area bound is the wrong
proxy for RADIAL_EXPAND: a burst's *mask* is sparse (radiating lines), so animating it moves
only the drawn effect -- the panel background inside the bbox but outside the mask is not in
the mask, so neither the transform nor compositing moves it. If real effect masks have low
bbox density (mask area / tight-bbox area), the bbox-area bound can be relaxed in favor of a
post-segmentation sparseness check without moving background pixels.

Loads grounding + SAM once, and for each requested semantic label on each page runs a real
grounding prompt against the page (and against each detected panel crop), then a real SAM
mask for every candidate box, measuring: bbox area as a fraction of the region, mask bbox
density, and mask area as a fraction of the region. Prints a table; no rendering, no VLM.

Run on the GPU worker (ADR 0003). Usage:
    python scripts/run_phase16_effect_mask_diagnostic.py \
        --pages P1.png P2.png \
        --labels speed_lines impact_burst \
        --dino /kaggle/working/models/dino --sam /kaggle/working/models/sam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis.panels import detect_panels
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.grounding.ground import _prompt_from_label
from manga_animation.segmentation import Sam21Client


def _bbox_density(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    return float(len(xs)) / float(bw * bh)


def _measure_region(region_name: str, region: np.ndarray, label: str, dino, sam) -> None:
    prompt = _prompt_from_label(label)
    detections = dino.detect(region, prompt)
    if not detections:
        print(f"  {label:16s} {region_name}: no detection")
        return
    r_area = region.shape[0] * region.shape[1]
    for di, det in enumerate(detections[:3]):
        box = det.box
        masks = sam.segment(region, box)
        if not masks:
            print(f"  {label:16s} {region_name} cand{di} score={det.score:.3f}: no mask")
            continue
        best = max(masks, key=lambda m: m.iou_score)
        area_frac = (box.width * box.height) / r_area
        density = _bbox_density(best.mask)
        mask_frac = float((best.mask > 0).sum()) / r_area
        print(
            f"  {label:16s} {region_name} cand{di} score={det.score:.3f} "
            f"box_area_frac={area_frac:.3f} mask_density={density:.3f} "
            f"mask_area_frac={mask_frac:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, nargs="+", default=[])
    parser.add_argument("--labels", nargs="+", default=["speed_lines", "impact_burst"])
    parser.add_argument("--dino", type=str, required=True)
    parser.add_argument("--sam", type=str, required=True)
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()

    dino = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=device, dtype="float32")
    dino.load()
    sam.load()
    try:
        for page in args.pages:
            image = np.asarray(Image.open(page).convert("RGB"))
            panels = detect_panels(image)
            print(f"\n=== {page.name} === panels={len(panels)}")
            for label in args.labels:
                _measure_region("page", image, label, dino, sam)
                for panel in panels:
                    _measure_region(panel.id, panel.crop, label, dino, sam)
    finally:
        dino.unload()
        sam.unload()


if __name__ == "__main__":
    main()
