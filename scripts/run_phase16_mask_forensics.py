"""Phase 16 forensic: is the VLM right that raised_sword_12 / character_eyes_2 contain a
speech bubble with text?

The strengthened mask prompt flagged both GOOD benchmark masks as bad ("includes a speech
bubble with text"). This diagnostic re-runs the VLM on the exact same masked crop but asks a
neutral, open question -- "what distinct visual elements are inside the bright region?" --
so we can tell whether the VLM is genuinely seeing a bubble (maybe the ground-truth label is
wrong) or hallucinating one (false reject). cloth_5 is included as a positive control (it
REALLY contains a bubble+hand).

Run on the GPU worker. Usage:
    python scripts/run_phase16_mask_forensics.py --qwen /kaggle/working/models/qwen
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from manga_animation.analysis import Qwen25VLClient
from manga_animation.core.logging import setup_logging
from manga_animation.pipeline.types import BBoxPx
from manga_animation.validation.mask_semantics import _crop_with_mask_overlay

SAMPLES = [
    ("raised_sword_12",
     "outputs/debug/phase11_gpu_evidence/villainess_ending_scuffle_primary_mask.npy",
     "examples/realworld/villainess_ending_scuffle.png", "good (new prompt: bad)"),
    ("character_eyes_2",
     "outputs/debug/phase11_gpu_evidence/sss_hunter_gladiator_obj_character_eyes_2_mask.npy",
     "examples/realworld/sss_hunter_gladiator.png", "good (new prompt: bad)"),
    ("cloth_5",
     "outputs/debug/phase11_gpu_evidence/villainess_ending_scuffle_obj_cloth_5_mask.npy",
     "examples/realworld/villainess_ending_scuffle.png", "bad (control: really has bubble+hand)"),
]

_PROMPT = """You are looking at a cropped region of a manga page. The BRIGHT area is a \
segmentation mask being checked; the DARKENED area is surrounding context (ignore it). \
Describe ONLY what is visible inside the BRIGHT region, concretely and literally.

If the bright region contains any of these, name them explicitly:
- dialogue or sound-effect text / a speech bubble,
- a hand, arm, face, or body part,
- more than one distinct object,
- a single coherent object.

Answer with ONLY one JSON object, no prose: {"bright_region_content": "one short sentence \
listing exactly what is inside the bright area", "is_single_coherent_object": true or false, \
"contains_text_or_bubble": true or false}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=str, required=True)
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    setup_logging(debug=False)
    vlm = Qwen25VLClient(source=args.qwen, dtype="float32")
    try:
        for label, mask_path, page_path, tag in SAMPLES:
            mask = np.load(mask_path)
            img = cv2.imread(page_path)
            if img.shape[:2] != mask.shape:
                print(f"=== {label} ({tag}) === SHAPE MISMATCH mask={mask.shape} page={img.shape}")
                continue
            inside = mask > 0
            ys, xs = np.where(inside)
            bbox = BBoxPx(x0=int(xs.min()), y0=int(ys.min()),
                          x1=int(xs.max()) + 1, y1=int(ys.max()) + 1)
            crop = _crop_with_mask_overlay(img, mask, bbox)
            from PIL import Image
            raw = vlm.generate(Image.fromarray(crop), _PROMPT)
            print(f"=== {label} ({tag}) === bbox={bbox.as_xyxy()}")
            print(f"  VLM: {raw}")
    finally:
        vlm.unload()


if __name__ == "__main__":
    main()
