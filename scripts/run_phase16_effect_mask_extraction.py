"""Phase 16 effect-mask extraction (extraction-only, no VLM / validation / animation / rendering).

Recovers the EXACT two real drawn-effect masks the Phase 16 successful renders used
(docs/phase16-results.md Run 2 and Run 6), by re-running ONLY the deterministic
grounding + SAM path of the Phase 16 pipeline on the exact same scene-crop geometry:

  wind_breaker_sprint  panel_001  semantic_label=speed_lines           -> mesh_warp
  angels_of_war_fleet  panel_001  semantic_label=space_ship_impact_burst -> radial_expand

Why this is exact, and why no VLM is needed:
- The VLM analysis stage is nondeterministic, but its output was already recorded: the
  Phase 16 GPU logs name the exact accepted objects (obj_speed_lines_3,
  obj_space_ship_impact_burst_6), semantic labels, and transform kinds, and both were
  accepted at grounding candidate_rank=0.
- Grounding DINO and SAM 2.1 are deterministic for a fixed input image, prompt, and
  config. The scene crop is deterministic (`analysis/panels.py::detect_panels` +
  `derive_scene_crop_bbox`, CPU). So rank-0 candidate bbox + SAM best-iou mask reproduce
  the exact mask arrays the rendered videos were built from.
- This script calls the SAME production functions the pipeline calls
  (`ground_object_candidates`, `segment_object`) with the SAME models
  (grounding-dino-swin-l, sam2.1-hiera-base, float32) and the SAME grounding thresholds
  (0.25 / 0.2, in `grounding/client.py`). It changes no production code, thresholds,
  prompts, schemas, or mask-selection logic.

Run on the Kaggle/Jupyter GPU worker (ADR 0003), never locally:

    python scripts/run_phase16_effect_mask_extraction.py \
        --dino /kaggle/working/models/dino \
        --sam  /kaggle/working/models/sam \
        --env  kaggle

Outputs (git-ignored, downloaded back to the local checkout):
    outputs/debug/phase16_effects/wind_breaker_sprint_speed_lines_mask.npy
    outputs/debug/phase16_effects/angels_of_war_fleet_impact_burst_mask.npy
    outputs/debug/phase16_effects/manifest.json
plus per-sample original crop PNG, mask-overlay PNG, and a plain-mask PNG for QA.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis.panels import derive_scene_crop_bbox, detect_panels
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.grounding.ground import ground_object_candidates
from manga_animation.schemas.animation_plan import (
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
)
from manga_animation.segmentation import Sam21Client
from manga_animation.segmentation.segment import segment_object

_OUT_DIR = Path("outputs/debug/phase16_effects")

# The two exact targets from the Phase 16 successful renders. `object_id` mirrors the
# Phase 16 slugging convention; `label` is the exact VLM semantic_label recorded in the
# Phase 16 logs; `transform_kind` is the Phase 16 motion mapping for that label.
TARGETS = [
    {
        "page": "examples/realworld/wind_breaker_sprint.png",
        "panel_id": "panel_001",
        "semantic_label": "speed_lines",
        "object_id": "obj_speed_lines_3",
        "transform_kind": "mesh_warp",
    },
    {
        "page": "examples/realworld/angels_of_war_fleet.png",
        "panel_id": "panel_001",
        "semantic_label": "space_ship_impact_burst",
        "object_id": "obj_space_ship_impact_burst_6",
        "transform_kind": "radial_expand",
    },
]


def _mask_stats(mask: np.ndarray) -> dict[str, object]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"pixel_count": 0, "density": 0.0, "bbox": None}
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    return {
        "pixel_count": int((mask > 0).sum()),
        "density": float((mask > 0).sum()) / float(bw * bh),
        "bbox": [x0, y0, x1, y1],
    }


def _overlay(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = crop.copy()
    overlay[mask > 0] = (0, 0, 255)
    return np.asarray(Image.blend(
        Image.fromarray(crop), Image.fromarray(overlay), alpha=0.45
    ))


def _extract_one(
    target: dict[str, str],
    page_img: np.ndarray,
    page_shape: tuple[int, int],
    dino: GroundingDinoClient,
    sam: Sam21Client,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    from manga_animation.grounding.ground import _prompt_from_label

    # 1. Deterministic panel geometry exactly as run_page_panels computes it.
    candidates = detect_panels(page_img)
    if not candidates:
        raise RuntimeError(f"no panels detected for {target['page']}")
    if len(candidates) < 2:
        index = 0  # single-panel page (angels_of_war_fleet)
    else:
        index = int(target["panel_id"].split("_")[1]) - 1
    cand = candidates[index]
    if cand.panel_bbox.as_xyxy() == (0, 0, 0, 0):
        raise RuntimeError(f"degenerate panel for {target['page']}")
    scene_bbox = derive_scene_crop_bbox(
        cand.panel_bbox,
        page_shape,
        neighboring_panel_bboxes=tuple(c.panel_bbox for c in candidates),
    )
    crop = page_img[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]

    # 2. ObjectPlan carrying the recorded label; only semantic_label drives grounding.
    obj = ObjectPlan(
        object_id=target["object_id"],
        panel_id=target["panel_id"],
        semantic_label=target["semantic_label"],
        confidence=1.0,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind(target["transform_kind"]),
            amplitude=0.02,
        ),
    )
    # panel_bbox_px=None == the object's full-crop panel bbox in the panel pipeline
    # (the per-panel plan's single panel is normalized (0,0,1,1) on the crop).
    grounded = ground_object_candidates(crop, obj, dino, panel_bbox_px=None)
    accepted = grounded[0]  # rank-0 (top score) == the Phase 16 accepted candidate
    print(
        f"[{target['page']} {target['panel_id']}] rank-0 candidate "
        f"bbox={accepted.bbox.as_xyxy()} score={accepted.bbox.score:.4f} "
        f"prompt={_prompt_from_label(target['semantic_label'])!r}"
    )

    # 3. SAM mask on the exact accepted bbox, full-crop shape.
    seg = segment_object(crop, accepted, sam)
    mask = seg.mask
    return mask, crop, {
        "accepted_bbox": accepted.bbox.as_xyxy(),
        "grounding_score": accepted.bbox.score,
        "grounding_prompt": _prompt_from_label(target["semantic_label"]),
        "scene_crop_bbox": scene_bbox.as_xyxy(),
        "panel_bbox": cand.panel_bbox.as_xyxy(),
        "sam_iou_score": seg.iou_score,
        "model_ids": {"grounding": dino.model_id, "segmentation": sam.model_id},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino", type=str, required=True)
    parser.add_argument("--sam", type=str, required=True)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--out", type=Path, default=_OUT_DIR)
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()

    dino = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=device, dtype="float32")
    dino.load()
    sam.load()

    import torch

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    try:
        for target in TARGETS:
            page = Path(target["page"])
            page_img = np.asarray(Image.open(page).convert("RGB"))
            mask, crop, result = _extract_one(
                target, page_img, page_img.shape[:2], dino, sam
            )

            key = page.stem + "_" + target["semantic_label"]
            mask_path = out_dir / f"{key}_mask.npy"
            np.save(mask_path, mask)

            Image.fromarray(crop).save(out_dir / f"{key}_original.png")
            Image.fromarray(_overlay(crop, mask)).save(out_dir / f"{key}_mask_overlay.png")
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
                out_dir / f"{key}_mask_binary.png"
            )

            stats = _mask_stats(mask)
            entries.append(
                {
                    "sample": target["semantic_label"],
                    "source_page": target["page"],
                    "panel": target["panel_id"],
                    "semantic_label": target["semantic_label"],
                    "transform_kind": target["transform_kind"],
                    "mask_path": str(mask_path),
                    "mask_shape": list(mask.shape),
                    "pixel_count": stats["pixel_count"],
                    "mask_density": stats["density"],
                    "tight_bbox": stats["bbox"],
                    **result,
                }
            )
            print(
                f"[{target['page']} {target['panel_id']}] mask shape={mask.shape} "
                f"pixels={stats['pixel_count']} density={stats['density']:.3f} "
                f"tight_bbox={stats['bbox']} saved={mask_path}"
            )
    finally:
        dino.unload()
        sam.unload()

    from manga_animation.evaluation.harness import environment_metadata

    manifest = {
        "task": "phase16_effect_mask_extraction",
        "generated_at": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "environment": environment_metadata(device),
        "gpu_names": gpu_names,
        "config_env": args.env,
        "models": {
            "grounding": "grounding-dino-swin-l (float32)",
            "segmentation": "sam2.1-hiera-base (float32)",
        },
        "method": (
            "extraction-only: deterministic panel geometry (detect_panels + "
            "derive_scene_crop_bbox) -> Grounding DINO rank-0 candidate -> SAM 2.1 "
            "best-iou mask; no VLM, no validation, no animation, no rendering"
        ),
        "matches_phase16_render": (
            "same production grounding/segmentation functions, same models/config, same "
            "scene-crop geometry, and the Phase 16 logs record both objects accepted at "
            "candidate_rank=0 -- see docs/phase16-results.md Run 2 (speed_lines mesh_warp) "
            "and Run 6 (impact_burst radial_expand)"
        ),
        "samples": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
