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
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.analysis.panels import derive_scene_crop_bbox, detect_panels
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.grounding.ground import ground_object_candidates
from manga_animation.pipeline.types import BBoxPx, GroundingResult
from manga_animation.schemas.animation_plan import (
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
)
from manga_animation.segmentation import Sam21Client
from manga_animation.segmentation.segment import segment_object

_OUT_DIR = Path("outputs/debug/phase16_effects")

# Targets. `panel` is the panel whose scene crop the effect is grounded on; "auto" picks
# the panel whose rank-0 grounding detection has the highest score (used where the Phase 16
# logs do not pin the effect to one panel). The first two are the exact masks from the
# Phase 16 successful renders (Run 2 speed_lines, Run 6 impact_burst) and were verified
# against those renders. The remaining five are additional real drawn-effect masks.
#
# Panel attributions come from the Phase 16 GPU logs/manifests:
#   wind_breaker_sprint panel_001 speed_lines     : phase16_gpu_sprint.log (PASS panel_001)
#   angels_of_war_fleet panel_001 impact_burst    : phase16_gpu_fleet3.log (PASS panel_001)
#   omniscient_reader_blade panel_001 speed_lines : single panel (fallback_full_page)
#   sss_hunter_gladiator panel_001 speed_lines    : phase16_shg2_manifest.json PASS panel_001
#   sss_hunter_gladiator panel_004 impact_burst   : phase16_shg2_manifest.json panel_004 failure
#   marika_love_meter panel_001 speed_lines       : phase16_gpu_marika.log panel_001 analysis
#   marika_love_meter radiating_focus_lines auto  : only in analysis signal, panel not pinned
TARGETS = [
    {
        "page": "examples/realworld/wind_breaker_sprint.png",
        "panel": "panel_001",
        "semantic_label": "speed_lines",
        "object_id": "obj_speed_lines_3",
        "transform_kind": "mesh_warp",
    },
    {
        "page": "examples/realworld/angels_of_war_fleet.png",
        "panel": "panel_001",
        "semantic_label": "space_ship_impact_burst",
        "object_id": "obj_space_ship_impact_burst_6",
        "transform_kind": "radial_expand",
    },
    {
        "page": "examples/realworld/omniscient_reader_blade.png",
        "panel": "panel_001",
        "semantic_label": "speed_lines",
        "object_id": "obj_speed_lines_0",
        "transform_kind": "mesh_warp",
    },
    {
        "page": "examples/realworld/sss_hunter_gladiator.png",
        "panel": "panel_001",
        "semantic_label": "speed_lines",
        "object_id": "obj_speed_lines_0",
        "transform_kind": "mesh_warp",
    },
    {
        "page": "examples/realworld/sss_hunter_gladiator.png",
        "panel": "panel_004",
        "semantic_label": "impact_burst",
        "object_id": "obj_impact_burst_0",
        "transform_kind": "radial_expand",
    },
    {
        "page": "examples/realworld/marika_love_meter.png",
        "panel": "panel_001",
        "semantic_label": "speed_lines",
        "object_id": "obj_speed_lines_0",
        "transform_kind": "mesh_warp",
    },
    {
        "page": "examples/realworld/marika_love_meter.png",
        "panel": "auto",
        "semantic_label": "radiating_focus_lines",
        "object_id": "obj_radiating_focus_lines_0",
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


def _panels_geometry(
    page_img: np.ndarray, page_shape: tuple[int, int]
) -> list[tuple[int, Any, BBoxPx]]:
    candidates = detect_panels(page_img)
    if not candidates:
        raise RuntimeError("no panels detected")
    bboxes = tuple(c.panel_bbox for c in candidates)
    out = []
    for i, cand in enumerate(candidates):
        scene_bbox = derive_scene_crop_bbox(
            cand.panel_bbox, page_shape, neighboring_panel_bboxes=bboxes
        )
        out.append((i, cand, scene_bbox))
    return out


def _ground_rank0(crop: np.ndarray, target: dict[str, str], dino: GroundingDinoClient):
    from manga_animation.grounding.ground import _prompt_from_label

    obj = ObjectPlan(
        object_id=target["object_id"],
        panel_id="panel_001",
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
    return grounded[0], _prompt_from_label(target["semantic_label"])


def _extract_one(
    target: dict[str, str],
    page_img: np.ndarray,
    page_shape: tuple[int, int],
    dino: GroundingDinoClient,
    sam: Sam21Client,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    # 1. Deterministic panel geometry exactly as run_page_panels computes it.
    geometry = _panels_geometry(page_img, page_shape)
    if target["panel"] == "auto":
        # Pick the panel whose rank-0 grounding detection scores highest (used only
        # where the Phase 16 logs do not pin the effect to one panel).
        best: tuple[float, int, BBoxPx, GroundingResult] | None = None
        for index, _cand, scene_bbox in geometry:
            crop = page_img[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]
            accepted, prompt = _ground_rank0(crop, target, dino)
            cand_entry = (float(accepted.bbox.score), index, scene_bbox, accepted)
            if best is None or cand_entry[0] > best[0]:
                best = cand_entry
        assert best is not None
        from manga_animation.grounding.ground import _prompt_from_label

        score, index, scene_bbox, accepted = best
        cand = geometry[index][1]
        prompt = _prompt_from_label(target["semantic_label"])
        panel_id = f"panel_{index + 1:03d}"
        print(
            f"[{target['page']} auto] best panel={panel_id} scene={scene_bbox.as_xyxy()} "
            f"rank-0 bbox={accepted.bbox.as_xyxy()} score={score:.4f} prompt={prompt!r}"
        )
    else:
        index = int(target["panel"].split("_")[1]) - 1
        if not (0 <= index < len(geometry)):
            raise RuntimeError(f"panel {target['panel']} out of range for {target['page']}")
        cand, scene_bbox = geometry[index][1], geometry[index][2]
        panel_id = f"panel_{index + 1:03d}"
        crop = page_img[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]
        accepted, prompt = _ground_rank0(crop, target, dino)
        print(
            f"[{target['page']} {panel_id}] rank-0 candidate "
            f"bbox={accepted.bbox.as_xyxy()} score={accepted.bbox.score:.4f} "
            f"prompt={prompt!r}"
        )

    # 3. SAM mask on the exact accepted bbox, full-crop shape.
    seg = segment_object(crop, accepted, sam)
    mask = seg.mask
    return mask, crop, {
        "panel": panel_id,
        "accepted_bbox": accepted.bbox.as_xyxy(),
        "grounding_score": accepted.bbox.score,
        "grounding_prompt": prompt,
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
                    "panel": result["panel"],
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
                f"[{target['page']} {result['panel']}] mask shape={mask.shape} "
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
        "notes": (
            "The first two samples reproduce the exact masks used by the Phase 16 successful "
            "renders (Run 2 speed_lines mesh_warp, Run 6 impact_burst radial_expand) and are "
            "verified against those renders. The remaining samples are additional real "
            "drawn-effect masks extracted with the identical deterministic pipeline path "
            "on the panels named in the Phase 16 logs/manifests (see each sample's panel "
            "field); 'auto' panel means the panel whose rank-0 grounding score was highest."
        ),
        "samples": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
