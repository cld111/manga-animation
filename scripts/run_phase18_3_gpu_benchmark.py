"""Phase 18.3 GPU benchmark: the per-candidate VLM object-description stage on REAL data.

Runs on the remote Kaggle worker (real Qwen2.5-VL inference; never locally). Three parts:

  A. Coordinate-contract verification: `object_description.prompt.prepare_image_and_bbox`
     must produce exactly the image the Qwen2.5-VL processor will actually feed the model
     (same smart_resize rounding), so the bbox coordinates stated in the prompt match the
     pixel space the model sees. Compares our replication against the REAL processor output
     dims for a set of shapes (including the pathological tall-page case).
  B. Curated scenario pages (deterministic synthetic manga-like pages) covering the task
     brief's ten required cases -- single object, several nearby objects, a bbox containing
     several objects, a partial bbox, an occluded object, a small object, a complex
     background, several visually similar objects, an object DINO would find but that is bad
     for animation (text/rigid), and a partially-animatable object. Each is described by the
     REAL Qwen2.5-VL through the production `describe_object` entry point.
  C. Real project pages: real Grounding DINO detections (top candidates) for known
     semantic labels are fed through the same `describe_object` call -- the actual
     production candidate path.

The VLM client receives the FULL image plus bbox coordinates in every call (never a crop,
never a bbox visualization) -- part A proves the coordinate correspondence, parts B/C record
the raw prompt+image sizes for audit.

Usage (on the worker):

    python scripts/run_phase18_3_gpu_benchmark.py \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --pages examples/realworld/*.png \
        --out outputs/experiments/phase18_3_<ts>.json

Writes the JSON report plus a visual package (page + drawn bbox per case) next to it, all
under git-ignored `outputs/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from manga_animation.analysis.client import Qwen25VLClient, VLMClient
from manga_animation.grounding.client import GroundingDinoClient
from manga_animation.object_description.describe import describe_object
from manga_animation.object_description.prompt import prepare_image_and_bbox
from manga_animation.pipeline.types import BBoxPx
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
    Vector2,
)

SCENARIO_LABELS: dict[str, str] = {
    "single_object": "single unambiguous object",
    "several_nearby": "several objects near each other",
    "multi_object_box": "bbox containing several objects",
    "partial_object": "bbox partially covering the object",
    "occluded_object": "object partially occluded by another",
    "small_object": "small object",
    "complex_background": "object on a complex background",
    "similar_objects": "several visually similar objects",
    "text_rigid": "object DINO might find but bad for animation (text/rigid)",
    "partially_animatable": "object animatable only partially",
}


@dataclass(frozen=True, slots=True)
class CuratedCase:
    """One scenario: a synthetic page + a bbox + the semantic label to ask about."""

    case_id: str
    semantic_label: str
    page: np.ndarray
    bbox: tuple[int, int, int, int]


def _fill(page: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    page[y0:y1, x0:x1] = color


def _draw_character(
    page: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """A crude but visually distinct 'character': head circle + body rectangle + hair cap."""
    cx, cy = center
    body_w, body_h = size, int(size * 1.4)
    _fill(page, (cx - body_w // 2, cy, cx + body_w // 2, cy + body_h), color)
    head_r = size // 3
    yy, xx = np.mgrid[max(0, cy - head_r) : cy + head_r, max(0, cx - head_r) : cx + head_r]
    if yy.size == 0 or xx.size == 0:
        return
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= head_r**2
    sub = page[max(0, cy - head_r) : cy + head_r, max(0, cx - head_r) : cx + head_r]
    sub[mask] = color
    # hair cap (top third of the head)
    hair_h = max(1, head_r // 2)
    _fill(page, (cx - head_r, cy - head_r, cx + head_r, cy - head_r + hair_h), (60, 40, 30))


def _build_scenario_pages() -> list[CuratedCase]:
    """Deterministic synthetic pages for every required scenario. Each case's bbox is the
    *candidate* the pipeline asks the VLM about -- some intentionally bad (that is the point
    of the semantic validation layer)."""
    cases: list[CuratedCase] = []
    W, H = 720, 640

    def new_page() -> np.ndarray:
        return np.full((H, W, 3), (245, 244, 238), dtype=np.uint8)

    # 1. Single unambiguous object: one character in the middle of a plain page.
    page = new_page()
    _draw_character(page, (360, 200), 90, (180, 60, 60))
    cases.append(CuratedCase("single_object", "character", page, (300, 160, 420, 400)))

    # 2. Several objects near each other: two characters, candidate box around the left one.
    page = new_page()
    _draw_character(page, (280, 200), 90, (180, 60, 60))
    _draw_character(page, (470, 220), 70, (60, 60, 180))
    cases.append(CuratedCase("several_nearby", "character", page, (210, 150, 350, 420)))

    # 3. A bbox that CONTAINS several objects: box spanning both characters.
    cases.append(
        CuratedCase(
            "multi_object_box", "character", page.copy(), (180, 140, 560, 440)
        )
    )

    # 4. Partial bbox: only the character's head (top part of the figure).
    cases.append(CuratedCase("partial_object", "character", page.copy(), (300, 130, 420, 240)))

    # 5. Occluded object: a character half-covered by a large foreground box.
    page = new_page()
    _draw_character(page, (360, 220), 90, (180, 60, 60))
    _fill(page, (260, 260, 460, 420), (90, 90, 100))  # occluding slab
    cases.append(CuratedCase("occluded_object", "character", page, (300, 160, 420, 430)))

    # 6. Small object: tiny character on a large page.
    page = new_page()
    _draw_character(page, (360, 300), 26, (180, 60, 60))
    cases.append(CuratedCase("small_object", "character", page, (345, 285, 375, 360)))

    # 7. Complex background: character over dense noise/pattern.
    page = new_page()
    rng = np.random.default_rng(7)
    page[:] = rng.integers(150, 245, size=(H, W, 3), dtype=np.uint8)
    _draw_character(page, (360, 250), 90, (180, 60, 60))
    cases.append(CuratedCase("complex_background", "character", page, (300, 190, 420, 440)))

    # 8. Several visually similar objects: three same-colored characters; box around one.
    page = new_page()
    _draw_character(page, (180, 250), 80, (180, 60, 60))
    _draw_character(page, (360, 250), 80, (180, 60, 60))
    _draw_character(page, (540, 250), 80, (180, 60, 60))
    cases.append(
        CuratedCase("similar_objects", "character", page, (300, 200, 420, 380))
    )

    # 9. Text / rigid content (a banner of lettering) -- DINO-style detection could fire on
    #    text; the VLM must say not_animatable / reject rather than propose animating it.
    page = new_page()
    img = Image.fromarray(page)
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 300, 600, 360], outline=(0, 0, 0), width=3)
    draw.text((150, 312), "ATTACK!  SKILL!", fill=(0, 0, 0))
    page = np.asarray(img)
    cases.append(CuratedCase("text_rigid", "text_banner", page, (120, 300, 600, 360)))

    # 10. Partially animatable: a character whose visible part could move but whose lower
    #     body is fused into a static wall -- the VLM should report the constraint.
    page = new_page()
    _draw_character(page, (360, 250), 90, (180, 60, 60))
    _fill(page, (280, 380, 440, 460), (120, 110, 100))  # wall fused with the lower body
    cases.append(
        CuratedCase(
            "partially_animatable", "character", page, (300, 190, 420, 440)
        )
    )

    return cases


def _object_plan(label: str, case_id: str) -> ObjectPlan:
    return ObjectPlan(
        object_id=f"obj_{case_id}",
        panel_id="panel_1",
        semantic_label=label,
        confidence=0.9,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.02,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
        ),
    )


def _record_call(
    client: VLMClient,
    result: Any,
    *,
    page: np.ndarray,
    bbox: BBoxPx,
    prompt: str,
    prepared_image: Image.Image,
) -> dict[str, Any]:
    return {
        "accepted": result.accepted,
        "assessment": result.assessment,
        "matches_semantic_label": result.matches_semantic_label,
        "animatable": result.animatable,
        "object_identity": result.object_identity,
        "motion": (
            result.motion_spec.model_dump() if result.motion_spec is not None else None
        ),
        "movable_parts": list(result.movable_parts),
        "static_parts": list(result.static_parts),
        "constraints": list(result.constraints),
        "neighbor_conflicts": list(result.neighbor_conflicts),
        "confidence": result.confidence,
        "reason": result.reason,
        "rejection_reason": result.rejection_reason,
        "model_id": result.model_id,
        "raw_responses": list(result.raw_responses),
        # The input contract the brief requires verifying: the VLM saw the FULL image at this
        # size, the bbox was given in ITS pixel coordinates (never a crop/visualization).
        "vlm_image_size": (prepared_image.width, prepared_image.height),
        "bbox_in_vlm_pixels": list(bbox.as_xyxy()),
        "prompt_excerpt": prompt[:400],
    }


def _visualize(page: np.ndarray, bbox: BBoxPx, path: Path) -> None:
    img = Image.fromarray(page).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox.as_xyxy(), outline=(255, 0, 0), width=4)
    img.save(path)


def run_part_a(qwen_source: str, out_dir: Path) -> list[dict[str, Any]]:
    """Coordinate contract: our prepare_image_and_bbox vs the REAL processor's smart_resize."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(qwen_source)
    sizes = [(1024, 1536), (720, 5062), (600, 400), (1536, 1536), (218, 1536), (200, 220)]
    records: list[dict[str, Any]] = []
    for w, h in sizes:
        img = Image.new("RGB", (w, h), (240, 240, 245))
        inputs = processor(text=[""], images=[img], return_tensors="pt")
        grid = inputs["image_grid_thw"][0].tolist()  # (t, grid_h, grid_w)
        processor_dims = (grid[1] * 14, grid[2] * 14)
        prepared = prepare_image_and_bbox(
            img, BBoxPx(10, 10, w // 2, h // 2), max_long_edge=1536
        )
        ours = prepared.image.size
        match = tuple(ours) == processor_dims
        records.append(
            {
                "input_size": (w, h),
                "processor_grid_dims": processor_dims,
                "ours_dims": ours,
                "match": match,
            }
        )
        print(f"  part A: {w}x{h} processor={processor_dims} ours={ours} match={match}")
    (out_dir / "part_a_processor_contract.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )
    return records


def run_part_b(vlm_client: VLMClient, out_dir: Path) -> list[dict[str, Any]]:
    """Curated scenario pages through the production describe_object call."""
    records: list[dict[str, Any]] = []
    for case in _build_scenario_pages():
        bbox = BBoxPx(*case.bbox, score=0.9)
        prepared = prepare_image_and_bbox(
            Image.fromarray(case.page), bbox, max_long_edge=1536
        )
        from manga_animation.object_description.prompt import build_prompt

        prompt = build_prompt(prepared=prepared, semantic_label=case.semantic_label)
        from manga_animation.object_description.describe import describe_object

        result = describe_object(
            case.page,
            bbox,
            _object_plan(case.semantic_label, case.case_id),
            vlm_client,
            max_long_edge=1536,
        )
        record = _record_call(
            vlm_client,
            result,
            page=case.page,
            bbox=bbox,
            prompt=prompt,
            prepared_image=prepared.image,
        )
        record["case_id"] = case.case_id
        record["scenario"] = SCENARIO_LABELS[case.case_id]
        record["semantic_label"] = case.semantic_label
        records.append(record)
        _visualize(case.page, bbox, out_dir / f"part_b_{case.case_id}.png")
        print(
            f"  part B {case.case_id}: accepted={result.accepted} "
            f"assessment={result.assessment} confidence={result.confidence}"
        )
    return records


def run_part_c(
    vlm_client: VLMClient,
    grounding_client: GroundingDinoClient,
    pages: list[Path],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Real pages: real DINO detections fed through the production describe_object call."""
    from manga_animation.grounding.ground import ground_object_candidates

    records: list[dict[str, Any]] = []
    for page_path in pages:
        image = np.asarray(Image.open(page_path).convert("RGB"))
        for label in ("character", "weapon", "flag", "speed lines"):
            try:
                candidates = ground_object_candidates(
                    image,
                    _object_plan(label, f"{page_path.stem}_{label}"),
                    grounding_client,
                    max_candidates=3,
                )
            except Exception as exc:  # noqa: BLE001 -- a missing detection is a normal outcome
                print(f"  part C {page_path.stem}/{label}: no detection ({type(exc).__name__})")
                continue
            for rank, candidate in enumerate(candidates):
                result = describe_object(
                    image,
                    candidate.bbox,
                    _object_plan(label, f"{page_path.stem}_{label}_{rank}"),
                    vlm_client,
                    max_long_edge=1536,
                )
                record = _record_call(
                    vlm_client,
                    result,
                    page=image,
                    bbox=candidate.bbox,
                    prompt="(built inside describe_object)",
                    prepared_image=prepare_image_and_bbox(
                        Image.fromarray(image), candidate.bbox, max_long_edge=1536
                    ).image,
                )
                record["page"] = page_path.name
                record["semantic_label"] = label
                record["dino_rank"] = rank
                record["dino_bbox"] = list(candidate.bbox.as_xyxy())
                record["dino_score"] = candidate.bbox.score
                records.append(record)
                _visualize(
                    image, candidate.bbox, out_dir / f"part_c_{page_path.stem}_{label}_{rank}.png"
                )
                print(
                    f"  part C {page_path.stem}/{label} rank={rank} score="
                    f"{candidate.bbox.score:.3f}: accepted={result.accepted} "
                    f"assessment={result.assessment}"
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", required=True, help="local Qwen2.5-VL model dir on the worker")
    parser.add_argument("--dino", required=True, help="local Grounding DINO model dir")
    parser.add_argument("--pages", nargs="*", default=[], help="real project pages")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--parts", default="abc", help="which parts to run (default: all)"
    )
    parser.add_argument("--resolution", type=int, default=1536)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    report: dict[str, Any] = {
        "phase": "18.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parts": {},
    }

    vlm_client = Qwen25VLClient(source=args.qwen, dtype="float16", max_new_tokens=768)

    try:
        if "a" in args.parts:
            report["parts"]["a_processor_contract"] = run_part_a(args.qwen, out_dir)

        if "b" in args.parts:
            # Qwen25VLClient lazy-loads inside its first generate() call.
            report["parts"]["b_curated"] = run_part_b(vlm_client, out_dir)

        if "c" in args.parts:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            grounding_client = GroundingDinoClient(
                source=args.dino, device=device, dtype="float32"
            )
            try:
                grounding_client.load()
                report["parts"]["c_real_pages"] = run_part_c(
                    vlm_client, grounding_client, [Path(p) for p in args.pages], out_dir
                )
            finally:
                grounding_client.unload()
    finally:
        vlm_client.unload()

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
