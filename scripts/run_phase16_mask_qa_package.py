"""Phase 16 human-QA package generator for the 13-sample semantic-mask benchmark.

Produces a self-contained visual QA dataset under `outputs/phase16_mask_qa/` so a human can
answer, per sample, the five mask-quality questions:
  1. What is the intended animation target?
  2. Is the target actually present?
  3. Does the mask cover the target sufficiently?
  4. Does the mask include unrelated content (objects/background/text/faces/borders)?
  5. Is the mask safe to animate under the project's invariants?

Per sample, the package includes (all pure local CPU, reusing cached artifacts -- NO GPU, NO
model calls, NO src/ changes):
  - {sample_id}/page_with_bbox_and_mask.png : full page with bbox rectangle + mask contour
  - {sample_id}/crop_with_mask_overlay.png  : crop around bbox, mask shown semi-transparent
  - {sample_id}/crop_mask_contour.png       : crop with mask boundary drawn
  - {sample_id}/mask_binary.png             : the binary mask alone (inverted for visibility)
  - {sample_id}/hole_mask.png               : reconstruction hole mask, when cached
  - {sample_id}/checklist.txt               : human checklist/form with the project labels
  - labels_template.json                    : machine-readable template to fill in
  - MANUAL_LABELING.md                      : how to perform the labeling

The machine-readable template carries the current semantic-gate decision (old/new prompt
benchmark outputs, when present) purely as context -- the human verdict is authoritative and
must be recorded, not the model's.

Mandatory review cases (flagged in the summary): raised_sword_12, character_eyes_2, cloth_5.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

_BENCHMARK = Path("configs/phase12_semantic_mask_benchmark.yaml")
_OUT = Path("outputs/phase16_mask_qa")
BBox4 = tuple[int, int, int, int]
_MANDATORY = {
    "villainess_ending_scuffle_obj_raised_sword_12",
    "sss_hunter_gladiator_obj_character_eyes_2",
    "villainess_ending_scuffle_obj_cloth_5",
}

_FAILURE_OPTIONS = [
    "NONE",
    "UNDER_SEGMENTATION",
    "OVER_SEGMENTATION",
    "BACKGROUND_CONTAMINATION",
    "OTHER_OBJECT_CONTAMINATION",
    "TEXT_CONTAMINATION",
    "FACE_CONTAMINATION",
    "BORDER_CONTAMINATION",
    "OTHER",
]


def _load_benchmark(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())["samples"]


def _load_gate_decision(sample_id: str) -> dict:
    """Best-effort: current semantic-gate prediction + reason from the Phase 16 benchmark
    re-run, if the JSON artifact is present. Returns {} when absent (context only)."""
    from pathlib import Path as P

    cand = P("outputs/experiments/phase12_benchmark_newprompt.json")
    if not cand.exists():
        return {}
    try:
        data = json.loads(cand.read_text())
    except json.JSONDecodeError:
        return {}
    for rep in data.get("reports", []):
        if "vlm" not in rep.get("method", ""):
            continue
        for p in rep.get("predictions", []):
            if p.get("sample_id") == sample_id:
                return {
                    "gate_prediction": p.get("predicted"),
                    "gate_confidence": p.get("score"),
                    "gate_reason": p.get("reason"),
                }
    return {}


def _page_overlay(page: np.ndarray, mask: np.ndarray, bbox: BBox4) -> np.ndarray:
    """Page-sized view: bbox rectangle + mask contour. Mask contour uses the original
    page resolution (mask is full-source-image-shape)."""
    vis = page.copy()
    x0, y0, x1, y1 = bbox
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis


def _crop_overlay(page: np.ndarray, mask: np.ndarray, bbox: BBox4,
                    margin: float = 0.08) -> np.ndarray:
    """Crop around bbox + margin; mask drawn semi-transparent red over the image."""
    h, w = page.shape[:2]
    x0, y0, x1, y1 = bbox
    mx = max(1, int((x1 - x0) * margin))
    my = max(1, int((y1 - y0) * margin))
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(w, x1 + mx), min(h, y1 + my)
    crop = page[cy0:cy1, cx0:cx1].copy()
    mask_crop = mask[cy0:cy1, cx0:cx1] > 0
    overlay = crop.copy()
    overlay[mask_crop] = (255, 0, 0)
    blend = cv2.addWeighted(crop, 0.6, overlay, 0.4, 0)
    return blend


def _crop_contour(page: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int],
                  margin: float = 0.08) -> np.ndarray:
    """Crop with only the mask boundary drawn (green) on the plain image."""
    h, w = page.shape[:2]
    x0, y0, x1, y1 = bbox
    mx = max(1, int((x1 - x0) * margin))
    my = max(1, int((y1 - y0) * margin))
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(w, x1 + mx), min(h, y1 + my)
    crop = page[cy0:cy1, cx0:cx1].copy()
    mask_crop = mask[cy0:cy1, cx0:cx1]
    contours, _ = cv2.findContours((mask_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)
    return crop


def _mask_binary(mask: np.ndarray) -> np.ndarray:
    """The mask alone: white on black, cropped to its tight bbox + small margin."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = 8
    h, w = mask.shape
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    sub = (mask[cy0:cy1, cx0:cx1] > 0).astype(np.uint8) * 255
    return cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)


def _hole_mask(mask: np.ndarray, hole: np.ndarray | None, bbox: BBox4) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    pad = 8
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    canvas = np.zeros((cy1 - cy0, cx1 - cx0, 3), dtype=np.uint8)
    canvas[(mask[cy0:cy1, cx0:cx1] > 0)] = (255, 255, 255)  # object = white
    if hole is not None:
        hsub = hole[cy0:cy1, cx0:cx1] > 0
        canvas[hsub] = (0, 0, 255)  # hole = red
    return canvas


def _checklist_text(sample: dict, gate: dict) -> str:
    lines = []
    lines.append(f"SAMPLE: {sample['sample_id']}")
    lines.append(f"SEMANTIC LABEL (intended target): {sample['semantic_label']}")
    lines.append(f"TRANSFORM: {sample['transform_kind']}")
    lines.append(f"BENCHMARK GROUND TRUTH: {sample['ground_truth']} "
                 f"(difficulty: {sample['difficulty']})")
    lines.append("")
    lines.append("SEMANTIC GATE (current prompt, CONTEXT ONLY -- your verdict is authoritative):")
    if gate:
        lines.append(f"  prediction={gate.get('gate_prediction')} "
                     f"confidence={gate.get('gate_confidence')}")
        lines.append(f"  reason={gate.get('gate_reason')}")
    else:
        lines.append("  (no cached gate decision artifact found)")
    lines.append("")
    lines.append("EVIDENCE (from benchmark config):")
    lines.append(f"  {sample['evidence']}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("HUMAN LABELING FORM")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Look at the images in this directory and answer:")
    lines.append("")
    label = sample["semantic_label"]
    lines.append(f"Q1. TARGET_PRESENT: is the intended target ('{label}') "
                 "actually present in the image?")
    lines.append("    YES / NO / UNCERTAIN")
    lines.append("")
    lines.append("Q2. MASK_QUALITY: is the mask's coverage of the target GOOD or BAD?")
    lines.append("    GOOD / BAD / UNCERTAIN")
    lines.append("")
    lines.append("Q3. FAILURE_TYPE (choose the single most relevant; NONE if the mask is clean):")
    for i, f in enumerate(_FAILURE_OPTIONS):
        lines.append(f"    {i}. {f}")
    lines.append("")
    lines.append("Q4. ANIMATION_SAFE: under the project invariants (preserve pixels outside the")
    lines.append("    mask; never animate text/faces/borders; deterministic local motion), is this")
    lines.append(f"    mask safe to animate with a {sample['transform_kind']} transform?")
    lines.append("    YES / NO / UNCERTAIN")
    lines.append("")
    mandatory = "YES" if sample["sample_id"] in _MANDATORY else "no"
    lines.append(f"MANDATORY REVIEW CASE: {mandatory}")
    return "\n".join(lines)


def main() -> None:
    samples = _load_benchmark(_BENCHMARK)
    _OUT.mkdir(parents=True, exist_ok=True)

    labels_template: list[dict] = []
    mandatory_found: list[str] = []

    for s in samples:
        sid = s["sample_id"]
        page = cv2.imread(s["source_page"])
        mask = np.load(s["mask_path"])
        assert page is not None, f"cannot read page {s['source_page']}"
        assert page.shape[:2] == mask.shape[:2], (
            f"page {page.shape[:2]} != mask {mask.shape[:2]} for {sid}"
        )
        bbox: BBox4 = (
            int(s["bbox_xyxy"][0]), int(s["bbox_xyxy"][1]),
            int(s["bbox_xyxy"][2]), int(s["bbox_xyxy"][3]),
        )
        gate = _load_gate_decision(sid)

        # hole mask, if cached
        hole_path = Path(s["mask_path"]).with_name(
            Path(s["mask_path"]).name.replace("_mask.npy", "_hole_mask.npy")
        )
        hole = np.load(hole_path) if hole_path.exists() else None

        out_dir = _OUT / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "page_with_bbox_and_mask.png"),
                    _page_overlay(page, mask, bbox))
        cv2.imwrite(str(out_dir / "crop_with_mask_overlay.png"),
                    _crop_overlay(page, mask, bbox))
        cv2.imwrite(str(out_dir / "crop_mask_contour.png"),
                    _crop_contour(page, mask, bbox))
        cv2.imwrite(str(out_dir / "mask_binary.png"), _mask_binary(mask))
        if hole is not None:
            cv2.imwrite(str(out_dir / "hole_mask.png"), _hole_mask(mask, hole, bbox))
        (out_dir / "checklist.txt").write_text(_checklist_text(s, gate))

        if sid in _MANDATORY:
            mandatory_found.append(sid)

        labels_template.append(
            {
                "sample_id": sid,
                "semantic_label": s["semantic_label"],
                "transform_kind": s["transform_kind"],
                "benchmark_ground_truth": s["ground_truth"],
                "difficulty": s["difficulty"],
                "mandatory_review": sid in _MANDATORY,
                "TARGET_PRESENT": None,  # YES / NO / UNCERTAIN
                "MASK_QUALITY": None,  # GOOD / BAD / UNCERTAIN
                "FAILURE_TYPE": None,  # one of _FAILURE_OPTIONS
                "ANIMATION_SAFE": None,  # YES / NO / UNCERTAIN
                "notes": "",
            }
        )

    (_OUT / "labels_template.json").write_text(
        json.dumps({"samples": labels_template}, indent=2) + "\n"
    )

    # Summary
    summary = [
        "# Phase 16 semantic-mask human-QA package",
        "",
        f"Generated locally from cached artifacts ({len(samples)} samples).",
        "No GPU, no model calls.",
        "",
        "## Mandatory review cases",
        "",
    ]
    for sid in sorted(_MANDATORY):
        mark = "PRESENT" if sid in mandatory_found else "**MISSING from benchmark**"
        summary.append(f"- `{sid}` -- {mark}")
    missing_mand = sorted(_MANDATORY - set(mandatory_found))
    if missing_mand:
        summary.append(f"\nMISSING mandatory samples: {missing_mand}")

    summary += [
        "",
        "## How to label",
        "",
        "1. Open `labels_template.json` (or use the per-sample `checklist.txt`).",
        "2. For each sample, look at the four images:",
        "   - `page_with_bbox_and_mask.png` (context: where on the page)",
        "   - `crop_with_mask_overlay.png` (mask semi-transparent over artwork)",
        "   - `crop_mask_contour.png` (mask boundary on the artwork)",
        "   - `mask_binary.png` (mask alone)",
        "3. Answer the five questions in `checklist.txt` / the JSON.",
        "4. Mandatory cases must be labeled first and reviewed:",
        "   raised_sword_12, character_eyes_2, cloth_5.",
        "",
        "Definitions:",
        "- TARGET_PRESENT: is the intended object/effect drawn in the image at all?",
        "- MASK_QUALITY: does the mask cover the target well (GOOD) or poorly/not (BAD)?",
        "- FAILURE_TYPE: single most relevant defect, or NONE if clean.",
        "- ANIMATION_SAFE: YES if animating this mask would respect the project invariants.",
        "",
        "Do NOT change your verdict to match the benchmark ground_truth or the gate prediction;",
        "your visual judgment is the authority here.",
    ]
    (_OUT / "MANUAL_LABELING.md").write_text("\n".join(summary) + "\n")

    print(f"Wrote QA package to {_OUT}")
    print(f"Samples: {len(samples)} | mandatory present: {mandatory_found}")
    for sid in sorted(_MANDATORY):
        print(f"  mandatory: {sid} -> {'PRESENT' if sid in mandatory_found else 'MISSING'}")


if __name__ == "__main__":
    main()
