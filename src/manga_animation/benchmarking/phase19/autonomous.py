"""Phase 19 autonomous mode: full page + generic instruction -> OMG-LLaVA -> target + mask.

The autonomous experiment (phase brief section 4/18) deliberately gives the model the FULL page
and a GENERIC animation-target instruction -- no GT bbox, no GT mask, no target crop, no
GT-derived description. The model must understand the scene, identify an animation-worthy
element, and emit its mask. This module runs that experiment over a representative subset of
the benchmark pages, saves every artifact, and builds the visual gallery for the qualitative
review (semantic plausibility / animation suitability / instance correctness / safety / mask
usability). Those five judgments are inherently qualitative and are recorded as an editable
review template for the human reviewer -- never auto-scored by a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase19.masks import tight_bbox_from_mask
from manga_animation.benchmarking.phase19.parsing import target_context
from manga_animation.benchmarking.phase19.run import (
    _reset_vram_peak,
    _vram_peak_mb,
    masks_to_original,
)

# The five qualitative criteria the human reviewer fills in per page (section 18 A-E). They
# are deliberately not auto-scored: each is a visual judgment.
REVIEW_CRITERIA = (
    "semantic_plausibility",   # does the selected target correspond to depicted dynamics?
    "animation_suitability",   # would animating this target make visual sense?
    "instance_coherent",       # is the selected object a coherent specific instance?
    "safe",                    # does the mask avoid forbidden regions?
    "mask_usable",             # could the mask feed the existing deterministic pipeline?
)


@dataclass
class AutonomousPageRecord:
    """One autonomous page's artifacts + the (initially empty) human review."""

    page_key: str
    image_path: str
    prompt: str
    status: str  # "ok" | "inference_error"
    error_detail: str | None = None
    output_text: str = ""
    target_context: str | None = None
    n_masks: int = 0
    mask_paths: list[str] = field(default_factory=list)
    bbox: list[int] | None = None
    latency_seconds: float = 0.0
    vram_peak_mb: float | None = None
    review: dict[str, Any] = field(
        default_factory=lambda: {c: None for c in REVIEW_CRITERIA}
    )
    failure_category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_key": self.page_key,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "status": self.status,
            "error_detail": self.error_detail,
            "output_text": self.output_text,
            "target_context": self.target_context,
            "n_masks": self.n_masks,
            "mask_paths": self.mask_paths,
            "bbox": self.bbox,
            "latency_seconds": self.latency_seconds,
            "vram_peak_mb": self.vram_peak_mb,
            "review": self.review,
            "failure_category": self.failure_category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutonomousPageRecord:
        return cls(
            page_key=data["page_key"],
            image_path=data["image_path"],
            prompt=data["prompt"],
            status=data["status"],
            error_detail=data.get("error_detail"),
            output_text=data.get("output_text", ""),
            target_context=data.get("target_context"),
            n_masks=int(data.get("n_masks", 0)),
            mask_paths=list(data.get("mask_paths", [])),
            bbox=data.get("bbox"),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            vram_peak_mb=data.get("vram_peak_mb"),
            review=dict(data.get("review", {})),
            failure_category=data.get("failure_category"),
        )


def unique_pages(manifest) -> list[str]:
    """Deterministic, manifest-ordered list of the benchmark's unique pages."""
    seen: list[str] = []
    for sample in manifest.samples:
        key = f"{sample.book}_{sample.page_index:03d}"
        if key not in seen:
            seen.append(key)
    return seen


def run_autonomous_pages(
    manifest,
    dataset_dir: Path,
    out_dir: Path,
    adapter,
    *,
    instruction: str,
    limit: int | None = None,
) -> list[AutonomousPageRecord]:
    """Run the autonomous experiment over the manifest's unique pages (up to `limit`). The model
    stays loaded; each page saves page / text / masks / bbox."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = unique_pages(manifest)
    if limit is not None:
        pages = pages[:limit]
    records: list[AutonomousPageRecord] = []
    for i, page_key in enumerate(pages):
        # The first manifest sample on the page carries the page image file name.
        sample = next(s for s in manifest.samples if f"{s.book}_{s.page_index:03d}" == page_key)
        image = np.asarray(Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB"))
        _reset_vram_peak()
        error: Exception | None = None
        out = None
        try:
            out = adapter.predict(image, instruction)
        except Exception as exc:  # noqa: BLE001 -- a failed page is recorded, never dropped
            error = exc
        latency = getattr(out, "latency_seconds", 0.0) if out is not None else 0.0
        vram = _vram_peak_mb()

        mask_paths: list[str] = []
        bbox = None
        masks = masks_to_original(out.masks, tuple(image.shape[:2])) if out is not None else []
        for j, mask in enumerate(masks):
            path = out_dir / f"{page_key}.autonomous.mask{j}.npz"
            np.savez_compressed(path, mask=mask.astype(np.uint8) * 255)
            mask_paths.append(str(path))
        if masks:
            try:
                bbox = list(tight_bbox_from_mask(masks[0]))
            except ValueError:
                bbox = None
        if out is not None:
            Image.fromarray(image).save(out_dir / f"{page_key}.autonomous.page.png")
            (out_dir / f"{page_key}.autonomous.text.txt").write_text(out.text, encoding="utf-8")

        records.append(
            AutonomousPageRecord(
                page_key=page_key,
                image_path=str(out_dir / f"{page_key}.autonomous.page.png"),
                prompt=instruction,
                status="inference_error" if error is not None else "ok",
                error_detail=f"{type(error).__name__}: {error}" if error is not None else None,
                output_text=out.text if out is not None else "",
                target_context=target_context(out.text) if out is not None else None,
                n_masks=len(masks),
                mask_paths=mask_paths,
                bbox=bbox,
                latency_seconds=latency,
                vram_peak_mb=vram,
            )
        )
        if (i + 1) % 5 == 0 or i == len(pages) - 1:
            print(f"[phase19] autonomous {i + 1}/{len(pages)} pages done")
    return records


def save_autonomous_records(records: list[AutonomousPageRecord], out_dir: Path) -> Path:
    path = out_dir / "autonomous_pages.json"
    path.write_text(json.dumps([r.as_dict() for r in records], indent=2), encoding="utf-8")
    return path


def build_autonomous_gallery(records: list[AutonomousPageRecord], out_dir: Path) -> list[Path]:
    """Montage per page: original + the first predicted mask overlay (others when present), for
    the human reviewer. Purely CPU (masks are on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in records:
        image = np.asarray(Image.open(record.image_path).convert("RGB"))
        h, w = image.shape[:2]
        scale = min(1.0, 560.0 / max(h, w))
        small = Image.fromarray(image).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
        small_arr = np.asarray(small)
        sh, sw = small_arr.shape[:2]
        panels = [small_arr.copy()]
        for path in record.mask_paths:
            if not Path(path).exists():
                continue
            mask = np.load(path)["mask"] > 0
            resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                (sw, sh), Image.Resampling.NEAREST
            )
            overlay = small_arr.copy()
            small_mask = np.asarray(resized) > 0
            overlay[small_mask] = (
                overlay[small_mask] * 0.6 + np.array([255, 80, 80]) * 0.4
            ).astype(np.uint8)
            panels.append(overlay)
        n = len(panels)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        canvas = np.zeros((sh * rows, sw * cols, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            r, c = divmod(i, cols)
            canvas[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw] = panel
        out_path = out_dir / f"autonomous_{record.page_key}.png"
        Image.fromarray(canvas).save(out_path)
        written.append(out_path)
    return written
