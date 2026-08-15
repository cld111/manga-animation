"""Phase 17 report: aggregation, category/safety analysis, visual failure packages.

Everything here is CPU-side, runs after the GPU experiments finished (their masks/intermediates
are already on disk), and produces the machine-readable results plus the markdown report the
phase brief's final-report section requires. Distributions always carry median/percentiles and
a failure count -- never a bare mean over silently-dropped samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.dataset import (
    CATEGORY_IDS,
    decode_rle,
    ms92_book_annotations,
)
from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase17.metrics import (
    Distribution,
    compute_distribution,
)
from manga_animation.benchmarking.phase17.run import SampleResult

# A top detection whose bbox IoU with the GT box is below this is treated as a
# wrong-object/wrong-instance selection for the Exp-B diagnosis (it did not find OUR instance).
_WRONG_OBJECT_BBOX_IOU = 0.30


@dataclass
class PerSampleMetrics:
    sample_id: str
    category: str
    a_iou: float | None
    a_dice: float | None
    b_bbox_iou: float | None
    b_gt_coverage: float | None
    b_area_ratio: float | None
    c_iou: float | None
    c_dice: float | None
    c_precision: float | None
    c_recall: float | None
    c_outcome: str
    c_failure_detail: str | None = None
    gap_a_minus_c: float | None = None


@dataclass
class BenchmarkReport:
    """All numeric evidence for one full benchmark run, per experiment and per category."""

    n_samples: int
    n_categories: dict[str, int]
    exp_a: dict[str, Distribution]
    exp_b: dict[str, Distribution]
    exp_c: dict[str, Distribution]
    exp_c_outcomes: dict[str, int]
    detection_rate: dict[str, int]  # detected / no_detection / grounding_error
    wrong_object_count: int
    category_exp_a: dict[str, dict[str, Distribution]]
    category_exp_b: dict[str, dict[str, Distribution]]
    category_exp_c: dict[str, dict[str, Distribution]]
    per_sample: list[PerSampleMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def dist(d: Distribution) -> dict[str, Any]:
            return d.as_dict()

        return {
            "n_samples": self.n_samples,
            "n_categories": self.n_categories,
            "exp_a": {k: dist(v) for k, v in self.exp_a.items()},
            "exp_b": {k: dist(v) for k, v in self.exp_b.items()},
            "exp_c": {k: dist(v) for k, v in self.exp_c.items()},
            "exp_c_outcomes": self.exp_c_outcomes,
            "detection_rate": self.detection_rate,
            "wrong_object_count": self.wrong_object_count,
            "category_exp_a": {
                cat: {k: dist(v) for k, v in metrics.items()}
                for cat, metrics in self.category_exp_a.items()
            },
            "category_exp_b": {
                cat: {k: dist(v) for k, v in metrics.items()}
                for cat, metrics in self.category_exp_b.items()
            },
            "category_exp_c": {
                cat: {k: dist(v) for k, v in metrics.items()}
                for cat, metrics in self.category_exp_c.items()
            },
            "per_sample": [m.__dict__ for m in self.per_sample],
        }


def _dist(values: list[float], failures: int = 0) -> Distribution:
    return compute_distribution(values, failures=failures)


def aggregate(results: list[SampleResult]) -> BenchmarkReport:
    """Aggregate per-sample results into per-experiment and per-category distributions."""
    n = len(results)
    n_categories: dict[str, int] = {}
    for r in results:
        n_categories[r.category] = n_categories.get(r.category, 0) + 1

    def values(results: list[SampleResult], path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for r in results:
            node: Any = getattr(r, path[0])
            for key in path[1:]:
                node = node[key]
            if isinstance(node, (int, float)) and node is not None:
                out.append(float(node))
        return out

    exp_a = {
        metric: _dist(values(results, ("exp_a", metric)))
        for metric in ("iou", "dice", "precision", "recall")
    }
    exp_b = {
        metric: _dist(values(results, ("exp_b", metric)))
        for metric in ("bbox_iou", "gt_coverage", "area_ratio")
    }
    accepted = [r for r in results if r.exp_c.get("outcome") == "accepted"]
    exp_c = {
        metric: _dist(values(accepted, ("exp_c", metric)))
        for metric in ("iou", "dice", "precision", "recall")
    }
    exp_c_outcomes: dict[str, int] = {}
    detection: dict[str, int] = {"detected": 0, "no_detection": 0, "grounding_error": 0}
    for r in results:
        outcome = str(r.exp_c.get("outcome"))
        exp_c_outcomes[outcome] = exp_c_outcomes.get(outcome, 0) + 1
        status = r.exp_b.get("status", "no_detection")
        if status.startswith("grounding_error"):
            detection["grounding_error"] = detection.get("grounding_error", 0) + 1
        else:
            detection[status] = detection.get(status, 0) + 1

    wrong_object = 0
    per_sample: list[PerSampleMetrics] = []
    for r in results:
        a_iou = r.exp_a.get("iou")
        b_iou = r.exp_b.get("bbox_iou")
        if b_iou is not None and b_iou < _WRONG_OBJECT_BBOX_IOU:
            wrong_object += 1
        c_iou = r.exp_c.get("iou") if r.exp_c.get("outcome") == "accepted" else None
        gap = a_iou - c_iou if (a_iou is not None and c_iou is not None) else None
        per_sample.append(
            PerSampleMetrics(
                sample_id=r.sample_id,
                category=r.category,
                a_iou=a_iou,
                a_dice=r.exp_a.get("dice"),
                b_bbox_iou=b_iou,
                b_gt_coverage=r.exp_b.get("gt_coverage"),
                b_area_ratio=r.exp_b.get("area_ratio"),
                c_iou=c_iou,
                c_dice=r.exp_c.get("dice") if r.exp_c.get("outcome") == "accepted" else None,
                c_precision=(
                    r.exp_c.get("precision") if r.exp_c.get("outcome") == "accepted" else None
                ),
                c_recall=r.exp_c.get("recall") if r.exp_c.get("outcome") == "accepted" else None,
                c_outcome=str(r.exp_c.get("outcome")),
                c_failure_detail=r.exp_c.get("failure_detail"),
                gap_a_minus_c=gap,
            )
        )

    def by_category(path: tuple[str, ...], metric: str) -> dict[str, Distribution]:
        out: dict[str, Distribution] = {}
        for cat in sorted(n_categories):
            cat_results = [r for r in results if r.category == cat]
            out[cat] = _dist(values(cat_results, path + (metric,)))
        return out

    # Only the main object category is guaranteed non-empty; forbidden categories are handled
    # by the separate safety track, so category-level here covers whatever categories the
    # manifest actually contains.
    category_exp_a = {
        cat: {
            metric: by_category(("exp_a",), metric)[cat]
            for metric in ("iou", "dice", "precision", "recall")
        }
        for cat in n_categories
    }
    category_exp_b = {
        cat: {
            metric: by_category(("exp_b",), metric)[cat]
            for metric in ("bbox_iou", "gt_coverage", "area_ratio")
        }
        for cat in n_categories
    }
    cat_accepted = {
        cat: [r for r in results if r.category == cat and r.exp_c.get("outcome") == "accepted"]
        for cat in n_categories
    }
    category_exp_c = {
        cat: {
            metric: _dist(values(cat_accepted[cat], ("exp_c", metric)))
            for metric in ("iou", "dice", "precision", "recall")
        }
        for cat in n_categories
    }

    return BenchmarkReport(
        n_samples=n,
        n_categories=n_categories,
        exp_a=exp_a,
        exp_b=exp_b,
        exp_c=exp_c,
        exp_c_outcomes=exp_c_outcomes,
        detection_rate=detection,
        wrong_object_count=wrong_object,
        category_exp_a=category_exp_a,
        category_exp_b=category_exp_b,
        category_exp_c=category_exp_c,
        per_sample=per_sample,
    )


def compute_forbidden_overlap(
    results: list[SampleResult],
    manifest: BenchmarkManifest,
    cache_dir: Path,
    hf_token: str,
) -> dict[str, dict[str, float]]:
    """Safety-track: for each main-object sample, how much of the FINAL production mask covers
    forbidden GT regions (text / balloon / frame / onomatopoeia -- face is deliberately
    excluded because a character body GT mask legitimately contains its face).

    Returns `{sample_id: {forbidden_total, text, balloon, frame, onomatopoeia}}` where each
    value is the fraction of the final mask's pixels inside that forbidden category's GT masks
    on the same page. Only samples with an accepted production mask are included.
    """
    books = sorted({s.book for s in manifest.samples})
    book_annotations: dict[str, dict[str, Any]] = {
        book: ms92_book_annotations(hf_token, book, cache_dir / "ms92") for book in books
    }
    # forbidden GT masks per (book, page_index)
    forbidden_by_page: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for book, data in book_annotations.items():
        by_page: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            by_page.setdefault(ann["image_id"], []).append(ann)
        for page, anns in by_page.items():
            forbidden_by_page[(book, page)] = {
                name: (
                    np.stack(
                        [
                            decode_rle(a["segmentation"])
                            for a in anns
                            if a["category_id"] == CATEGORY_IDS[name]
                        ]
                    ).any(axis=0)
                    if any(a["category_id"] == CATEGORY_IDS[name] for a in anns)
                    else np.zeros(
                        (data["images"][0]["height"], data["images"][0]["width"]), dtype=bool
                    )
                )
                for name in ("text", "balloon", "frame", "onomatopoeia")
            }

    out: dict[str, dict[str, float]] = {}
    for r in results:
        final_path = r.exp_c.get("final_mask_path")
        if not final_path or not Path(final_path).exists():
            continue
        final_mask = np.load(final_path)["mask"] > 0
        area = int(final_mask.sum())
        if area == 0:
            continue
        # look up the page from the manifest
        sample = next(s for s in manifest.samples if s.sample_id == r.sample_id)
        page_masks = forbidden_by_page.get((sample.book, sample.page_index))
        if page_masks is None:
            continue
        per_cat: dict[str, float] = {}
        for name, m in page_masks.items():
            if m.shape[:2] != final_mask.shape[:2]:
                continue
            per_cat[name] = float(np.count_nonzero(final_mask & m) / area)
        per_cat["forbidden_total"] = float(
            np.count_nonzero(
                final_mask
                & np.logical_or.reduce(
                    [page_masks[n] for n in ("text", "balloon", "frame", "onomatopoeia")]
                )
            )
            / area
        )
        out[r.sample_id] = per_cat
    return out


def _rank_failure_cases(results: list[SampleResult]) -> list[str]:
    """Deterministic priority list of sample_ids worth a visual package: worst Exp-A, worst
    Exp-C, biggest A-minus-C gap, gate rejections, and wrong-object selections."""
    scored: list[tuple[float, str, str]] = []
    for r in results:
        a_iou = r.exp_a.get("iou")
        c_iou = r.exp_c.get("iou") if r.exp_c.get("outcome") == "accepted" else None
        gap = (a_iou - c_iou) if (a_iou is not None and c_iou is not None) else None
        scored.append((a_iou if a_iou is not None else -1.0, f"a_worst:{r.sample_id}", r.sample_id))
        if c_iou is not None:
            scored.append((c_iou, f"c_worst:{r.sample_id}", r.sample_id))
        if gap is not None:
            scored.append((gap, f"gap:{r.sample_id}", r.sample_id))
        if r.exp_c.get("outcome") == "segment_gate_rejected":
            scored.append((1.0, f"gate_reject:{r.sample_id}", r.sample_id))
        if r.exp_b.get("bbox_iou") is not None and r.exp_b["bbox_iou"] < _WRONG_OBJECT_BBOX_IOU:
            scored.append((1.0, f"wrong_object:{r.sample_id}", r.sample_id))
    scored.sort(key=lambda t: t[0])
    seen: list[str] = []
    for _, _reason, sid in scored:
        if sid not in seen:
            seen.append(sid)
    return seen


def build_visual_failures(
    results: list[SampleResult],
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    *,
    max_cases: int = 12,
) -> list[Path]:
    """Save one montage PNG per important failure case (brief section 11): original image,
    GT bbox, DINO bbox, GT mask, SAM-on-GT-bbox mask, SAM-on-DINO-bbox mask, and final
    production mask, plus overlays. Returns the written paths. Purely CPU (masks on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {s.sample_id: s for s in manifest.samples}
    priority = _rank_failure_cases(results)[:max_cases]
    written: list[Path] = []
    for sample_id in priority:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        image = np.asarray(Image.open(dataset_dir / f"{sample_id}.png").convert("RGB"))
        gt_mask = np.load(dataset_dir / f"{sample_id}.mask.npz")["mask"] > 0
        result = next(r for r in results if r.sample_id == sample_id)
        panels: dict[str, np.ndarray] = {}
        a_path = result.exp_a.get("mask_path")
        if a_path and Path(a_path).exists():
            panels["SAM(GT bbox)"] = np.load(a_path)["mask"] > 0
        raw_path = result.exp_c.get("raw_mask_path")
        if raw_path and Path(raw_path).exists():
            panels["SAM(DINO bbox)"] = np.load(raw_path)["mask"] > 0
        final_path = result.exp_c.get("final_mask_path")
        if final_path and Path(final_path).exists():
            panels["final production"] = np.load(final_path)["mask"] > 0

        h, w = image.shape[:2]
        scale = min(1.0, 480.0 / max(h, w))
        small = Image.fromarray(image).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
        small_arr = np.asarray(small)
        sh, sw = small_arr.shape[:2]

        def overlay_boxes(
            img: np.ndarray = small_arr,
            sc: float = scale,
            sample_ref=sample,
            result_ref=result,
        ) -> np.ndarray:
            canvas = img.copy()
            import cv2

            def draw(box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
                x0, y0, x1, y1 = (int(v * sc) for v in box)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

            draw(sample_ref.gt_bbox, (0, 255, 0))
            top_bbox = result_ref.exp_b.get("top_bbox")
            if top_bbox:
                x0, y0, x1, y1 = (int(v) for v in top_bbox)
                draw((x0, y0, x1, y1), (255, 0, 0))
            sel = result_ref.exp_c.get("selected_bbox")
            if sel:
                x0, y0, x1, y1 = (int(v) for v in sel)
                draw((x0, y0, x1, y1), (0, 0, 255))
            return canvas

        def mask_panel(
            mask: np.ndarray,
            img: np.ndarray = small_arr,
            width: int = sw,
            height: int = sh,
        ) -> np.ndarray:
            canvas = img.copy()
            resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                (width, height), Image.Resampling.NEAREST
            )
            small_mask = np.asarray(resized) > 0
            canvas[small_mask] = (canvas[small_mask] * 0.6 + np.array([255, 80, 80]) * 0.4).astype(
                np.uint8
            )
            return canvas

        grid = [
            overlay_boxes(),
            mask_panel(gt_mask),
        ]
        for m in panels.values():
            grid.append(mask_panel(m))

        n = len(grid)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        cell_h, cell_w = grid[0].shape[:2]
        canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
        for i, g in enumerate(grid):
            r, c = divmod(i, cols)
            canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = g

        out_path = out_dir / f"failure_{sample_id}.png"
        Image.fromarray(canvas).save(out_path)
        written.append(out_path)
    return written


def write_report(
    report: BenchmarkReport,
    results: list[SampleResult],
    out_dir: Path,
    *,
    title: str = "Phase 17 object segmentation diagnostic",
) -> tuple[Path, Path]:
    """Write `report.json` (machine-readable) and `report.md` (human-readable) into `out_dir`.
    The markdown follows the phase brief's final-report structure and separates observed facts
    from hypotheses."""
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    def fmt(d: Distribution) -> str:
        if d.count == 0:
            return "n/a"
        return (
            f"n={d.count} mean={d.mean:.3f} med={d.median:.3f} "
            f"p25={d.p25:.3f} p75={d.p75:.3f} p05={d.p05:.3f} p95={d.p95:.3f} "
            f"min={d.minimum:.3f} max={d.maximum:.3f} std={d.std:.3f} "
            f"(failures={d.failures})"
        )

    lines: list[str] = [
        f"# {title}",
        "",
        f"- samples: {report.n_samples}",
        f"- categories: {report.n_categories}",
        f"- production outcomes (Exp C): {report.exp_c_outcomes}",
        f"- Exp B detection: {report.detection_rate}",
        f"- wrong-object selections "
        f"(Exp B, bbox IoU < {_WRONG_OBJECT_BBOX_IOU}): {report.wrong_object_count}",
        "",
        "## Experiment A — pure SAM (GT bbox -> SAM -> mask)",
        "",
        "| metric | distribution |",
        "|---|---|",
    ]
    for metric, d in report.exp_a.items():
        lines.append(f"| {metric} | {fmt(d)} |")
    lines += [
        "",
        "## Experiment B — Grounding DINO localization (DINO bbox vs GT bbox)",
        "",
        "| metric | distribution |",
        "|---|---|",
    ]
    for metric, d in report.exp_b.items():
        lines.append(f"| {metric} | {fmt(d)} |")
    lines += [
        "",
        "## Experiment C — real production DINO -> selection -> SAM -> post-processing",
        "",
        "| metric | distribution (accepted samples only) |",
        "|---|---|",
    ]
    for metric, d in report.exp_c.items():
        lines.append(f"| {metric} | {fmt(d)} |")
    lines += [
        "",
        "## Category-level (main object category)",
        "",
    ]
    for cat in sorted(report.n_categories):
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("- Exp A IoU: " + fmt(report.category_exp_a[cat]["iou"]))
        lines.append("- Exp B bbox IoU: " + fmt(report.category_exp_b[cat]["bbox_iou"]))
        lines.append("- Exp C IoU: " + fmt(report.category_exp_c[cat]["iou"]))
        lines.append("")
    lines += [
        "",
        "## Per-sample detail (machine-readable in report.json)",
        "",
        "```json",
        json.dumps([m.__dict__ for m in report.per_sample], indent=1),
        "```",
        "",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
