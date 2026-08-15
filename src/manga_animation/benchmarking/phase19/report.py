"""Phase 19 report: aggregate the controlled benchmark, assign the failure taxonomy, compute
safety overlap, and write `report.json`/`report.md`.

Everything here is CPU-side and runs after the GPU pass (masks and per-sample records are on
disk). Distributions carry median/percentiles and a failure count (phase-17 convention), and
the failure taxonomy is assigned deterministically from measured signals (`failure_taxonomy`).
The forbidden-overlap safety track reuses the phase-17 MS92 GT (`text`/`balloon`/`frame`/
`onomatopoeia`) and needs an HF token only for that optional step.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from manga_animation.benchmarking.phase17.dataset import (
    CATEGORY_IDS,
    _page_from_file_name,
    decode_rle,
    ms92_book_annotations,
)
from manga_animation.benchmarking.phase17.metrics import (
    Distribution,
    MaskMetrics,
)
from manga_animation.benchmarking.phase19.failure_taxonomy import (
    CATEGORY_LABELS,
    CONTAMINATION_FRACTION,
    SampleSignal,
    classify,
)
from manga_animation.benchmarking.phase19.metrics import (
    AggregateMetrics,
    TargetMetrics,
    aggregate_metrics,
)
from manga_animation.benchmarking.phase19.run import ControlledSampleRecord

# Forbidden GT categories fed into the safety overlap (face is excluded: a character body
# legitimately contains its face, phase-17 convention).
_FORBIDDEN = ("text", "balloon", "frame", "onomatopoeia")

_ZERO = MaskMetrics(iou=0.0, dice=0.0, precision=0.0, recall=0.0)


def metrics_list(records: list[ControlledSampleRecord]) -> list[TargetMetrics]:
    """Every record's metrics, with zero-overlap for inference errors / no-mask outcomes so the
    aggregate never drops a failure."""
    out: list[TargetMetrics] = []
    for r in records:
        if r.metrics is not None:
            out.append(
                TargetMetrics(
                    sample_id=r.sample_id,
                    metrics=MaskMetrics(
                        iou=float(r.metrics["iou"]),
                        dice=float(r.metrics["dice"]),
                        precision=float(r.metrics["precision"]),
                        recall=float(r.metrics["recall"]),
                    ),
                    instance_correct=bool(r.metrics["instance_correct"]),
                    end_to_end_success=bool(r.metrics["end_to_end_success"]),
                    recall_hits={float(t): bool(v) for t, v in r.metrics["recall_hits"].items()},
                )
            )
        else:
            out.append(
                TargetMetrics(
                    sample_id=r.sample_id,
                    metrics=_ZERO,
                    instance_correct=False,
                    end_to_end_success=False,
                    recall_hits={0.25: False, 0.50: False, 0.75: False},
                )
            )
    return out


def assign_failure_categories(
    records: list[ControlledSampleRecord],
) -> list[ControlledSampleRecord]:
    """Assign each record its primary failure-taxonomy category from its measured signals."""
    for r in records:
        r.failure_category = classify(
            SampleSignal(
                status="inference_error" if r.status == "inference_error" else "ok",
                n_masks=r.n_masks,
                coord_ok=(r.metrics is not None),
                instance_correct=bool(r.metrics["instance_correct"]) if r.metrics else False,
                iou=float(r.metrics["iou"]) if r.metrics else 0.0,
                no_target_text=r.target_not_found_text,
                forbidden=r.forbidden or {},
                multi_instance=r.multi_instance,
                manual_category=r.manual_category,
            )
        )
    return records


def compute_forbidden_overlap(
    records: list[ControlledSampleRecord],
    manifest,
    cache_dir: Path,
    hf_token: str,
) -> dict[str, dict[str, float]]:
    """Fraction of each predicted mask inside the page's GT forbidden masks (text / balloon /
    frame / onomatopoeia). Reuses the phase-17 MS92 annotation download/decode path. Only
    records with a saved mask are included."""
    books = sorted({s.book for s in manifest.samples})
    book_annotations: dict[str, dict[str, Any]] = {
        book: ms92_book_annotations(hf_token, book, cache_dir / "ms92") for book in books
    }
    wanted_pages = {(s.book, s.page_index) for s in manifest.samples}
    forbidden_by_page: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for book, data in book_annotations.items():
        by_page: dict[int, list[dict[str, Any]]] = {}
        img_by_id = {img["id"]: img for img in data["images"]}
        page_size: dict[int, tuple[int, int]] = {}
        for ann in data["annotations"]:
            img = img_by_id.get(ann["image_id"])
            if img is None:
                continue
            page = _page_from_file_name(img["file_name"])
            page_size.setdefault(page, (img["height"], img["width"]))
            if (book, page) not in wanted_pages:
                continue
            by_page.setdefault(page, []).append(ann)
        for page, anns in by_page.items():
            h, w = page_size[page]
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
                    else np.zeros((h, w), dtype=bool)
                )
                for name in _FORBIDDEN
            }

    sample_by_id = {s.sample_id: s for s in manifest.samples}
    out: dict[str, dict[str, float]] = {}
    for r in records:
        if not r.pred_mask_path or not Path(r.pred_mask_path).exists():
            continue
        mask = np.load(r.pred_mask_path)["mask"] > 0
        area = int(mask.sum())
        if area == 0:
            continue
        sample = sample_by_id.get(r.sample_id)
        if sample is None:
            continue
        page_masks = forbidden_by_page.get((sample.book, sample.page_index))
        if page_masks is None:
            continue
        per_cat: dict[str, float] = {}
        for name, m in page_masks.items():
            if m.shape != mask.shape:
                continue
            per_cat[name] = float(np.count_nonzero(mask & m) / area)
        out[r.sample_id] = per_cat
    return out


def apply_forbidden(
    records: list[ControlledSampleRecord], forbidden: dict[str, dict[str, float]]
) -> list[ControlledSampleRecord]:
    for r in records:
        if r.sample_id in forbidden:
            r.forbidden = forbidden[r.sample_id]
    return records


@dataclass
class Phase19Report:
    condition: str
    provenance: str
    n_targets: int
    metrics: AggregateMetrics
    failure_counts: dict[str, int]
    latency_median: float | None
    latency_p95: float | None
    vram_peak_median_mb: float | None
    contaminated_count: int  # forbidden_total >= CONTAMINATION_FRACTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "provenance": self.provenance,
            "n_targets": self.n_targets,
            "metrics": self.metrics.as_dict(),
            "failure_counts": self.failure_counts,
            "latency_median_seconds": self.latency_median,
            "latency_p95_seconds": self.latency_p95,
            "vram_peak_median_mb": self.vram_peak_median_mb,
            "contaminated_count": self.contaminated_count,
            "failure_labels": {k: CATEGORY_LABELS[k] for k in sorted(self.failure_counts)},
        }


def build_report(
    records: list[ControlledSampleRecord], *, condition: str, provenance: str
) -> Phase19Report:
    records = assign_failure_categories(records)
    aggr = aggregate_metrics(metrics_list(records))
    lats = [r.latency_seconds for r in records if r.status == "ok" and r.latency_seconds > 0]
    vrams = [v for v in (r.vram_peak_mb for r in records) if v is not None]
    counts: dict[str, int] = {}
    for r in records:
        if r.failure_category is not None:
            counts[r.failure_category] = counts.get(r.failure_category, 0) + 1
    contaminated = sum(
        1 for r in records if r.forbidden and sum(r.forbidden.values()) >= CONTAMINATION_FRACTION
    )
    return Phase19Report(
        condition=condition,
        provenance=provenance,
        n_targets=len(records),
        metrics=aggr,
        failure_counts=counts,
        latency_median=statistics.median(lats) if lats else None,
        latency_p95=(
            sorted(lats)[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else None
        ),
        vram_peak_median_mb=statistics.median(vrams) if vrams else None,
        contaminated_count=contaminated,
    )


def write_report(
    report: Phase19Report, records: list[ControlledSampleRecord], out_dir: Path
) -> tuple[Path, Path]:
    """Write `report.json` and `report.md` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    def fmt(d: Distribution) -> str:
        if d.count == 0:
            return "n/a"
        return (
            f"n={d.count} mean={d.mean:.3f} med={d.median:.3f} p25={d.p25:.3f} "
            f"p75={d.p75:.3f} p05={d.p05:.3f} p95={d.p95:.3f} min={d.minimum:.3f} "
            f"max={d.maximum:.3f} std={d.std:.3f} (failures={d.failures})"
        )

    lines = [
        "# Phase 19 controlled benchmark — OMG-LLaVA referring segmentation",
        "",
        f"- condition: **{report.condition}** (provenance: {report.provenance})",
        f"- targets: {report.n_targets}",
        f"- END_TO_END_SUCCESS (correct instance AND IoU >= 0.50): "
        f"{report.metrics.end_to_end_success_rate:.3f}",
        f"- instance-correct rate: {report.metrics.instance_correct_rate:.3f}",
        "",
        "## Mask quality",
        "",
        "| metric | distribution |",
        "|---|---|",
        f"| iou | {fmt(report.metrics.iou)} |",
        f"| dice | {fmt(report.metrics.dice)} |",
        "",
        "## Recall",
        "",
        "| IoU threshold | recall |",
        "|---|---|",
    ]
    for t, v in report.metrics.recall_at.items():
        lines.append(f"| >= {t:.2f} | {v:.3f} |")
    lines += [
        "",
        "## Failure taxonomy",
        "",
        "| category | label | count |",
        "|---|---|---|",
    ]
    for cat in sorted(report.failure_counts):
        lines.append(f"| {cat} | {CATEGORY_LABELS.get(cat, cat)} | {report.failure_counts[cat]} |")
    lines += [
        "",
        "## Performance",
        "",
        f"- latency median: {report.latency_median:.2f}s" if report.latency_median is not None
        else "- latency: n/a",
        f"- latency P95: {report.latency_p95:.2f}s" if report.latency_p95 is not None else "",
        f"- VRAM peak median: {report.vram_peak_median_mb:.0f} MB"
        if report.vram_peak_median_mb is not None else "- VRAM: n/a",
        f"- contaminated masks (forbidden_total >= {CONTAMINATION_FRACTION}): "
        f"{report.contaminated_count}",
        "",
        "## Per-sample detail (machine-readable in report.json)",
        "",
        "```json",
        json.dumps(
            [
                {
                    "sample_id": r.sample_id,
                    "status": r.status,
                    "n_masks": r.n_masks,
                    "iou": r.metrics["iou"] if r.metrics else None,
                    "instance_correct": r.metrics["instance_correct"] if r.metrics else None,
                    "failure": r.failure_category,
                    "target_context": r.target_context,
                    "forbidden": r.forbidden,
                }
                for r in records
            ],
            indent=1,
        ),
        "```",
        "",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
