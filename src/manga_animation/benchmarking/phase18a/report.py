"""Phase 18.2A report: aggregation + report.json/report.md.

CPU-side, runs after the GPU collections. Produces the machine-readable `report.json` and the
human-readable `report.md` under the experiment dir, plus the per-target detail file. The
markdown follows the phase brief's required OBSERVED / INTERPRETATION / HYPOTHESES /
LIMITATIONS / ARCHITECTURAL RECOMMENDATION structure and never presents hypotheses as facts.

The DINO reference numbers in the comparison table are frozen constants from Phase 18.1's
measured results (docs/phase18.1-results.md) -- this phase does not re-run DINO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from manga_animation.benchmarking.phase17.metrics import Distribution
from manga_animation.benchmarking.phase18a.classify import ERROR_CATEGORY_NAMES
from manga_animation.benchmarking.phase18a.metrics import (
    PerTargetMetrics,
    Phase18aMetrics,
    compute_metrics,
)

# Phase 18.1 measured DINO candidate-recall numbers (frozen references, not re-measured here).
DINO_TOP1_RECALL = 0.062
DINO_RALL_RECALL = 0.891
# Phase 17 measured GT-bbox->SAM median mask IoU (docs/phase17-results.md Experiment A).
PHASE17_GT_SAM_MEDIAN_IOU = 0.884

_ERROR_CATEGORY_ORDER = (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)


@dataclass
class Phase18aReport:
    n_targets: int
    metrics: Phase18aMetrics
    category_counts: dict[str, int]
    per_target: list[PerTargetMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_targets": self.n_targets,
            "metrics": self.metrics.as_dict(),
            "category_counts": self.category_counts,
            "per_target": [t.as_dict() for t in self.per_target],
        }


def build_report(targets: list[PerTargetMetrics]) -> Phase18aReport:
    metrics = compute_metrics(targets)
    counts: dict[str, int] = {}
    for t in targets:
        counts[t.error_category or "unclassified"] = (
            counts.get(t.error_category or "unclassified", 0) + 1
        )
    return Phase18aReport(
        n_targets=len(targets),
        metrics=metrics,
        category_counts=counts,
        per_target=targets,
    )


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def _dist_line(label: str, d: Distribution) -> str:
    if d.count == 0:
        return f"- {label}: n=0 (none)"
    return (
        f"- {label}: n={d.count} mean={d.mean:.3f} median={d.median:.3f} "
        f"p25={d.p25:.3f} p75={d.p75:.3f} min={d.minimum:.3f} max={d.maximum:.3f}"
    )


def write_report(report: Phase18aReport, out_dir: Path) -> tuple[Path, Path]:
    """Write `report.json`, `per_target.json`, and `report.md` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    (out_dir / "per_target.json").write_text(
        json.dumps([t.as_dict() for t in report.per_target], indent=1), encoding="utf-8"
    )

    m = report.metrics
    lines: list[str] = [
        "# Phase 18.2A — Qwen2.5-VL Direct Target Localization Benchmark",
        "",
        f"- targets: {report.n_targets} (same 64 human-annotated `body` instances as "
        "Phase 17/18.1)",
        f"- found (usable bbox): {m.n_found}/{report.n_targets} ({_percent(m.found_rate)})",
        f"- coordinate-conversion failures: {m.n_conversion_failures}",
        "",
        "## Qwen direct bbox vs GT bbox",
        "",
        _dist_line("IoU (found samples)", m.bbox_iou_found),
        _dist_line("IoU (all 64, not-found = 0)", m.bbox_iou_all),
        _dist_line("GT coverage (found)", m.gt_coverage_found),
        _dist_line("area ratio (found)", m.area_ratio_found),
        "",
        "| IoU threshold | recall (all 64) | recall (found) |",
        "|---|---|---|",
    ]
    for t in (0.25, 0.50, 0.75):
        lines.append(
            f"| {t:.2f} | {_percent(m.recall_all[t])} | {_percent(m.recall_found[t])} |"
        )
    lines += [
        "",
        "**Primary metric: Recall@IoU>=0.5 = "
        f"{_percent(m.recall_all[0.5])}** (all targets).",
        "",
        "## Error categories (min, phase-brief taxonomy)",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for cat in _ERROR_CATEGORY_ORDER:
        name = {
            9: "9 coordinate conversion failure",
            8: "8 VLM not found",
            7: "7 target outside panel / page grab",
            6: "6 multiple similar objects (visual review)",
            5: "5 partially hidden object (visual review)",
            4: "4 bbox too small",
            3: "3 bbox too large",
            2: "2 wrong instance",
            1: "1 correct object, imprecise bbox",
            0: "0 good (IoU >= 0.75)",
        }[cat]
        count = report.category_counts.get(ERROR_CATEGORY_NAMES[cat], 0)
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "## Downstream: Qwen bbox -> SAM -> mask vs GT mask",
        "",
        _dist_line("Qwen bbox -> SAM mask IoU (found samples)", m.qs_mask_iou_found),
        _dist_line("Qwen bbox -> SAM mask IoU (all 64, not-found = 0)", m.qs_mask_iou_all),
        "",
        "## Reference: GT bbox -> SAM -> mask vs GT mask",
        "",
        _dist_line("GT bbox -> SAM mask IoU", m.gs_mask_iou),
        f"(Phase 17 reference median: {PHASE17_GT_SAM_MEDIAN_IOU:.3f})",
        "",
        "## Comparison with the existing pipeline",
        "",
        "| signal | value |",
        "|---|---|",
        f"| DINO top-1 bbox recall (Phase 18.1) | {_percent(DINO_TOP1_RECALL)} |",
        f"| DINO candidate availability R@All (Phase 18.1) | {_percent(DINO_RALL_RECALL)} |",
        f"| Qwen direct bbox recall@0.5 | {_percent(m.recall_all[0.5])} |",
        f"| Qwen bbox -> SAM median mask IoU | {_fmt(m.qs_mask_iou_found.median)} |",
        f"| GT bbox -> SAM median mask IoU | {_fmt(m.gs_mask_iou.median)} |",
        "",
        "Full analysis, interpretation, hypotheses, limitations, and the architectural "
        "recommendation: docs/phase18.2a-qwen-bbox-results.md.",
        "",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
