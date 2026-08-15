"""Phase 18.1 report: aggregate Recall@K curves, category split, and the markdown report.

Everything here is CPU-side, runs after the DINO detection collection, and writes both the
machine-readable `report.json` and the human-readable `docs/phase18.1-results.md` (via the CLI
script). The report answers the phase question -- how often the correct target is present among
all DINO candidates and at what rank -- and explicitly separates OBSERVED FACTS / INTERPRETATION
/ HYPOTHESES / NEXT RECOMMENDATION (the brief forbids presenting hypotheses as facts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from manga_animation.benchmarking.phase18.candidates import (
    RECALL_K_VALUES,
    RECALL_THRESHOLDS,
    RecallCurve,
    TargetRecall,
    recall_curves,
)


@dataclass
class Phase18Report:
    n_targets: int
    n_pages: int
    curves: dict[float, RecallCurve]
    targets: list[TargetRecall] = field(default_factory=list)
    # Per-threshold counts of targets whose correct candidate is below top-1 but exists.
    candidate_below_top1: dict[float, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_targets": self.n_targets,
            "n_pages": self.n_pages,
            "curves": {str(t): c.as_dict() for t, c in sorted(self.curves.items())},
            "candidate_below_top1": self.candidate_below_top1,
            "per_target": [t.as_dict() for t in self.targets],
        }


def build_report(targets: list[TargetRecall]) -> Phase18Report:
    curves = recall_curves(targets)
    n_pages = len({t.page_key for t in targets})
    candidate_below_top1: dict[float, int] = {}
    for t in RECALL_THRESHOLDS:
        if t in curves:
            candidate_below_top1[t] = curves[t].category_counts.get("B", 0)
    return Phase18Report(
        n_targets=len(targets),
        n_pages=n_pages,
        curves=curves,
        targets=targets,
        candidate_below_top1=candidate_below_top1,
    )


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def write_report(report: Phase18Report, out_dir: Path) -> tuple[Path, Path]:
    """Write `report.json` (machine-readable) and `report.md` (human-readable) into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Phase 18.1 — DINO Candidate Recall Benchmark",
        "",
        f"- targets: {report.n_targets} (phase-17 human-annotated `body` instances)",
        f"- unique pages: {report.n_pages}",
        "",
        "## Recall@K (fraction of targets with a correct candidate at rank <= K)",
        "",
        "| IoU threshold | R@1 | R@3 | R@5 | R@10 | R@20 | R@All | A | B | C |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in sorted(report.curves):
        c = report.curves[t]
        r = c.recall_at_k
        def cell(k):
            return _fmt(r.get(k))
        lines.append(
            f"| {t:.2f} | {cell(1)} | {cell(3)} | {cell(5)} | {cell(10)} | {cell(20)} | "
            f"{cell(None)} | {c.category_counts['A']} | {c.category_counts['B']} | "
            f"{c.category_counts['C']} |"
        )
    lines += [
        "",
        "Category A = correct candidate exists AND is top-1; B = exists but below top-1; "
        "C = no candidate at this IoU threshold.",
        "",
        "## Answer",
        "",
        "OBSERVED FACTS: (filled in docs/phase18.1-results.md after inspection)",
        "INTERPRETATION / HYPOTHESES / NEXT RECOMMENDATION: docs/phase18.1-results.md.",
        "",
        "## Per-target detail (machine-readable in report.json)",
        "",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
