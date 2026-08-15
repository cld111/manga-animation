"""Phase 18.2 report: per-strategy selection accuracy, category split, cost, and markdown.

CPU-side aggregation over the per-target reranking results. Compares each VLM strategy against
the Phase 18.1 baseline (DINO top-1, R@1 = 6.2%) and the candidate-availability upper bound
(R@All = 89.1%). Output: `report.json` (machine-readable) + `report.md` (human-readable); the
full OBSERVED/INTERPRETATION/HYPOTHESES/LIMITATIONS/NEXT report is `docs/phase18.2-results.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STRATEGIES = ("A", "B", "C")

BASELINE_DINO_R1 = 0.062  # Phase 18.1: DINO top-1 selection accuracy
CANDIDATE_UPPER_BOUND = 0.891  # Phase 18.1: R@All at IoU >= 0.5 (57/64)


@dataclass
class StrategyResult:
    strategy: str
    n_targets: int
    n_eligible: int  # targets with a correct candidate in the pool (category != C)
    sel_acc_all: float  # selection accuracy @1 over ALL targets
    sel_acc_eligible: float  # over eligible targets only
    recall_at_k: dict[int, float]  # R@K of the correct candidate in the VLM-ranked order
    # among eligible targets
    n_selected_wrong: int  # eligible targets where the VLM picked a wrong candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_targets": self.n_targets,
            "n_eligible": self.n_eligible,
            "sel_acc_all": self.sel_acc_all,
            "sel_acc_eligible": self.sel_acc_eligible,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "n_selected_wrong": self.n_selected_wrong,
        }


@dataclass
class Phase18Report:
    baseline_dino_r1: float
    candidate_upper_bound: float
    n_targets: int
    n_category_c: int
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    per_target: list[dict[str, Any]] = field(default_factory=list)
    perf: dict[str, Any] = field(default_factory=dict)
    error_classes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_dino_r1": self.baseline_dino_r1,
            "candidate_upper_bound": self.candidate_upper_bound,
            "n_targets": self.n_targets,
            "n_category_c": self.n_category_c,
            "strategies": {s: r.as_dict() for s, r in self.strategies.items()},
            "per_target": self.per_target,
            "perf": self.perf,
            "error_classes": self.error_classes,
        }


def classify_error(entry: dict[str, Any], strategy: str) -> str:
    """Provisional, data-driven error class for a target whose VLM selection was wrong
    (evaluation-only; the visual packages are the authoritative adjudication)."""
    sel = entry["strategies"][strategy]
    if entry["best_available_iou"] < 0.5:
        return "8_candidate_absent"
    if sel["selected_correct"]:
        return "correct"
    if sel["selected_matches"] is None:
        return "7_vlm_unparseable"
    if sel["selected_matches"] is False:
        return "6_vlm_rejects_all"
    # VLM said "yes" to the wrong candidate.
    return "1_semantic_confusion_multiple_characters"


def build_report(
    per_target: list[dict[str, Any]], perf: dict[str, Any]
) -> Phase18Report:
    n = len(per_target)
    n_category_c = sum(1 for e in per_target if e["best_available_iou"] < 0.5)
    strategies: dict[str, StrategyResult] = {}
    for strategy in STRATEGIES:
        eligible = [e for e in per_target if e["best_available_iou"] >= 0.5]
        sel_all = sum(1 for e in per_target if e["strategies"][strategy]["selected_correct"])
        sel_elig = sum(1 for e in eligible if e["strategies"][strategy]["selected_correct"])
        recall_at_k: dict[int, float] = {}
        for k in (1, 3, 5, 10):
            hits = sum(
                1
                for e in eligible
                if e["strategies"][strategy]["best_correct_rank"] is not None
                and e["strategies"][strategy]["best_correct_rank"] <= k
            )
            recall_at_k[k] = hits / len(eligible) if eligible else 0.0
        n_selected_wrong = sum(
            1
            for e in eligible
            if not e["strategies"][strategy]["selected_correct"]
        )
        strategies[strategy] = StrategyResult(
            strategy=strategy,
            n_targets=n,
            n_eligible=len(eligible),
            sel_acc_all=sel_all / n if n else 0.0,
            sel_acc_eligible=sel_elig / len(eligible) if eligible else 0.0,
            recall_at_k=recall_at_k,
            n_selected_wrong=n_selected_wrong,
        )
    # Error classes for strategy A over all targets.
    error_classes: dict[str, int] = {}
    for e in per_target:
        cls = classify_error(e, "A")
        error_classes[cls] = error_classes.get(cls, 0) + 1
    return Phase18Report(
        baseline_dino_r1=BASELINE_DINO_R1,
        candidate_upper_bound=CANDIDATE_UPPER_BOUND,
        n_targets=n,
        n_category_c=n_category_c,
        strategies=strategies,
        per_target=per_target,
        perf=perf,
        error_classes=error_classes,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(report: Phase18Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    lines = [
        "# Phase 18.2 — VLM-Guided Candidate Reranking",
        "",
        f"- targets: {report.n_targets} (64 phase-17/18.1 targets)",
        f"- category C (no candidate >= 0.5 IoU): {report.n_category_c}",
        f"- baseline DINO top-1: {_pct(report.baseline_dino_r1)}",
        f"- candidate upper bound (R@All, phase 18.1): {_pct(report.candidate_upper_bound)}",
        "",
        "## Selection Accuracy @1 (selected candidate IoU >= 0.5 with GT)",
        "",
        "| strategy | sel@1 all | sel@1 eligible | n_eligible | n_selected_wrong |",
        "|---|---|---|---|---|",
    ]
    for s in ("A", "B", "C"):
        r = report.strategies[s]
        lines.append(
            f"| {s} | {_pct(r.sel_acc_all)} | {_pct(r.sel_acc_eligible)} | "
            f"{r.n_eligible} | {r.n_selected_wrong} |"
        )
    lines += [
        "",
        "## Recall@K of the correct candidate in the VLM-ranked order (eligible targets)",
        "",
        "| strategy | R@1 | R@3 | R@5 | R@10 |",
        "|---|---|---|---|---|",
    ]
    for s in ("A", "B", "C"):
        r = report.strategies[s]
        lines.append(
            f"| {s} | {_pct(r.recall_at_k[1])} | {_pct(r.recall_at_k[3])} | "
            f"{_pct(r.recall_at_k[5])} | {_pct(r.recall_at_k[10])} |"
        )
    lines += [
        "",
        "## Cost (measured)",
        "",
        f"- VLM calls: {report.perf.get('vlm_calls', 'n/a')} "
        f"(cached: {report.perf.get('cached_calls', 'n/a')})",
        f"- total elapsed: {report.perf.get('total_elapsed_s', 'n/a')} s",
        "",
        "## Error classes (strategy A, provisional -- visual packages are authoritative)",
        "",
        "| class | count |",
        "|---|---|",
    ]
    for cls, count in sorted(report.error_classes.items()):
        lines.append(f"| {cls} | {count} |")
    lines += [
        "",
        "Full OBSERVED/INTERPRETATION/HYPOTHESES/LIMITATIONS/NEXT: docs/phase18.2-results.md",
        "",
    ]
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
