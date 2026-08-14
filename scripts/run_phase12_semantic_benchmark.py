"""Phase 12 semantic-mask validation leaderboard: evaluate candidate methods against the real

benchmark (`configs/phase12_semantic_mask_benchmark.yaml`, see
`src/manga_animation/evaluation/mask_dataset.py` for the schema and
`docs/decisions/0018-semantic-mask-validation.md` for the design this benchmarks). Workstream
30 ("mask quality leaderboard") + Workstream 48 ("lightweight large-evaluation mode") -- a new
candidate method is addable without rewriting this evaluator (see `_METHODS` below).

Two families of candidate method:

1. **Geometric-only baselines** (`_geometric_signal_method`) -- deterministic, no model call,
   reusing Phase 11's own four signals (fragmentation/density/aspect-ratio/solidity). Per
   signal, the BEST possible single threshold on this benchmark's own 13 real samples is found
   by exhaustive sweep (`_best_threshold`) -- giving the geometric approach its fairest possible
   shot, not a strawman -- and reported as `geometric:<signal_name>`. This formalizes, with
   fresh numbers recomputed directly from the real local `.npy` arrays (not copied from prose),
   Phase 11's own negative finding (docs/phase11-results.md section 7).

2. **VLM mask-crop verification** (`_vlm_method`, `validation.mask_semantics.verify_mask_semantics`)
   -- the real Phase 12 candidate. Needs a live `VLMClient`; `--vlm real` loads the real Qwen
   client (GPU-worker only, per ADR 0003 -- never run locally) and `--vlm none` (the default)
   skips this method entirely with a clear note, so this script stays runnable locally for the
   geometric baselines without requiring GPU access.

Usage:
    uv run python scripts/run_phase12_semantic_benchmark.py             # geometric only
    uv run python scripts/run_phase12_semantic_benchmark.py --vlm real  # + real VLM (GPU worker)

Writes `outputs/experiments/phase12_semantic_benchmark_<timestamp>.json` and prints a summary
table. Samples whose real local artifacts are missing (git-ignored, ADR 0002) are SKIPPED, not
counted as failures -- the report states exactly how many of the 13 benchmark entries were
actually evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from manga_animation.evaluation.harness import environment_metadata, git_commit
from manga_animation.evaluation.mask_dataset import (
    DEFAULT_MASK_BENCHMARK_PATH,
    MaskSemanticSample,
    load_mask_semantic_benchmark,
)
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PivotSpec,
    TransformKind,
    Vector2,
)
from manga_animation.validation.mask_semantics import verify_mask_semantics

_GEOMETRIC_SIGNALS = (
    "second_component_area_fraction",
    "bbox_density",
    "aspect_ratio",
    "convex_hull_solidity",
)


class _VLMClient(Protocol):
    def generate(self, image, prompt: str) -> str: ...

    def unload(self) -> None: ...


@dataclass
class MethodPrediction:
    sample_id: str
    ground_truth: Literal["good", "bad"]
    difficulty: Literal["typical", "difficult"]
    predicted: Literal["good", "bad", "abstain"]
    score: float | None
    reason: str


@dataclass
class MethodReport:
    method: str
    n_evaluated: int
    n_bad: int
    n_good: int
    true_positive: int  # predicted bad, actually bad
    false_positive: int  # predicted bad, actually good (over-rejection)
    true_negative: int  # predicted good, actually good
    false_negative: int  # predicted good, actually bad (unsafe: a bad mask reaches the renderer)
    abstain: int
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    rejection_rate: float
    abstain_rate: float
    predictions: list[MethodPrediction]


def _object_plan(sample: MaskSemanticSample) -> ObjectPlan:
    transform_kind = TransformKind(sample.transform_kind)
    # verify_mask_semantics never reads ObjectPlan.motion (unlike validate.py's prompt, which
    # threads transform_kind into its own prompt as motion context) -- this only needs to
    # satisfy ObjectPlan's own schema validity (TRANSLATE requires a direction vector).
    direction = Vector2(x=1.0, y=0.0) if transform_kind == TransformKind.TRANSLATE else None
    return ObjectPlan(
        object_id=f"obj_{sample.sample_id}",
        panel_id="panel_1",
        semantic_label=sample.semantic_label,
        confidence=0.8,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=transform_kind,
            direction=direction,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    )


def _best_threshold(
    values: list[float], labels: list[Literal["good", "bad"]]
) -> tuple[float, str, float]:
    """Exhaustive sweep over every candidate cut point between sorted distinct values, both

    directions (`"bad if above"` / `"bad if below"`) -- returns `(threshold, direction,
    accuracy)` for whichever combination classifies this benchmark's own 13 samples best. This
    is deliberately the BEST case for a geometric threshold, not an arbitrary guess, so a
    negative result here is not dismissible as "the threshold was picked badly."
    """
    candidates = sorted(set(values))
    cut_points = [(candidates[i] + candidates[i + 1]) / 2 for i in range(len(candidates) - 1)]
    if not cut_points:
        cut_points = [candidates[0]]
    best = (cut_points[0], "above", -1.0)
    for cut in cut_points:
        for direction in ("above", "below"):
            correct = 0
            for v, label in zip(values, labels, strict=True):
                predicted_bad = (v > cut) if direction == "above" else (v < cut)
                predicted = "bad" if predicted_bad else "good"
                if predicted == label:
                    correct += 1
            accuracy = correct / len(values)
            if accuracy > best[2]:
                best = (cut, direction, accuracy)
    return best


def _geometric_signal_method(
    signal_name: str,
    samples: list[MaskSemanticSample],
    signals_by_sample: dict[str, dict[str, float]],
) -> MethodReport:
    values = [signals_by_sample[s.sample_id][signal_name] for s in samples]
    labels: list[Literal["good", "bad"]] = [s.ground_truth for s in samples]
    threshold, direction, _ = _best_threshold(values, labels)

    predictions = []
    for sample, value in zip(samples, values, strict=True):
        predicted_bad = (value > threshold) if direction == "above" else (value < threshold)
        predicted: Literal["good", "bad"] = "bad" if predicted_bad else "good"
        predictions.append(
            MethodPrediction(
                sample_id=sample.sample_id,
                ground_truth=sample.ground_truth,
                difficulty=sample.difficulty,
                predicted=predicted,
                score=value,
                reason=(
                    f"{signal_name}={value:.4f}, best-fit threshold {direction} {threshold:.4f}"
                ),
            )
        )
    return _summarize(f"geometric:{signal_name}", predictions)


def _vlm_method(samples: list[MaskSemanticSample], vlm_client: _VLMClient) -> MethodReport:
    predictions = []
    for sample in samples:
        image = sample.load_image()
        mask = sample.load_mask()
        result = verify_mask_semantics(image, _object_plan(sample), mask, sample.bbox, vlm_client)
        predicted: Literal["good", "bad", "abstain"]
        if result.verdict == "accept":
            predicted = "good"
        elif result.verdict == "reject":
            predicted = "bad"
        else:
            predicted = "abstain"
        predictions.append(
            MethodPrediction(
                sample_id=sample.sample_id,
                ground_truth=sample.ground_truth,
                difficulty=sample.difficulty,
                predicted=predicted,
                score=result.vlm_confidence,
                reason=result.reason,
            )
        )
    return _summarize("vlm_mask_crop_v1", predictions)


def _summarize(method: str, predictions: list[MethodPrediction]) -> MethodReport:
    tp = sum(1 for p in predictions if p.predicted == "bad" and p.ground_truth == "bad")
    fp = sum(1 for p in predictions if p.predicted == "bad" and p.ground_truth == "good")
    tn = sum(1 for p in predictions if p.predicted == "good" and p.ground_truth == "good")
    fn = sum(1 for p in predictions if p.predicted == "good" and p.ground_truth == "bad")
    abstain = sum(1 for p in predictions if p.predicted == "abstain")
    n = len(predictions)
    n_bad = sum(1 for p in predictions if p.ground_truth == "bad")
    n_good = n - n_bad
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / n_good if n_good > 0 else None
    fnr = fn / n_bad if n_bad > 0 else None
    return MethodReport(
        method=method,
        n_evaluated=n,
        n_bad=n_bad,
        n_good=n_good,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        abstain=abstain,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        rejection_rate=(tp + fp) / n if n > 0 else 0.0,
        abstain_rate=abstain / n if n > 0 else 0.0,
        predictions=predictions,
    )


def _print_report(report: MethodReport) -> None:
    def _fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    print(
        f"{report.method:32s} n={report.n_evaluated:2d} "
        f"TP={report.true_positive} FP={report.false_positive} "
        f"TN={report.true_negative} FN={report.false_negative} ABSTAIN={report.abstain}  "
        f"precision={_fmt(report.precision)} recall={_fmt(report.recall)} "
        f"FPR={_fmt(report.false_positive_rate)} FNR={_fmt(report.false_negative_rate)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vlm",
        choices=["none", "real"],
        default="none",
        help="'real' loads the live Qwen2.5-VL client (GPU worker only, ADR 0003); "
        "'none' (default) skips the VLM method and runs geometric baselines only.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MASK_BENCHMARK_PATH)
    args = parser.parse_args()

    all_samples = load_mask_semantic_benchmark(args.dataset)
    samples = [s for s in all_samples if s.artifacts_available()]
    skipped = [s.sample_id for s in all_samples if not s.artifacts_available()]

    print(
        f"benchmark: {len(all_samples)} defined, {len(samples)} evaluated, {len(skipped)} skipped"
    )
    if skipped:
        print(f"  skipped (real local artifacts not present on this checkout): {skipped}")
    if not samples:
        raise SystemExit(
            "no benchmark samples have real local artifacts on this checkout -- nothing to evaluate"
        )

    # Import lazily -- avoids any cv2 dependency at module import time for callers that only
    # want the dataset loader.
    from manga_animation.validation.mask_semantics import _compute_geometric_signals

    signals_by_sample = {
        s.sample_id: _compute_geometric_signals(s.load_mask(), s.bbox) for s in samples
    }

    reports = [
        _geometric_signal_method(signal_name, samples, signals_by_sample)
        for signal_name in _GEOMETRIC_SIGNALS
    ]

    candidate_id: str | None = None
    source: str | None = None
    if args.vlm == "real":
        from manga_animation.analysis import Qwen25VLClient
        from manga_animation.benchmarking.registry import load_candidates
        from manga_animation.core.config import load_config

        config = load_config("kaggle")
        candidate_id = config.model_variants.get("vlm", "qwen2.5-vl-7b-instruct")
        # Resolve the shortlist id (e.g. "qwen2.5-vl-7b-instruct") to its real HF source (e.g.
        # "Qwen/Qwen2.5-VL-7B-Instruct") -- same resolution orchestrator.py::_candidate_source
        # already does; model identity is config-driven, never hardcoded (see "Model
        # Abstraction" in docs/architecture.md).
        source = next(c.source for c in load_candidates()["vlm"] if c.id == candidate_id)
        vlm_client = Qwen25VLClient(source=source, dtype=config.dtype)
        try:
            reports.append(_vlm_method(samples, vlm_client))
        finally:
            vlm_client.unload()
    else:
        print("--vlm none: skipping the VLM method (run with --vlm real on the GPU worker)")

    print()
    for report in reports:
        _print_report(report)
    print(
        "\nCAUTION (geometric:* rows only): each threshold above was found by an exhaustive "
        "best-fit sweep over these SAME 13 samples, then evaluated on those same 13 samples --\n"
        "this is training-set accuracy, not held-out generalization evidence. With n=13 and "
        "~24 (cutpoint x direction) configurations tried per signal, beating a naive majority-\n"
        "class baseline (always predict 'good': precision n/a, recall 0.0, accuracy 8/13=0.615) "
        "is expected from multiple-comparisons alone, not proof the signal generalizes. This\n"
        "reproduces Phase 11's own qualitative finding (docs/phase11-results.md section 7: "
        "these signals' raw VALUE RANGES overlap between confirmed-bad and good real masks) "
        "with\nfresh numbers, not evidence that overlapping ranges suddenly work when a "
        "threshold is hand-fit to them -- see docs/phase12-results.md's calibration-study "
        "section."
    )

    experiments_dir = Path("outputs/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = experiments_dir / f"phase12_semantic_benchmark_{timestamp}.json"
    dataset_fingerprint = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    output = {
        "git_commit": git_commit(),
        "dataset_path": str(args.dataset),
        "dataset_fingerprint": dataset_fingerprint,
        "environment": environment_metadata("cuda" if args.vlm == "real" else "cpu"),
        "n_defined": len(all_samples),
        "n_evaluated": len(samples),
        "skipped_sample_ids": skipped,
        "vlm_mode": args.vlm,
        "reports": [asdict(r) for r in reports],
    }
    if args.vlm == "real":
        output["vlm_candidate_id"] = candidate_id
        output["vlm_source"] = source
    out_path.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
