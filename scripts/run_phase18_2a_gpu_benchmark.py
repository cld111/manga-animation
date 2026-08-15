"""Phase 18.2A GPU benchmark: Qwen2.5-VL direct target localization + Qwen-bbox->SAM.

    For each of the 64 phase-17 `body` targets: can Qwen2.5-VL, given the full page and the
    production target description, localize the SPECIFIC target instance? Measures:
      - Qwen direct bbox  vs GT bbox            (bbox IoU, recall@0.5)
      - Qwen bbox -> SAM  vs GT mask            (downstream mask quality)
      - GT bbox  -> SAM   vs GT mask            (Phase 17 reference, re-measured same-run)

Run on the Kaggle/Jupyter GPU worker (only the Qwen VLM then SAM load; DINO is NOT touched):

    python scripts/run_phase18_2a_gpu_benchmark.py \
        --manifest configs/phase17_benchmark.yaml \
        --qwen /kaggle/working/models/qwen \
        --sam  /kaggle/working/models/sam \
        --out  outputs/experiments/phase18_2a_qwen_bbox

The VLM stage is cached per sample (`predictions_by_sample.json`) and the SAM masks are saved
as npz artifacts, so the run resumes. `--report-only` rebuilds report/visuals from an existing
run dir with no GPU work. `--profile` defaults to `kaggle` (VLM float16 -- the only verified
2xT4 configuration for Qwen2.5-VL-7B, see Phase 18.2).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient
from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest, load_manifest
from manga_animation.benchmarking.phase18a.report import build_report, write_report
from manga_animation.benchmarking.phase18a.run import (
    build_per_target_metrics,
    collect_direct_predictions,
    collect_sam_masks,
)
from manga_animation.benchmarking.phase18a.visuals import build_visual_packages
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.segmentation import Sam21Client

DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
DEFAULT_OUT = "outputs/experiments/phase18_2a_qwen_bbox"


def _trim_manifest(manifest: BenchmarkManifest, limit: int) -> BenchmarkManifest:
    return BenchmarkManifest(
        version=manifest.version,
        seed=manifest.seed,
        main_category=manifest.main_category,
        samples=manifest.samples[:limit],
    )


def _report_only(
    manifest: BenchmarkManifest, run_dir: Path, out_dir: Path, dataset_dir: Path
) -> None:
    """Rebuild metrics/report/visuals from a saved run dir (no GPU)."""
    from manga_animation.benchmarking.phase18a.metrics import PerTargetMetrics

    preds_path = run_dir / "predictions_by_sample.json"
    if not preds_path.exists():
        raise SystemExit(f"no predictions_by_sample.json in {run_dir}")
    raw = json.loads(preds_path.read_text(encoding="utf-8"))
    sam = {}
    sam_path = run_dir / "sam_masks_by_sample.json"
    if sam_path.exists():
        sam = json.loads(sam_path.read_text(encoding="utf-8"))

    targets: list[PerTargetMetrics] = []
    for sample in manifest.samples:
        entry = raw[sample.sample_id]
        pred = entry["prediction"]
        pixel_box = tuple(pred["pixel_box"]) if pred.get("pixel_box") else None
        bbox_iou = entry.get("bbox_iou")
        found = bool(pred["found"])
        error = pred.get("error")
        from manga_animation.benchmarking.phase18a.classify import classify

        classification = classify(
            sample.gt_bbox, pixel_box, found, error, entry["page_size"][1], entry["page_size"][0]
        )
        s = sam.get(sample.sample_id, {})
        targets.append(
            PerTargetMetrics(
                sample_id=sample.sample_id,
                gt_bbox=sample.gt_bbox,
                found=found,
                pixel_box=pixel_box,
                bbox_iou=bbox_iou,
                gt_coverage=entry.get("gt_coverage"),
                area_ratio=entry.get("area_ratio"),
                error=error,
                error_category=classification.name,
                gs_mask_iou=s.get("gs_iou"),
                qs_mask_iou=s.get("qs_iou"),
            )
        )

    report = build_report(targets)
    json_path, md_path = write_report(report, run_dir)
    print(f"report: {json_path}\n        {md_path}")
    written = build_visual_packages(targets, manifest, dataset_dir, run_dir / "visuals")
    print(f"visual packages: {len(written)} under {run_dir / 'visuals'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18.2A Qwen direct-bbox benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--qwen", required=True, help="Qwen VLM model dir on the worker")
    parser.add_argument("--sam", required=True, help="SAM 2.1 model dir on the worker")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N samples")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="VLM decode budget")
    parser.add_argument("--skip-report", action="store_true", help="skip CPU report/visuals")
    parser.add_argument(
        "--report-only",
        metavar="RUN_DIR",
        default=None,
        help="rebuild report/visuals from an existing run dir (no GPU); RUN_DIR is relative "
        "to --out",
    )
    parser.add_argument(
        "--profile",
        default="kaggle",
        help="config profile (default 'kaggle' -> VLM dtype float16, the only verified 2xT4 "
        "configuration for Qwen2.5-VL-7B; float32 OOMs on a shared T4 pair)",
    )
    args = parser.parse_args()

    setup_logging()
    manifest = load_manifest(Path(args.manifest))
    if args.limit is not None:
        if args.limit >= len(manifest.samples):
            raise SystemExit(f"--limit {args.limit} >= manifest size {len(manifest.samples)}")
        manifest = _trim_manifest(manifest, args.limit)

    out_dir = Path(args.out)
    dataset_dir = out_dir.parent / "phase17_object_segmentation" / "dataset"
    if not (dataset_dir / f"{manifest.samples[0].sample_id}.png").exists():
        raise SystemExit(
            f"dataset artifacts not found under {dataset_dir} -- run the phase-17 prepare step"
        )

    if args.report_only is not None:
        run_dir = out_dir / args.report_only
        _report_only(manifest, run_dir, out_dir, dataset_dir)
        return

    run_dir = out_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(env=args.profile)
    device = config.resolve_device()
    vlm_client = Qwen25VLClient(
        source=args.qwen, dtype=config.dtype, max_new_tokens=args.max_new_tokens
    )
    segmentation_client = Sam21Client(source=args.sam, device=device, dtype="float32")

    print(
        f"phase18.2a: {len(manifest.samples)} targets, device={device}, "
        f"vlm_dtype={config.dtype}"
    )

    records = collect_direct_predictions(
        manifest, dataset_dir, run_dir, vlm_client
    )
    sam_results = collect_sam_masks(manifest, dataset_dir, run_dir, segmentation_client, records)
    targets = build_per_target_metrics(manifest, records, sam_results)
    (run_dir / "per_target.json").write_text(
        json.dumps([t.as_dict() for t in targets], indent=1), encoding="utf-8"
    )

    if not args.skip_report:
        report = build_report(targets)
        json_path, md_path = write_report(report, run_dir)
        print(f"report: {json_path}\n        {md_path}")
        written = build_visual_packages(targets, manifest, dataset_dir, run_dir / "visuals")
        print(f"visual packages: {len(written)} under {run_dir / 'visuals'}")

    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "n_targets": len(manifest.samples),
                "limit": args.limit,
                "device": device,
                "vlm_dtype": config.dtype,
                "max_new_tokens": args.max_new_tokens,
                "created_at": datetime.now(UTC).isoformat(),
                "environment": environment_metadata(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("DONE")


if __name__ == "__main__":
    main()
