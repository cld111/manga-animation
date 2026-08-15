"""Phase 17 GPU benchmark: run Experiments A/B/C with the real production models.

    EXPERIMENT A   GT bbox -> SAM 2.1 -> mask                 (pure SAM quality)
    EXPERIMENT B   image -> Grounding DINO -> bbox            (localization, no SAM)
    EXPERIMENT C   image -> DINO -> selection -> SAM -> gates (real production path)

Run on the Kaggle/Jupyter GPU worker, never locally (CLAUDE.md / ADR 0003). Only Grounding
DINO and SAM 2.1 run (phase brief section 14 -- no Qwen, no LaMa, no animation).

Usage (models as local dirs on the worker, dataset prepared first):

    python scripts/run_phase17_gpu_benchmark.py \
        --manifest configs/phase17_benchmark.yaml \
        --dino /kaggle/working/models/dino \
        --sam  /kaggle/working/models/sam \
        --out  outputs/experiments/phase17_object_segmentation

Writes per-sample intermediate masks/results plus report.json/report.md and the visual failure
packages under `--out`. Optionally `--limit N` for a short verification run.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.benchmarking.phase17.manifest import load_manifest
from manga_animation.benchmarking.phase17.report import (
    aggregate,
    build_visual_failures,
    compute_forbidden_overlap,
    write_report,
)
from manga_animation.benchmarking.phase17.run import run_benchmark_experiments
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.grounding import GroundingDinoClient
from manga_animation.segmentation import Sam21Client

DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
DEFAULT_OUT = "outputs/experiments/phase17_object_segmentation"


def _trim_manifest(manifest, limit: int):
    """Return a manifest-like object with only the first `limit` samples (verification runs)."""
    from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest

    return BenchmarkManifest(
        version=manifest.version,
        seed=manifest.seed,
        main_category=manifest.main_category,
        samples=manifest.samples[:limit],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 object segmentation GPU benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dino", required=True, help="Grounding DINO model dir on the worker")
    parser.add_argument("--sam", required=True, help="SAM 2.1 model dir on the worker")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N samples")
    parser.add_argument("--skip-report", action="store_true", help="skip CPU report/visualization")
    args = parser.parse_args()

    setup_logging()
    manifest = load_manifest(Path(args.manifest))
    if args.limit is not None:
        if args.limit >= len(manifest.samples):
            raise SystemExit(f"--limit {args.limit} >= manifest size {len(manifest.samples)}")
        manifest = _trim_manifest(manifest, args.limit)

    out_dir = Path(args.out)
    dataset_dir = out_dir / "dataset"
    run_dir = out_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Verify artifacts for every sample before spending a GPU pass.
    missing = [
        f"{s.sample_id}.png"
        for s in manifest.samples
        if not (dataset_dir / f"{s.sample_id}.png").exists()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} sample images missing under {dataset_dir} (run the prepare "
            f"script first): {missing[:5]}"
        )

    config = load_config()
    device = config.resolve_device()
    if device not in ("cuda",):
        print(f"WARNING: resolved device is {device!r} -- expected cuda on the GPU worker")

    # Production dtype choice (float32 for these two stages -- see orchestrator docstring).
    grounding_client = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    segmentation_client = Sam21Client(source=args.sam, device=device, dtype="float32")

    print(
        f"running {len(manifest.samples)} samples (A: SAM-on-GT-box, B: DINO, "
        "C: production path) on device {device}"
    )
    results = run_benchmark_experiments(
        manifest,
        dataset_dir,
        run_dir,
        grounding_client,
        segmentation_client,
    )

    per_sample_path = run_dir / "per_sample_results.json"
    per_sample_path.write_text(
        json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8"
    )
    print(f"per-sample results: {per_sample_path}")

    if not args.skip_report:
        report = aggregate(results)
        json_path, md_path = write_report(report, results, run_dir)
        print(f"report: {json_path}\n        {md_path}")

        visual_dir = run_dir / "visual_failures"
        written = build_visual_failures(results, manifest, dataset_dir, visual_dir)
        print(f"visual failure packages: {len(written)} under {visual_dir}")

        forbidden = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            forbidden = compute_forbidden_overlap(
                results, manifest, out_dir / "cache", hf_token
            )
        (run_dir / "forbidden_overlap.json").write_text(
            json.dumps(forbidden, indent=2), encoding="utf-8"
        )

    run_meta = {
        "manifest": args.manifest,
        "n_samples": len(manifest.samples),
        "limit": args.limit,
        "device": device,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"run metadata: {run_dir / 'run_meta.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
