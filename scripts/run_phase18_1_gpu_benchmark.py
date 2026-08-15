"""Phase 18.1 GPU benchmark: DINO candidate recall.

    For each of the 64 phase-17 GT `body` targets, does a correct bbox exist among ALL
    Grounding DINO detections on its page, and at what rank by DINO confidence?

Run on the Kaggle/Jupyter GPU worker (only Grounding DINO loads):

    python scripts/run_phase18_1_gpu_benchmark.py \
        --manifest configs/phase17_benchmark.yaml \
        --dino /kaggle/working/models/dino \
        --out  outputs/experiments/phase18_1_candidate_recall

Writes `detections_by_page.json`, `per_target_recall.json`, and `report.json`/`report.md`
under `--out`. `--limit` restricts the number of unique pages (verification runs).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.benchmarking.phase17.manifest import load_manifest
from manga_animation.benchmarking.phase18.report import build_report, write_report
from manga_animation.benchmarking.phase18.run import collect_detections, compute_target_recall
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.grounding import GroundingDinoClient

DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
DEFAULT_OUT = "outputs/experiments/phase18_1_candidate_recall"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18.1 DINO candidate recall benchmark")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dino", required=True, help="Grounding DINO model dir on the worker")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="limit number of unique pages")
    args = parser.parse_args()

    setup_logging()
    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.out)
    dataset_dir = out_dir.parent / "phase17_object_segmentation" / "dataset"
    # The phase-17 dataset lives next to this experiment's parent output dir.
    if not dataset_dir.exists():
        # allow an explicit dataset dir alongside --out
        dataset_dir = out_dir / "dataset"
    if not (dataset_dir / f"{manifest.samples[0].sample_id}.png").exists():
        raise SystemExit(
            "dataset artifacts not found under "
            f"{dataset_dir} -- run the phase-17 prepare step first"
        )

    if args.limit is not None:
        unique = sorted({f"{s.book}_{s.page_index:03d}" for s in manifest.samples})
        keep = set(unique[: args.limit])
        from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest

        manifest = BenchmarkManifest(
            version=manifest.version,
            seed=manifest.seed,
            main_category=manifest.main_category,
            samples=[s for s in manifest.samples if f"{s.book}_{s.page_index:03d}" in keep],
        )

    config = load_config()
    device = config.resolve_device()
    grounding_client = GroundingDinoClient(source=args.dino, device=device, dtype="float32")

    print(
        f"collecting DINO detections for {len({s.sample_id for s in manifest.samples})} targets "
        f"across {len({f'{s.book}_{s.page_index:03d}' for s in manifest.samples})} pages"
    )
    detections_by_page = collect_detections(manifest, dataset_dir, out_dir, grounding_client)
    targets = compute_target_recall(manifest, detections_by_page)
    (out_dir / "per_target_recall.json").write_text(
        json.dumps([t.as_dict() for t in targets], indent=1), encoding="utf-8"
    )
    report = build_report(targets)
    json_path, md_path = write_report(report, out_dir)
    print(f"report: {json_path}\n        {md_path}")

    run_meta = {
        "manifest": args.manifest,
        "n_targets": report.n_targets,
        "n_pages": report.n_pages,
        "device": device,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
