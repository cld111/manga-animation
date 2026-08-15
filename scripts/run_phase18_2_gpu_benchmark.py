"""Phase 18.2 GPU benchmark: VLM-guided candidate reranking.

    For each of the 64 phase-17/18.1 targets: score every DINO candidate on its page with the
    production Qwen VLM verification mechanism, rank by strategies A/B/C, and measure selection
    accuracy @1 against GT (evaluation only; GT never reaches the selector).

Run on the Kaggle/Jupyter GPU worker (only the Qwen VLM loads; DINO detections are REUSED from
Phase 18.1's saved `detections_by_page.json`):

    python scripts/run_phase18_2_gpu_benchmark.py \
        --manifest configs/phase17_benchmark.yaml \
        --detections outputs/experiments/phase18_1_candidate_recall/detections_by_page.json \
        --vlm  /kaggle/working/models/qwen \
        --out  outputs/experiments/phase18_2_candidate_reranking

VLM scores are cached per (page, box) under `--out/vlm_scores_by_page.json` so the run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient
from manga_animation.benchmarking.phase17.manifest import load_manifest
from manga_animation.benchmarking.phase18.report_rerank import build_report, write_report
from manga_animation.benchmarking.phase18.run_rerank import (
    collect_vlm_scores,
    rerank_targets,
)
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata

DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
DEFAULT_DETECTIONS = (
    "outputs/experiments/phase18_1_candidate_recall/detections_by_page.json"
)
DEFAULT_OUT = "outputs/experiments/phase18_2_candidate_reranking"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18.2 VLM candidate reranking")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--detections", default=DEFAULT_DETECTIONS)
    parser.add_argument("--vlm", required=True, help="Qwen VLM model dir on the worker")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="limit unique pages")
    parser.add_argument(
        "--profile",
        default="kaggle",
        help="config profile (default 'kaggle' -> VLM dtype float16, the only verified 2xT4 "
        "configuration for Qwen2.5-VL-7B; float32 OOMs on a shared T4 pair)",
    )
    args = parser.parse_args()

    setup_logging()
    manifest = load_manifest(Path(args.manifest))
    detections_by_page = json.loads(Path(args.detections).read_text(encoding="utf-8"))
    if args.limit is not None:
        keep = set(sorted(detections_by_page)[: args.limit])
        detections_by_page = {k: v for k, v in detections_by_page.items() if k in keep}
        from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest

        manifest = BenchmarkManifest(
            version=manifest.version,
            seed=manifest.seed,
            main_category=manifest.main_category,
            samples=[s for s in manifest.samples if f"{s.book}_{s.page_index:03d}" in keep],
        )

    out_dir = Path(args.out)
    dataset_dir = out_dir.parent / "phase17_object_segmentation" / "dataset"
    if not (dataset_dir / f"{manifest.samples[0].sample_id}.png").exists():
        raise SystemExit(f"dataset artifacts not found under {dataset_dir}")

    config = load_config(env=args.profile)
    device = config.resolve_device()
    vlm_client = Qwen25VLClient(source=args.vlm, dtype=config.dtype)

    n_pages = len(detections_by_page)
    n_candidates = sum(len(v) for v in detections_by_page.values())
    print(
        f"scoring {n_candidates} unique candidate boxes over {n_pages} pages with the "
        f"production Qwen VLM (device={device})"
    )
    t0 = time.perf_counter()
    scores_by_page, perf = collect_vlm_scores(
        manifest, dataset_dir, detections_by_page, out_dir, vlm_client
    )
    perf["wall_s"] = round(time.perf_counter() - t0, 2)
    image_shapes: dict[str, tuple[int, int]] = {}
    for sample in manifest.samples:
        page_key = f"{sample.book}_{sample.page_index:03d}"
        from PIL import Image

        with Image.open(dataset_dir / f"{sample.sample_id}.png") as img:
            image_shapes.setdefault(page_key, img.size[::-1])  # (h, w)
    per_target = rerank_targets(manifest, detections_by_page, scores_by_page, image_shapes)
    (out_dir / "per_target_rerank.json").write_text(
        json.dumps(per_target, indent=1), encoding="utf-8"
    )
    report = build_report(per_target, perf)
    json_path, md_path = write_report(report, out_dir)
    print(f"report: {json_path}\n        {md_path}")
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "detections": args.detections,
                "n_targets": report.n_targets,
                "n_pages": n_pages,
                "n_candidates": n_candidates,
                "device": device,
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
