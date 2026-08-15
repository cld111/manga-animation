"""Phase 17 dataset preparation: MS92 annotations + Manga109 images -> normalized benchmark.

Builds the committed manifest (`configs/phase17_benchmark.yaml`) from a deterministic,
size/context-stratified selection over the body-instance candidate pool, then materializes
the selected samples (page images + GT masks) as git-ignored artifacts.

Two-stage by design (GPU discipline): the candidate pool needs only MS92 annotations; page
images are downloaded ONLY for the pages the manifest actually selected, so a large candidate
book list costs ~tens of MB, not the full Manga109.

Usage (needs HF_TOKEN for the gated MS92 annotations):

    HF_TOKEN=<token> uv run python scripts/run_phase17_prepare_dataset.py \
        --out outputs/experiments/phase17_object_segmentation

Runs locally (CPU + download) or on the remote worker; deterministic given the same token and
inputs. Never modifies GT masks (pycocotools RLE decode + tight-bbox derivation only).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import yaml

from manga_animation.benchmarking.phase17.dataset import (
    MAIN_OBJECT_CATEGORY,
    candidate_pool_from_ms92,
    materialize_samples,
    mirror_available_pages,
)
from manga_animation.benchmarking.phase17.manifest import build_manifest, write_manifest

DEFAULT_OUT = "outputs/experiments/phase17_object_segmentation"
DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
BOOKS_CONFIG = "configs/phase17_benchmark_books.yaml"


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN must be set: the MS92/MangaSegmentation annotations are gated and "
            "require authentication (and prior gated-access approval)."
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Phase 17 benchmark dataset")
    parser.add_argument("--books-config", default=BOOKS_CONFIG, help="book list + selection config")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="output manifest path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output dir for normalized artifacts")
    parser.add_argument("--cache", default=None, help="cache dir for downloaded MS92 jsons")
    args = parser.parse_args()

    token = _require_token()
    config = yaml.safe_load(Path(args.books_config).read_text(encoding="utf-8"))
    if int(config["version"]) != 1:
        raise SystemExit(f"unsupported books config version {config['version']}")
    seed = int(config["seed"])
    target_size = int(config["target_size"])
    verify_pages = int(config["verify_pages"])
    books = list(config["candidate_books"])
    out_dir = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out_dir / "cache"

    print(f"building candidate pool from {len(books)} books (category={MAIN_OBJECT_CATEGORY}) ...")
    available = mirror_available_pages(token)
    print(f"mirror image availability: {len(available)} pages across {len(books)} candidate books")
    pool, _page_sizes = candidate_pool_from_ms92(token, books, cache_dir, available_pages=available)
    pool.sort(key=lambda c: c.sample_id)
    n_pages = len({(c.book, c.page_index) for c in pool})
    print(f"pool: {len(pool)} body instances across {n_pages} pages, {len(books)} books")

    if len(pool) < target_size:
        raise SystemExit(
            f"pool too small ({len(pool)} < target_size {target_size}) -- add candidate books"
        )

    manifest = build_manifest(pool, seed=seed, target_size=target_size)
    write_manifest(manifest, Path(args.manifest))
    print(f"manifest written: {args.manifest} ({len(manifest.samples)} samples)")

    print("materializing selected samples (images + GT masks) ...")
    samples = materialize_samples(
        manifest,
        hf_token=token,
        out_dir=out_dir,
        cache_dir=cache_dir,
        verify_pages=verify_pages,
    )
    print(f"materialized {len(samples)} samples under {out_dir / 'dataset'}")

    summary = {
        "manifest": str(Path(args.manifest)),
        "seed": seed,
        "target_size": target_size,
        "candidate_books": books,
        "pool_size": len(pool),
        "materialized": len(samples),
        "pages": sorted({f"{s.book}_{s.page_index:03d}" for s in samples}),
        "per_book": {
            book: sum(1 for s in samples if s.book == book)
            for book in sorted({s.book for s in samples})
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    summary_path = out_dir / "prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
