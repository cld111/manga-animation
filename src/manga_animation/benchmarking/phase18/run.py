"""Phase 18.1 GPU runner: collect ALL DINO detections per page and measure candidate recall.

The phase-17 data only kept DINO's top-3 detections per sample, so this phase re-runs DINO
once per UNIQUE page (52 pages for the phase-17 manifest) with the exact production prompt
and records every detection above DINO's own threshold. Recall is then measured locally (the
pure logic in `candidates.py`): for each of the 64 GT targets, is a correct candidate present
in its page's detection set, and what is its rank by DINO confidence?

Faithful to production: the same `GroundingDinoClient` and the same `_prompt_from_label`
prompt; `max_candidates` is NOT applied because recall is about what is *available to select
from*, and the cap is a selection step, not a detection step.

GPU discipline: only Grounding DINO runs; SAM/animation/etc. are not touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.dataset import MangaSegSample
from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase18.candidates import TargetRecall, measure_target
from manga_animation.grounding.client import GroundingClient
from manga_animation.pipeline.lifecycle import ModelStage


def _unique_pages(manifest: BenchmarkManifest) -> dict[str, list[MangaSegSample]]:
    pages: dict[str, list[MangaSegSample]] = {}
    for sample in manifest.samples:
        pages.setdefault(f"{sample.book}_{sample.page_index:03d}", []).append(sample)
    return pages


def collect_detections(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    grounding_client: GroundingClient,
) -> dict[str, list[dict[str, Any]]]:
    """Run DINO once per unique page and save every detection. Returns
    `{page_key: [{"box": [...], "score": ...}, ...]}` (also persisted to
    `out_dir/detections_by_page.json`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _unique_pages(manifest)
    detections_by_page: dict[str, list[dict[str, Any]]] = {}
    with ModelStage(grounding_client, name="grounding"):
        for page_key, samples in sorted(pages.items()):
            prompt = samples[0].prompt
            image_path = dataset_dir / f"{samples[0].sample_id}.png"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            detections = grounding_client.detect(image, prompt)
            detections_by_page[page_key] = [
                {"box": list(d.box), "score": d.score} for d in detections
            ]
    (out_dir / "detections_by_page.json").write_text(
        json.dumps(detections_by_page, indent=1), encoding="utf-8"
    )
    return detections_by_page


def _raw_detection(box: list[int], score: float):
    class _Raw:
        def __init__(self, box, score):
            self.box = box
            self.score = score

    return _Raw(box, score)


def compute_target_recall(
    manifest: BenchmarkManifest, detections_by_page: dict[str, list[dict[str, Any]]]
) -> list[TargetRecall]:
    """Measure recall for every GT target against its page's detection set."""
    records: list[TargetRecall] = []
    for sample in manifest.samples:
        page_key = f"{sample.book}_{sample.page_index:03d}"
        dets = detections_by_page.get(page_key, [])
        records.append(
            measure_target(
                sample_id=sample.sample_id,
                page_key=page_key,
                gt_bbox=sample.gt_bbox,
                detections=[_raw_detection(d["box"], d["score"]) for d in dets],
            )
        )
    return records
