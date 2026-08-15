"""Phase 18.2 GPU runner: score every DINO candidate with the production VLM mechanism, then
measure reranked selection accuracy.

Reuses Phase 18.1's saved per-page DINO detections (`detections_by_page.json`) so 18.1 and 18.2
are directly comparable. Loads the production Qwen VLM once (ModelStage), scores every UNIQUE
candidate box per page with the production verification prompt (cached to disk, resumable),
then ranks per strategy (A/B/C) and compares the selected candidate to GT -- evaluation only.

GPU discipline: only the VLM runs; DINO/SAM/animation are not loaded. VLM calls are cached per
(page, box) so a partial/failed run resumes without re-inferring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.analysis.client import VLMClient
from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase18.rerank import (
    SpecificCandidateScore,
    VlmCandidateScore,
    production_object_plan,
    rank_of_best_correct,
    rank_of_best_specific,
    rank_scores,
    rank_specific,
    selected_is_correct,
    specific_is_correct,
    specific_score_candidate,
    vlm_score_candidate,
)
from manga_animation.pipeline.lifecycle import ModelStage


def _box_key(box: list[int]) -> str:
    return f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"


def collect_vlm_scores(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    detections_by_page: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    vlm_client: VLMClient,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Score every unique candidate box per page. `mode` selects the prompt/scorer:
    "presence" (production `validate_target` prompt) or "specific" (benchmark-only
    instance-specific contrastive prompt). Returns `(scores_by_page, perf)`; caches per
    (page, box) to `vlm_scores_by_page.json` / `specific_scores_by_page.json` (resumable)."""
    return _collect_scores(
        manifest,
        dataset_dir,
        detections_by_page,
        out_dir,
        vlm_client,
        mode="presence",
    )


def collect_specific_scores(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    detections_by_page: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    vlm_client: VLMClient,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _collect_scores(
        manifest,
        dataset_dir,
        detections_by_page,
        out_dir,
        vlm_client,
        mode="specific",
    )


def _collect_scores(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    detections_by_page: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    vlm_client: VLMClient,
    *,
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    object_plan_ref = production_object_plan()
    score_fn: Any
    if mode == "presence":
        cache_name = "vlm_scores_by_page.json"
        score_fn = lambda image, box: vlm_score_candidate(  # noqa: E731
            vlm_client, image, object_plan_ref, box
        )
    else:
        cache_name = "specific_scores_by_page.json"
        score_fn = lambda image, box: specific_score_candidate(  # noqa: E731
            vlm_client, image, box
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / cache_name
    cached: dict[str, dict[str, dict[str, Any]]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"resuming from cached {cache_name}: {len(cached)} pages")

    sample_by_page: dict[str, Any] = {}
    for sample in manifest.samples:
        sample_by_page.setdefault(f"{sample.book}_{sample.page_index:03d}", sample)

    scores_by_page: dict[str, dict[str, dict[str, Any]]] = {}
    perf: dict[str, Any] = {
        "vlm_calls": 0,
        "cached_calls": 0,
        "total_elapsed_s": 0.0,
        "per_page_elapsed_s": {},
    }
    start = time.perf_counter()
    with ModelStage(vlm_client, name="vlm_rerank"):
        for page_key, detections in sorted(detections_by_page.items()):
            page_start = time.perf_counter()
            image = None
            page_scores: dict[str, dict[str, Any]] = {}
            for det in detections:
                box = [int(v) for v in det["box"]]
                key = _box_key(box)
                if key in cached.get(page_key, {}):
                    page_scores[key] = dict(cached[page_key][key])
                    perf["cached_calls"] += 1
                    continue
                if image is None:
                    sample = sample_by_page[page_key]
                    image = np.asarray(
                        Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB")
                    )
                raw = score_fn(image, (box[0], box[1], box[2], box[3]))
                entry = raw.as_dict()
                if mode == "presence":
                    entry["dino_score"] = float(det["score"])
                page_scores[key] = entry
                perf["vlm_calls"] += 1
            scores_by_page[page_key] = page_scores
            perf["per_page_elapsed_s"][page_key] = round(time.perf_counter() - page_start, 2)
            if perf["vlm_calls"] % 25 == 0:
                print(f"  {page_key}: {perf['vlm_calls']} VLM calls so far", flush=True)

    perf["total_elapsed_s"] = round(time.perf_counter() - start, 2)
    cache_path.write_text(
        json.dumps({page: scores for page, scores in scores_by_page.items()}, indent=1),
        encoding="utf-8",
    )
    return scores_by_page, perf


def rerank_targets(
    manifest: BenchmarkManifest,
    detections_by_page: dict[str, list[dict[str, Any]]],
    scores_by_page: dict[str, dict[str, VlmCandidateScore]],
    image_shapes: dict[str, tuple[int, int]],
    *,
    specific_scores_by_page: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Rank each target's page candidates per strategy (A/B/C) and, when the instance-specific
    scores are provided, strategy S (specific-instance contrastive, no DINO score)."""
    from manga_animation.benchmarking.phase17.metrics import bbox_iou

    results: list[dict[str, Any]] = []
    for sample in manifest.samples:
        page_key = f"{sample.book}_{sample.page_index:03d}"
        dets = detections_by_page.get(page_key, [])
        page_scores = scores_by_page.get(page_key, {})
        scores: list[VlmCandidateScore] = []
        for d in dets:
            e = page_scores.get(_box_key([int(v) for v in d["box"]]))
            if e is None:
                continue
            scores.append(
                VlmCandidateScore(
                    box=tuple(e["box"]),
                    dino_score=float(e.get("dino_score", 0.0)),
                    matches=e.get("matches"),
                    confidence=e.get("confidence"),
                    reason=e.get("reason"),
                )
            )
        gt = sample.gt_bbox
        entry: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "page_key": page_key,
            "gt_bbox": list(gt),
            "n_candidates": len(scores),
            "best_available_iou": max(
                (bbox_iou(gt, s.box) for s in scores), default=0.0
            ),
            "strategies": {},
        }
        for strategy in ("A", "B", "C"):
            ranked = rank_scores(
                scores, strategy, image_shape=image_shapes.get(page_key)
            )
            correct = selected_is_correct(gt, ranked)
            best_rank = rank_of_best_correct(gt, ranked)
            top1 = ranked[0] if ranked else None
            entry["strategies"][strategy] = {
                "selected_correct": correct,
                "best_correct_rank": best_rank,
                "selected_box": list(top1.box) if top1 else None,
                "selected_matches": top1.matches if top1 else None,
                "selected_confidence": top1.confidence if top1 else None,
                "selected_dino_score": top1.dino_score if top1 else None,
            }
        if specific_scores_by_page is not None:
            sp_page = specific_scores_by_page.get(page_key, {})
            sp_scores: list[SpecificCandidateScore] = []
            for d in dets:
                e = sp_page.get(_box_key([int(v) for v in d["box"]]))
                if e is None:
                    continue
                sp_scores.append(
                    SpecificCandidateScore(
                        box=tuple(e["box"]),
                        is_specific=e.get("is_specific"),
                        confidence=e.get("confidence"),
                        reason=e.get("reason"),
                    )
                )
            ranked_s = rank_specific(sp_scores)
            top_s = ranked_s[0] if ranked_s else None
            entry["strategies"]["S"] = {
                "selected_correct": specific_is_correct(gt, ranked_s),
                "best_correct_rank": rank_of_best_specific(gt, ranked_s),
                "selected_box": list(top_s.box) if top_s else None,
                "selected_matches": top_s.is_specific if top_s else None,
                "selected_confidence": top_s.confidence if top_s else None,
                "selected_dino_score": None,
            }
        results.append(entry)
    return results
