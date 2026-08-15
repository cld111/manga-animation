"""Phase 17 benchmark manifest: the canonical, tracked list of benchmark samples.

The manifest is committed source (like `configs/phase3_3_eval_dataset.yaml`); the normalized
page images / GT masks it references are git-ignored artifacts re-generated on demand. Sample
selection is deterministic (seeded) and stratified over size / aspect / page-context so a
small benchmark (~50-100 instances) still spans the brief's required variety instead of a
hand-picked easy subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from manga_animation.benchmarking.phase17.dataset import (
    MAIN_OBJECT_CATEGORY,
    CandidateInstance,
)

MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestSample:
    """One benchmark sample as recorded in the committed manifest. `gt_bbox`/`gt_area` are
    recorded (from the human mask, never modified) so the manifest is self-describing even on a
    checkout whose artifacts are missing; the artifact files remain the authoritative GT."""

    sample_id: str
    book: str
    page_index: int
    instance_id: int
    category: str
    semantic_label: str
    prompt: str  # the exact DINO text prompt the production pipeline would build
    gt_bbox: tuple[int, int, int, int]
    gt_area: int
    page_size: tuple[int, int] = field(default=(0, 0))  # (height, width), for feature context
    features: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "book": self.book,
            "page_index": self.page_index,
            "instance_id": self.instance_id,
            "category": self.category,
            "semantic_label": self.semantic_label,
            "prompt": self.prompt,
            "gt_bbox": list(self.gt_bbox),
            "gt_area": self.gt_area,
            "page_size": list(self.page_size),
            "features": self.features,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    version: int
    seed: int
    main_category: str
    samples: list[ManifestSample]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "main_category": self.main_category,
            "samples": [s.as_dict() for s in self.samples],
        }


def sample_prompt(semantic_label: str) -> str:
    """The exact prompt production grounding would build for this label (reuses the real
    `_prompt_from_label` so Experiment B/C measure the actual prompt, not a rephrased one)."""
    from manga_animation.grounding.ground import _prompt_from_label

    return _prompt_from_label(semantic_label)


def _features_for(candidate: CandidateInstance) -> dict[str, float]:
    w, h = candidate.page_size
    bw = candidate.gt_bbox[2] - candidate.gt_bbox[0]
    bh = candidate.gt_bbox[3] - candidate.gt_bbox[1]
    return {
        "area_fraction": (bw * bh) / (w * h) if w and h else 0.0,
        "aspect_ratio": bw / bh if bh else 0.0,
        "silhouette_density": candidate.gt_area / (bw * bh) if bw and bh else 0.0,
        "page_area": float(w * h),
    }


def select_samples(
    candidates: list[CandidateInstance],
    *,
    seed: int,
    target_size: int,
) -> list[ManifestSample]:
    """Deterministic, stratified selection of `target_size` samples from a candidate pool.

    Strata (each computed purely from the human GT, no model involved):
      - size: bbox area as a fraction of its page, split into terciles (small / medium / large);
      - page-context: pages with many candidate instances (touching/overlapping characters)
        vs isolated ones.

    The selector guarantees a minimum quota per size tercile and per page-context bucket so a
    small benchmark cannot silently collapse into one easy region, then fills the remainder
    deterministically (seeded RNG). This is a curation heuristic, not a calibrated sampling
    distribution -- the brief's "~50-100 instances, prefer diverse and difficult" target is a
    coverage requirement, not a statistical one.
    """
    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if target_size > len(candidates):
        raise ValueError(
            f"target_size {target_size} exceeds candidate pool size {len(candidates)}"
        )
    import random

    rng = random.Random(seed)

    if not candidates:
        raise ValueError("candidate pool is empty -- nothing to select from")

    area_fracs = sorted(_features_for(c)["area_fraction"] for c in candidates)
    n = len(area_fracs)
    lo_cut = area_fracs[max(0, n // 3 - 1)]
    hi_cut = area_fracs[min(n - 1, 2 * n // 3)]

    def size_bucket(features: dict[str, float]) -> str:
        if features["area_fraction"] < lo_cut:
            return "small"
        if features["area_fraction"] < hi_cut:
            return "medium"
        return "large"

    by_page: dict[str, list[CandidateInstance]] = {}
    for c in candidates:
        by_page.setdefault(f"{c.book}_{c.page_index:03d}", []).append(c)

    def context_bucket(c: CandidateInstance) -> str:
        return "crowded" if len(by_page[f"{c.book}_{c.page_index:03d}"]) >= 4 else "isolated"

    buckets: dict[tuple[str, str], list[CandidateInstance]] = {}
    for c in candidates:
        buckets.setdefault((size_bucket(_features_for(c)), context_bucket(c)), []).append(c)
    # Deterministic per-seed shuffle within each bucket so different seeds produce different
    # (still stratified) selections, and the same seed reproduces byte-identically.
    for key in buckets:
        rng.shuffle(buckets[key])

    selected: list[CandidateInstance] = []
    chosen_ids: set[str] = set()
    order = ["small", "medium", "large"]
    while len(selected) < target_size:
        progressed = False
        for size in order:
            for ctx in ("crowded", "isolated"):
                remaining = [
                    c for c in buckets.get((size, ctx), []) if c.sample_id not in chosen_ids
                ]
                if remaining and len(selected) < target_size:
                    c = remaining[0]
                    selected.append(c)
                    chosen_ids.add(c.sample_id)
                    progressed = True
                    break
            if len(selected) >= target_size:
                break
        if not progressed and len(selected) < target_size:
            # All buckets exhausted in round-robin; drain the pool deterministically.
            for c in candidates:
                if c.sample_id not in chosen_ids:
                    selected.append(c)
                    chosen_ids.add(c.sample_id)
                    if len(selected) >= target_size:
                        break

    selected = selected[:target_size]
    return [
        ManifestSample(
            sample_id=c.sample_id,
            book=c.book,
            page_index=c.page_index,
            instance_id=c.instance_id,
            category=c.category,
            semantic_label=c.semantic_label,
            prompt=sample_prompt(c.semantic_label),
            gt_bbox=c.gt_bbox,
            gt_area=c.gt_area,
            page_size=c.page_size,
            features=_features_for(c),
        )
        for c in selected
    ]


def build_manifest(
    candidates: list[CandidateInstance],
    *,
    seed: int,
    target_size: int,
) -> BenchmarkManifest:
    """Select and package a full manifest from the candidate pool."""
    selected = select_samples(candidates, seed=seed, target_size=target_size)
    return BenchmarkManifest(
        version=MANIFEST_VERSION,
        seed=seed,
        main_category=MAIN_OBJECT_CATEGORY,
        samples=selected,
    )


def write_manifest(manifest: BenchmarkManifest, path: Path) -> None:
    """Write the canonical manifest YAML. This file is committed; never write dataset artifacts
    into it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.as_dict(), sort_keys=False), encoding="utf-8")


def load_manifest(path: Path | str) -> BenchmarkManifest:
    """Load and validate a committed manifest YAML."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"manifest {path} has version {data.get('version')}, expected {MANIFEST_VERSION}"
        )
    samples = []
    for entry in data["samples"]:
        samples.append(
            ManifestSample(
                sample_id=entry["sample_id"],
                book=entry["book"],
                page_index=int(entry["page_index"]),
                instance_id=int(entry["instance_id"]),
                category=entry["category"],
                semantic_label=entry["semantic_label"],
                prompt=entry["prompt"],
                gt_bbox=(int(entry["gt_bbox"][0]), int(entry["gt_bbox"][1]),
                         int(entry["gt_bbox"][2]), int(entry["gt_bbox"][3])),
                gt_area=int(entry["gt_area"]),
                page_size=(
                    int(entry.get("page_size", [0, 0])[0]),
                    int(entry.get("page_size", [0, 0])[1]),
                ),
                features=dict(entry.get("features", {})),
            )
        )
    return BenchmarkManifest(
        version=int(data["version"]),
        seed=int(data["seed"]),
        main_category=data["main_category"],
        samples=samples,
    )
