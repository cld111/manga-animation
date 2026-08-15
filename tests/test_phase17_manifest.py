"""Phase 17 manifest tests: deterministic stratified selection, quota guarantees, manifest
YAML round-trip, and the exact prompt the production pipeline would build."""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase17.dataset import CandidateInstance
from manga_animation.benchmarking.phase17.manifest import (
    build_manifest,
    load_manifest,
    sample_prompt,
    select_samples,
    write_manifest,
)


def make_pool(n: int, *, seed: int = 0) -> list[CandidateInstance]:
    """Synthetic pool: `n` body instances spread across pages, mixing sizes and crowding."""
    import random

    random.Random(seed)  # fixed seed so the pool's page layout is stable for the tests
    pool: list[CandidateInstance] = []
    for i in range(n):
        page = i // 5
        # Mix: small/large via bbox, crowded pages by construction (5 per page).
        if i % 3 == 0:
            box = (10, 10, 40, 60)  # small
        elif i % 3 == 1:
            box = (10, 10, 150, 200)  # large
        else:
            box = (30, 40, 130, 160)  # medium
        pool.append(
            CandidateInstance(
                sample_id=f"B_{page:03d}_{i}",
                book="B",
                page_index=page,
                instance_id=i,
                category="body",
                semantic_label="character_body",
                gt_bbox=box,
                gt_area=(box[2] - box[0]) * (box[3] - box[1]),
                page_size=(240, 200),
            )
        )
    return pool


def test_select_samples_is_deterministic():
    pool = make_pool(60)
    a = select_samples(pool, seed=42, target_size=20)
    b = select_samples(pool, seed=42, target_size=20)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert len(a) == 20


def test_select_samples_different_seed_different_selection():
    pool = make_pool(60)
    a = select_samples(pool, seed=42, target_size=20)
    b = select_samples(pool, seed=7, target_size=20)
    assert [s.sample_id for s in a] != [s.sample_id for s in b]


def test_select_samples_covers_all_size_strata():
    pool = make_pool(60)
    selected = select_samples(pool, seed=42, target_size=20)
    sizes = [s.features["area_fraction"] for s in selected]
    # The pool's small/large extremes must both be represented (not just medium).
    assert min(sizes) < 0.1
    assert max(sizes) > 0.5


def test_select_samples_covers_crowded_and_isolated_pages():
    pool = make_pool(60)  # 5 instances per page -> all "crowded" by the >=4 rule
    selected = select_samples(pool, seed=42, target_size=20)
    pages = {s.page_index for s in selected}
    assert len(pages) >= 4  # spreads across pages, not one page


def test_select_samples_raises_when_target_exceeds_pool():
    pool = make_pool(10)
    with pytest.raises(ValueError, match="exceeds candidate pool"):
        select_samples(pool, seed=1, target_size=11)


def test_sample_prompt_uses_production_convention():
    assert sample_prompt("character_body") == "character body."
    assert sample_prompt("raised_sword") == "raised sword."


def test_manifest_roundtrip(tmp_path):
    pool = make_pool(30)
    manifest = build_manifest(pool, seed=5, target_size=8)
    assert manifest.main_category == "body"
    path = tmp_path / "manifest.yaml"
    write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded.version == manifest.version
    assert [s.sample_id for s in loaded.samples] == [s.sample_id for s in manifest.samples]
    assert loaded.samples[0].prompt == "character body."
    assert loaded.samples[0].gt_bbox == manifest.samples[0].gt_bbox
    assert loaded.samples[0].features == manifest.samples[0].features


def test_load_manifest_rejects_wrong_version(tmp_path):
    pool = make_pool(10)
    manifest = build_manifest(pool, seed=1, target_size=3)
    path = tmp_path / "bad.yaml"
    write_manifest(manifest, path)
    text = path.read_text().replace("version: 1", "version: 99")
    path.write_text(text)
    with pytest.raises(ValueError, match="version"):
        load_manifest(path)
