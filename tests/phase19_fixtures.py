"""Shared fixtures for the Phase 19 tests (a non-test module so tests can import it cleanly)."""

from __future__ import annotations

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest, ManifestSample


def make_phase19_sample(
    sample_id: str = "TESTBOOK_001_1",
    *,
    gt_bbox=(30, 20, 90, 60),
    features: dict | None = None,
) -> ManifestSample:
    return ManifestSample(
        sample_id=sample_id,
        book="TESTBOOK",
        page_index=1,
        instance_id=1,
        category="body",
        semantic_label="character_body",
        prompt="character body.",
        gt_bbox=gt_bbox,
        gt_area=2400,
        page_size=(60, 100),
        features=features or {},
    )


def make_phase19_manifest(n: int = 6) -> BenchmarkManifest:
    samples = [
        make_phase19_sample(f"TESTBOOK_00{i}_1", gt_bbox=(30, 20, 90, 60))
        for i in range(n)
    ]
    return BenchmarkManifest(version=1, seed=17, main_category="body", samples=samples)
