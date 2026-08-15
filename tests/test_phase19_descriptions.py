"""Phase 19 description-classification tests (phase brief section 7: only PRODUCTION-AVAILABLE
descriptions may feed the primary controlled result)."""

from __future__ import annotations

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest, ManifestSample
from manga_animation.benchmarking.phase19.descriptions import (
    PRODUCTION_AVAILABLE,
    classify_manifest_descriptions,
    production_available_only,
)


def _sample(sample_id: str, semantic_label: str = "character_body") -> ManifestSample:
    return ManifestSample(
        sample_id=sample_id,
        book="TESTBOOK",
        page_index=1,
        instance_id=123,
        category="body",
        semantic_label=semantic_label,
        prompt=f"{semantic_label.replace('_', ' ')}.",
        gt_bbox=(0, 0, 10, 10),
        gt_area=100,
        page_size=(100, 100),
    )


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        version=1,
        seed=17,
        main_category="body",
        samples=[_sample("A_001_1"), _sample("A_002_2", semantic_label="flag_cloth")],
    )


def test_classify_marks_production_descriptions():
    classes = classify_manifest_descriptions(_manifest())
    assert len(classes) == 2
    for c in classes:
        assert c.provenance_d == PRODUCTION_AVAILABLE
        assert c.condition_d.provenance == PRODUCTION_AVAILABLE
    assert classes[0].production_description == "character body"
    assert classes[0].production_prompt == "character body."
    assert classes[0].condition_d.prompt.startswith("Can you please segment character body")


def test_production_available_only_filters():
    classes = classify_manifest_descriptions(_manifest())
    allowed = production_available_only(classes)
    assert len(allowed) == 2  # with the current manifest every target is production-available
