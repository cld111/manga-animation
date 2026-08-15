"""Phase 19 autonomous-mode tests: page selection, artifact saving, and the gallery builder --
driven by a fake adapter so no GPU stack is needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase19.autonomous import (
    REVIEW_CRITERIA,
    build_autonomous_gallery,
    run_autonomous_pages,
    save_autonomous_records,
    unique_pages,
)
from manga_animation.benchmarking.phase19.masks import SquarePad
from tests.phase19_fixtures import make_phase19_manifest


def _write_dataset(tmp_path: Path, manifest) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for sample in manifest.samples:
        img = np.zeros((60, 100, 3), dtype=np.uint8)
        Image.fromarray(img).save(dataset / f"{sample.sample_id}.png")
    return dataset


class _FakeAdapter:
    """Minimal adapter stub: one mask on the padded canvas per prediction."""

    def __init__(self, text: str = "The character on the right is running. [SEG]"):
        self.text = text

    def predict(self, image, prompt):
        pad = SquarePad.from_page_size(tuple(image.shape[:2]))
        canvas = np.zeros((pad.canvas_size, pad.canvas_size), dtype=bool)
        canvas[pad.sy : pad.sy + 60, pad.sx : pad.sx + 100] = True
        return SimpleNamespace(text=self.text, masks=[canvas])


def test_unique_pages():
    manifest = make_phase19_manifest(6)
    pages = unique_pages(manifest)
    assert pages == ["TESTBOOK_001"]


def test_run_autonomous_pages_saves_artifacts(tmp_path):
    manifest = make_phase19_manifest(6)
    dataset = _write_dataset(tmp_path, manifest)
    out_dir = tmp_path / "out"
    records = run_autonomous_pages(
        manifest, dataset, out_dir, _FakeAdapter(),
        instruction="analyze and select", limit=None,
    )
    assert len(records) == 1  # one unique page
    record = records[0]
    assert record.status == "ok"
    assert record.n_masks == 1
    assert record.target_context == "character on the right is running."
    assert (out_dir / f"{record.page_key}.autonomous.page.png").exists()
    assert Path(record.mask_paths[0]).exists()
    assert record.bbox == [0, 0, 100, 60]  # full-band padded mask cropped to page geometry
    saved = save_autonomous_records(records, out_dir)
    assert saved.exists()


def test_run_autonomous_records_inference_error(tmp_path):
    class _Broken:
        def predict(self, image, prompt):
            raise RuntimeError("oom")

    manifest = make_phase19_manifest(3)
    dataset = _write_dataset(tmp_path, manifest)
    records = run_autonomous_pages(
        manifest, dataset, tmp_path / "out2", _Broken(),
        instruction="x", limit=1,
    )
    assert records[0].status == "inference_error"
    assert "oom" in records[0].error_detail
    assert records[0].n_masks == 0


def test_gallery_built(tmp_path):
    manifest = make_phase19_manifest(6)
    dataset = _write_dataset(tmp_path, manifest)
    out_dir = tmp_path / "out3"
    records = run_autonomous_pages(
        manifest, dataset, out_dir, _FakeAdapter(), instruction="x", limit=1
    )
    written = build_autonomous_gallery(records, out_dir)
    assert len(written) == 1
    assert written[0].exists()
    assert written[0].name == f"autonomous_{records[0].page_key}.png"


def test_review_criteria_present():
    assert set(REVIEW_CRITERIA) == {
        "semantic_plausibility",
        "animation_suitability",
        "instance_coherent",
        "safe",
        "mask_usable",
    }
