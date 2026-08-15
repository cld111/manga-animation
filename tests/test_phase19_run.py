"""Phase 19 runner tests: record assembly from a fake adapter output (no GPU), five-target
selection, and mask conversion. The GPU loop itself is not tested locally (torch-free)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from manga_animation.benchmarking.phase19.masks import SquarePad
from manga_animation.benchmarking.phase19.run import (
    assemble_controlled_record,
    masks_pairwise_overlap,
    masks_to_original,
    select_five_targets,
)
from tests.phase19_fixtures import make_phase19_manifest, make_phase19_sample

_IMAGE = np.zeros((60, 100, 3), dtype=np.uint8)
_GT = np.zeros((60, 100), dtype=bool)
_GT[20:60, 30:90] = True


def _padded_mask(canvas_size: int = 100) -> np.ndarray:
    pad = SquarePad.from_page_size((60, 100))
    canvas = np.zeros((canvas_size, canvas_size), dtype=bool)
    canvas[pad.sy : pad.sy + 60, pad.sx : pad.sx + 100] = _GT
    return canvas


def _out(*, text="[SEG]", masks=None):
    return SimpleNamespace(text=text, masks=masks if masks is not None else [_padded_mask()])


def test_masks_to_original_crops_padding():
    padded = [_padded_mask()]
    orig = masks_to_original(padded, (60, 100))
    assert orig[0].shape == (60, 100)
    assert np.array_equal(orig[0], _GT)


def test_assemble_record_ok(tmp_path):
    sample = make_phase19_sample()
    record = assemble_controlled_record(
        sample, _IMAGE, _GT, _out(),
        condition="D", prompt="Can you please segment character body in the given image",
        provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
        latency_seconds=1.5, vram_peak_mb=2048.0,
    )
    assert record.status == "ok"
    assert record.n_masks == 1
    assert record.metrics["iou"] == pytest.approx(1.0)
    assert record.pred_mask_path is not None
    saved = np.load(record.pred_mask_path)["mask"] > 0
    assert np.array_equal(saved, _GT)
    assert record.pred_bbox == [30, 20, 90, 60]
    assert record.latency_seconds == 1.5
    assert record.vram_peak_mb == 2048.0


def test_assemble_record_no_mask(tmp_path):
    sample = make_phase19_sample()
    record = assemble_controlled_record(
        sample, _IMAGE, _GT, _out(text="I cannot find a character body here", masks=[]),
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
    )
    assert record.n_masks == 0
    assert record.metrics is None
    assert record.target_not_found_text
    assert record.pred_mask_path is None


def test_assemble_record_error(tmp_path):
    sample = make_phase19_sample()
    record = assemble_controlled_record(
        sample, _IMAGE, _GT, None,
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
        error=RuntimeError("cuda oom"),
    )
    assert record.status == "inference_error"
    assert "cuda oom" in record.error_detail
    assert record.metrics is None


def test_assemble_record_multiple_instances(tmp_path):
    sample = make_phase19_sample()
    # Two masks on different instances: a valid mask + a disjoint one.
    disjoint = np.zeros((100, 100), dtype=bool)
    disjoint[0:20, 0:20] = True
    record = assemble_controlled_record(
        sample, _IMAGE, _GT, _out(text="[SEG][SEG]", masks=[_padded_mask(), disjoint]),
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
    )
    assert record.multi_instance
    assert record.mask_overlap is not None and record.mask_overlap < 0.5


def test_masks_pairwise_overlap_same_object_not_multiple(tmp_path):
    sample = make_phase19_sample()
    same = _padded_mask()
    record = assemble_controlled_record(
        sample, _IMAGE, _GT, _out(text="[SEG][SEG]", masks=[same, same.copy()]),
        condition="D", prompt="p", provenance="PRODUCTION_AVAILABLE", out_dir=tmp_path,
    )
    assert not record.multi_instance


def test_masks_pairwise_overlap_none_for_single():
    assert masks_pairwise_overlap([_padded_mask()]) is None


def test_select_five_targets_unique_and_deterministic():
    manifest = make_phase19_manifest(8)
    five = select_five_targets(manifest)
    assert len(five) == 5
    assert len({s.sample_id for s in five}) == 5
    assert [s.sample_id for s in select_five_targets(manifest)] == [
        s.sample_id for s in five
    ]
