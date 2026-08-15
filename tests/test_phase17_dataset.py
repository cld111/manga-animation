"""Phase 17 dataset-prep tests: RLE decode, tight-bbox derivation, MS92 category filtering,
candidate pool building, mirror-agreement verification, and artifact materialization. Uses
synthetic in-memory MS92-format book JSONs with pycocotools-generated RLEs (no network)."""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as coco_mask

import manga_animation.benchmarking.phase17.dataset as ds
from manga_animation.benchmarking.phase17.dataset import (
    CATEGORY_IDS,
    CandidateInstance,
    MangaSegSample,
    candidate_pool_from_ms92,
    decode_rle,
    materialize_samples,
    tight_bbox_from_mask,
    verify_ms92_vs_mirror,
)


def _rle(mask: np.ndarray) -> dict:
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))


def make_book_json(
    pages: dict[int, tuple[int, int]],
    instances: dict[int, list[tuple[int, tuple[int, int, int, int]]]],
) -> dict:
    """`pages`: page_index -> (H, W); `instances`: page_index -> [(category_id, (x0,y0,x1,y1))]."""
    images = [
        {"id": pid, "width": w, "height": h, "file_name": f"BOOK/{pid:03d}.jpg"}
        for pid, (h, w) in pages.items()
    ]
    annotations = []
    ann_id = 0
    for pid, page_instances in instances.items():
        h, w = pages[pid]
        for category_id, (x0, y0, x1, y1) in page_instances:
            mask = np.zeros((h, w), dtype=bool)
            mask[y0:y1, x0:x1] = True
            annotations.append(
                {
                    "id": ann_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "image_id": pid,
                    "segmentation": _rle(mask),
                }
            )
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": []}


def _body(page: int, bbox: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int, int]]:
    return (CATEGORY_IDS["body"], bbox)


def _face(page: int, bbox: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int, int]]:
    return (CATEGORY_IDS["face"], bbox)


@pytest.fixture
def book_json():
    pages = {0: (100, 120), 1: (80, 90)}
    instances = {
        0: [_body(0, (10, 10, 30, 40)), _body(0, (50, 20, 90, 80)), _face(0, (12, 12, 22, 30))],
        1: [_body(1, (5, 5, 15, 15))],
    }
    return make_book_json(pages, instances)


def test_decode_rle_roundtrip():
    mask = np.zeros((40, 50), dtype=bool)
    mask[5:20, 10:30] = True
    decoded = decode_rle(_rle(mask))
    assert decoded.shape == (40, 50)
    assert np.array_equal(decoded, mask)


def test_tight_bbox_from_mask_half_open():
    mask = np.zeros((50, 60), dtype=bool)
    mask[7:23, 11:45] = True
    assert tight_bbox_from_mask(mask) == (11, 7, 45, 23)


def test_tight_bbox_empty_mask_raises():
    with pytest.raises(ValueError, match="empty mask"):
        tight_bbox_from_mask(np.zeros((10, 10), dtype=bool))


def test_pool_builds_only_body_instances(book_json, monkeypatch, tmp_path):
    def fake_ms92(token, book, cache_dir):
        return book_json

    monkeypatch.setattr(ds, "ms92_book_annotations", fake_ms92)
    pool, page_sizes = candidate_pool_from_ms92("tok", ["BOOK"], tmp_path)
    assert len(pool) == 3  # face instance excluded
    assert all(c.category == "body" for c in pool)
    by_id = {c.sample_id: c for c in pool}
    assert "BOOK_000_0" in by_id  # body instance with ann id 0
    assert "BOOK_001_3" in by_id  # page-1 body instance (the 4th annotation overall)
    first = by_id["BOOK_000_0"]
    assert first.gt_bbox == (10, 10, 30, 40)
    assert first.gt_area == 20 * 30
    assert first.page_size == (100, 120)
    assert first.semantic_label == "character_body"
    assert page_sizes["BOOK_000"] == (100, 120)
    assert page_sizes["BOOK_001"] == (80, 90)


def test_verify_ms92_vs_mirror_identical(book_json):
    page0_anns = [a for a in book_json["annotations"] if a["image_id"] == 0]
    mirror = {"annotations": page0_anns}
    assert verify_ms92_vs_mirror(page0_anns, mirror, "body") == pytest.approx(1.0)


def test_verify_ms92_vs_mirror_disagreeing_masks(book_json):
    page0_anns = [a for a in book_json["annotations"] if a["image_id"] == 0]
    anns = [dict(a) for a in page0_anns]
    # corrupt EVERY body instance's mask to empty so even the best pair cannot agree
    for a in anns:
        if a["category_id"] == CATEGORY_IDS["body"]:
            a["segmentation"] = _rle(np.zeros((100, 120), dtype=bool))
    mirror = {"annotations": anns}
    agreement = verify_ms92_vs_mirror(page0_anns, mirror, "body")
    assert agreement is not None and agreement < 0.5


def test_verify_ms92_vs_mirror_none_when_counts_disagree(book_json):
    page0_anns = [a for a in book_json["annotations"] if a["image_id"] == 0]
    fewer = page0_anns[:1]
    assert verify_ms92_vs_mirror(page0_anns, {"annotations": fewer}, "body") is None


def test_materialize_writes_artifacts_and_checks_dimensions(book_json, tmp_path, monkeypatch):
    from manga_animation.benchmarking.phase17 import manifest as man_mod

    candidates = [
        CandidateInstance(
            sample_id="BOOK_000_0",
            book="BOOK",
            page_index=0,
            instance_id=0,
            category="body",
            semantic_label="character_body",
            gt_bbox=(10, 10, 30, 40),
            gt_area=600,
            page_size=(100, 120),
        )
    ]
    manifest = man_mod.build_manifest(candidates, seed=1, target_size=1)

    captured: dict[str, np.ndarray] = {}
    monkeypatch.setattr(ds, "ms92_book_annotations", lambda tok, book, cache: book_json)

    def fake_image(tok, book, page):
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        captured[(book, page)] = img
        return img

    monkeypatch.setattr(ds, "mirror_page_image", fake_image)
    samples = materialize_samples(
        manifest, hf_token="tok", out_dir=tmp_path, cache_dir=tmp_path / "cache", verify_pages=0
    )
    assert len(samples) == 1
    s: MangaSegSample = samples[0]
    assert s.image_path.exists()
    assert s.mask_path.exists()
    mask = np.load(s.mask_path)["mask"]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert np.array_equal(mask > 0, decode_rle(book_json["annotations"][0]["segmentation"]))


def test_materialize_skips_dimension_mismatch(tmp_path, monkeypatch, book_json):
    from manga_animation.benchmarking.phase17 import manifest as man_mod

    candidates = [
        CandidateInstance(
            sample_id="BOOK_000_0",
            book="BOOK",
            page_index=0,
            instance_id=0,
            category="body",
            semantic_label="character_body",
            gt_bbox=(10, 10, 30, 40),
            gt_area=600,
            page_size=(100, 120),
        )
    ]
    manifest = man_mod.build_manifest(candidates, seed=1, target_size=1)
    monkeypatch.setattr(ds, "ms92_book_annotations", lambda tok, book, cache: book_json)
    # Wrong-sized mirror image: the MS92 record says 100x120, mirror returns 50x60.
    monkeypatch.setattr(
        ds, "mirror_page_image", lambda tok, book, page: np.zeros((50, 60, 3), dtype=np.uint8)
    )
    samples = materialize_samples(
        manifest, hf_token="tok", out_dir=tmp_path, cache_dir=tmp_path / "cache", verify_pages=0
    )
    assert samples == []
    skipped = (tmp_path / "dataset" / "skipped_pages.json").read_text()
    assert "does not match MS92 record" in skipped


def test_candidate_pool_sample_id_and_determinism(book_json, monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "ms92_book_annotations", lambda tok, book, cache: book_json)
    pool1, _ = candidate_pool_from_ms92("tok", ["BOOK"], tmp_path)
    pool2, _ = candidate_pool_from_ms92("tok", ["BOOK"], tmp_path)
    assert [c.sample_id for c in pool1] == [c.sample_id for c in pool2]
    assert pool1[0].sample_id == "BOOK_000_0"


def test_page_from_file_name():
    assert ds._page_from_file_name("ARMS/072.jpg") == 72
    assert ds._page_from_file_name("AkkeraKanjinchou/000.jpg") == 0


def test_pool_ignores_other_books_embedded_in_the_book_json(monkeypatch, tmp_path):
    # A book JSON can embed other books' pages in its images array; only its own pages count.
    pages = {0: (100, 120)}
    instances = {0: [_body(0, (10, 10, 30, 40))]}
    book_json = make_book_json(pages, instances)
    # Embed ARMS/072 (same within-book page number as BOOK/000 would collide with) -- a
    # same-numbered foreign page must never overwrite the BOOK page's size.
    book_json["images"].append({"id": 999, "width": 55, "height": 44, "file_name": "ARMS/000.jpg"})
    monkeypatch.setattr(ds, "ms92_book_annotations", lambda tok, book, cache: book_json)
    pool, page_sizes = candidate_pool_from_ms92("tok", ["BOOK"], tmp_path)
    assert page_sizes["BOOK_000"] == (100, 120)
    assert "ARMS_000" not in page_sizes
    assert len(pool) == 1
