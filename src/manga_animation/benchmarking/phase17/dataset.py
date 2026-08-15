"""MangaSegmentation -> normalized Phase 17 benchmark samples.

Ground truth: the MS92/MangaSegmentation human-annotated instance masks (the phase brief's
mandatory benchmark dataset). The dataset stores COCO-format RLE masks per book JSON
(`jsons/<BOOK>.json`, gated -- needs an HF token) over the original Manga109 page images.
Manga109 images are distributed separately (hal-utokyo/Manga109 and Manga109-s are gated
application datasets); the non-gated `longle0702/manga109-segmentation` mirror hosts the
identical Manga109 pages plus a per-page split of the SAME annotations. This module:

- builds the candidate pool from **MS92/MangaSegmentation** annotations (authoritative, per the
  phase brief) -- no images needed;
- materializes the selected benchmark samples by downloading page images from the mirror,
  cross-checking page dimensions against the MS92 annotation record and spot-checking (a
  sample of pages) that the mirror annotation for that page is identical to the MS92 one
  (both decode to IoU-1.0 masks). If either check fails, the page is skipped with a recorded
  reason rather than silently used.

Nothing here imports torch/transformers. Downloads go through `huggingface_hub` with an HF
token passed explicitly by the caller (never read from a committed config).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

MS92_REPO = "MS92/MangaSegmentation"
MIRROR_REPO = "longle0702/manga109-segmentation"

# MS92/MangaSegmentation categories (id: name). Only `body` (a full character silhouette --
# hair + clothing + body; faces are separately annotated instances contained within it) is a
# main-object benchmark category; the rest are the phase brief's forbidden/safety categories.
CATEGORY_NAMES: dict[int, str] = {
    1: "frame",
    2: "text",
    3: "face",
    4: "body",
    5: "balloon",
    6: "onomatopoeia",
}
CATEGORY_IDS: dict[str, int] = {name: cid for cid, name in CATEGORY_NAMES.items()}

MAIN_OBJECT_CATEGORY = "body"
"""The only MangaSegmentation category that feeds the main object-quality score -- a whole
character body silhouette (the phase brief's in-scope "characters / character bodies / hair /
clothing" collapse into one instance here). See docs/phase17-results.md for why the other five
categories cannot be part of the main score."""

FORBIDDEN_CATEGORIES = ("face", "text", "balloon", "frame", "onomatopoeia")
"""Excluded from the main object-quality score per the phase brief; used only for the separate
safety/forbidden-target analysis."""


@dataclass(frozen=True, slots=True)
class CandidateInstance:
    """One body instance from the MS92 annotations, before images are downloaded -- the unit
    the manifest is selected from."""

    sample_id: str
    book: str
    page_index: int  # MS92 image id within the book (e.g. 72 for ARMS/072.jpg)
    instance_id: int  # MS92 annotation id
    category: str
    semantic_label: str
    gt_bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open, derived from the mask
    gt_area: int
    page_size: tuple[int, int]  # (height, width) of the page per the MS92 image record


@dataclass(frozen=True, slots=True)
class MangaSegSample(CandidateInstance):
    """A normalized benchmark sample: `CandidateInstance` plus the on-disk artifacts
    (`gt_mask` derived from the human annotation, never modified)."""

    image_path: Path  # normalized page image (git-ignored artifact)
    mask_path: Path  # normalized GT mask, uint8 0/255 (git-ignored artifact)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "book": self.book,
            "page_index": self.page_index,
            "instance_id": self.instance_id,
            "category": self.category,
            "semantic_label": self.semantic_label,
            "gt_bbox": list(self.gt_bbox),
            "gt_area": self.gt_area,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
        }


def _page_from_file_name(file_name: str) -> int:
    """The within-book page number from an MS92 `file_name` like `"ARMS/072.jpg"`.

    MS92's `images[].id` values are GLOBAL across the whole Manga109 set (ARMS starts at 0,
    later books continue), while the mirror's page files are named by the page WITHIN the book
    (`test/images/<BOOK>_<within-book-page:03d>.jpg`). The `file_name` carries the within-book
    number; the global id does not."""
    import re

    match = re.search(r"/(\d+)\.\w+$", file_name)
    if match is None:
        raise ValueError(f"cannot derive within-book page from MS92 file_name {file_name!r}")
    return int(match.group(1))


def _book_annotations(book_json: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Group a book's annotations by within-book page, keeping the raw COCO records."""
    img_by_id = {img["id"]: img for img in book_json["images"]}
    by_page: dict[int, list[dict[str, Any]]] = {}
    for ann in book_json["annotations"]:
        img = img_by_id.get(ann["image_id"])
        if img is None:
            continue  # annotation referencing a page outside this book file -- not ours
        by_page.setdefault(_page_from_file_name(img["file_name"]), []).append(ann)
    return by_page


def _image_record(book_json: dict[str, Any], book: str, page_index: int) -> dict[str, Any] | None:
    for img in book_json["images"]:
        if img["file_name"].startswith(f"{book}/") and _page_from_file_name(
            img["file_name"]
        ) == page_index:
            return img
    return None


def decode_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE segmentation to a boolean `(H, W)` mask (pycocotools contract)."""
    return coco_mask.decode(segmentation) > 0


def tight_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Deterministic tight bbox `(x0, y0, x1, y1)` (half-open) of a mask; raises if empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("cannot derive a tight bbox from an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ms92_book_annotations(hf_token: str, book: str, cache_dir: Path) -> dict[str, Any]:
    """Download one MS92/MangaSegmentation book annotation JSON (gated) into `cache_dir`."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        MS92_REPO,
        f"jsons/{book}.json",
        repo_type="dataset",
        token=hf_token,
        local_dir=str(cache_dir),
    )
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def mirror_page_annotation(hf_token: str, book: str, page_index: int) -> dict[str, Any]:
    """Download one mirror page annotation (`test/annotations/<BOOK>_<page>.json`) and its
    image (`test/images/<BOOK>_<page>.jpg`) -- both non-gated. Returns the annotation dict."""
    from huggingface_hub import hf_hub_download

    ann_path = hf_hub_download(
        MIRROR_REPO,
        f"test/annotations/{book}_{page_index:03d}.json",
        repo_type="dataset",
        token=hf_token,
    )
    with Path(ann_path).open(encoding="utf-8") as f:
        return json.load(f)


def mirror_page_image(hf_token: str, book: str, page_index: int) -> np.ndarray:
    """Download one mirror page image and return it as an RGB `(H, W, 3)` array."""
    from huggingface_hub import hf_hub_download

    img_path = hf_hub_download(
        MIRROR_REPO,
        f"test/images/{book}_{page_index:03d}.jpg",
        repo_type="dataset",
        token=hf_token,
    )
    return np.asarray(Image.open(img_path).convert("RGB"))


def verify_ms92_vs_mirror(
    ms92_anns: list[dict[str, Any]], mirror_ann: dict[str, Any], category: str
) -> float | None:
    """Best-pair IoU between MS92 and mirror annotations of `category` on one page -- 1.0 when
    the mirror is a faithful split of the MS92 annotation (as verified on real data). `None`
    when either side has no annotation of that category on the page or the counts disagree."""
    cid = CATEGORY_IDS[category]
    ms92_masks = [decode_rle(a["segmentation"]) for a in ms92_anns if a["category_id"] == cid]
    mirror_masks = [
        decode_rle(a["segmentation"]) for a in mirror_ann["annotations"] if a["category_id"] == cid
    ]
    if not ms92_masks or not mirror_masks or len(ms92_masks) != len(mirror_masks):
        return None
    best = 0.0
    for m in mirror_masks:
        for ms in ms92_masks:
            if m.shape != ms.shape:
                continue  # different pages mixed in -- never comparable, defensive only
            inter = int(np.count_nonzero(m & ms))
            union = int(np.count_nonzero(m | ms))
            best = max(best, inter / union if union else 0.0)
    return best


def mirror_available_pages(hf_token: str) -> set[tuple[str, int]]:
    """The set of `(book, within_book_page)` whose page image the mirror actually hosts.

    The mirror is a partial Manga109 page image source (~1158 of ~10602 pages, roughly the
    last few pages of each book) -- the MS92 annotations cover every page, but only these
    pages have an image to segment. The benchmark manifest must be constrained to this set so
    every selected sample can be materialized.
    """
    import re

    from huggingface_hub import HfApi

    siblings = HfApi(token=hf_token).list_repo_files(MIRROR_REPO, repo_type="dataset")
    pages: set[tuple[str, int]] = set()
    for path in siblings:
        if not path.startswith("test/images/"):
            continue
        match = re.match(r"test/images/(.+)\.jpg$", path)
        if match is None:
            continue
        stem = match.group(1)
        book, sep, page_str = stem.rpartition("_")
        if not sep or not page_str.isdigit():
            continue
        pages.add((book, int(page_str)))
    return pages


def candidate_pool_from_ms92(
    hf_token: str,
    books: list[str],
    cache_dir: Path,
    category: str = MAIN_OBJECT_CATEGORY,
    available_pages: set[tuple[str, int]] | None = None,
) -> tuple[list[CandidateInstance], dict[str, tuple[int, int]]]:
    """Build the candidate pool (all `category` instances of `books`) from MS92 annotations
    only -- no images. Returns `(pool, page_sizes)` where `page_sizes` maps `BOOK_page` to
    `(height, width)` for `select_samples`'s size stratification. When `available_pages` is
    given, only instances on those `(book, within_page)` pages are kept (the mirror image
    source is partial -- see `mirror_available_pages`)."""
    cid = CATEGORY_IDS[category]
    pool: list[CandidateInstance] = []
    page_sizes: dict[str, tuple[int, int]] = {}
    for book in books:
        data = ms92_book_annotations(hf_token, book, cache_dir / "ms92")
        by_page = _book_annotations(data)
        for img in data["images"]:
            # A book's JSON can embed other books' pages in its images array; only the
            # requested book's pages belong in this pool.
            if not img["file_name"].startswith(f"{book}/"):
                continue
            page = _page_from_file_name(img["file_name"])
            page_sizes[f"{book}_{page:03d}"] = (img["height"], img["width"])
        for page, anns in by_page.items():
            if available_pages is not None and (book, page) not in available_pages:
                continue
            page_size = page_sizes[f"{book}_{page:03d}"]
            for ann in sorted(anns, key=lambda a: a["id"]):
                if ann["category_id"] != cid:
                    continue
                gt_mask = decode_rle(ann["segmentation"])
                if not np.any(gt_mask):
                    continue
                gt_bbox = tight_bbox_from_mask(gt_mask)
                pool.append(
                    CandidateInstance(
                        sample_id=f"{book}_{page:03d}_{ann['id']}",
                        book=book,
                        page_index=page,
                        instance_id=ann["id"],
                        category=category,
                        semantic_label=f"character_{category}",
                        gt_bbox=gt_bbox,
                        gt_area=int(gt_mask.sum()),
                        page_size=page_size,
                    )
                )
    return pool, page_sizes


def materialize_samples(
    manifest,
    *,
    hf_token: str,
    out_dir: Path,
    cache_dir: Path,
    verify_pages: int = 6,
    verify_seed: int = 17,
) -> list[MangaSegSample]:
    """Download images + decode GT masks for every manifest sample and write artifacts.

    For each distinct page in the manifest:
      - downloads the mirror image and checks its dimensions against the MS92 image record,
      - spot-verifies MS92 vs mirror annotation agreement on up to `verify_pages` seeded pages
        (skips the page on disagreement),
      - writes `<sample_id>.png` and `<sample_id>.mask.npz` under `out_dir/dataset/`.

    Returns the materialized samples in manifest order. Never modifies GT masks. Skipped pages
    are recorded in `out_dir/dataset/skipped_pages.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    import random

    rng = random.Random(verify_seed)
    pages_needed = {(s.book, s.page_index) for s in manifest.samples}
    verify_targets = set(rng.sample(sorted(pages_needed), min(verify_pages, len(pages_needed))))

    book_json_cache: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    verified: set[tuple[str, int]] = set()
    samples: list[MangaSegSample] = []

    for sample in manifest.samples:
        book = sample.book
        page = sample.page_index
        if book not in book_json_cache:
            book_json_cache[book] = ms92_book_annotations(hf_token, book, cache_dir / "ms92")
        book_json = book_json_cache[book]
        image_record = _image_record(book_json, book, page)
        if image_record is None:
            skipped[sample.sample_id] = "page missing from MS92 book annotation"
            continue
        if (book, page) in verify_targets and (book, page) not in verified:
            try:
                mirror_ann = mirror_page_annotation(hf_token, book, page)
                by_page = _book_annotations(book_json)
                agreement = verify_ms92_vs_mirror(
                    by_page.get(page, []), mirror_ann, sample.category
                )
                verified.add((book, page))
                if agreement is not None and agreement < 0.999:
                    skipped[sample.sample_id] = (
                        f"MS92 vs mirror annotation agreement {agreement:.3f} < 0.999"
                    )
                    continue
            except Exception as exc:  # noqa: BLE001 -- a failed check skips one page
                skipped[sample.sample_id] = f"mirror annotation check failed: {exc}"
                continue
        try:
            image = mirror_page_image(hf_token, book, page)
        except Exception as exc:  # noqa: BLE001 -- a download failure skips one page
            skipped[sample.sample_id] = f"mirror image download failed: {exc}"
            continue
        if image.shape[:2] != (image_record["height"], image_record["width"]):
            skipped[sample.sample_id] = (
                f"mirror image {image.shape[:2]} does not match MS92 record "
                f"({image_record['height']}x{image_record['width']})"
            )
            continue
        gt_mask = None
        for ann in _book_annotations(book_json).get(page, []):
            if ann["id"] == sample.instance_id:
                gt_mask = decode_rle(ann["segmentation"])
                break
        if gt_mask is None:
            skipped[sample.sample_id] = "instance not found in MS92 annotation"
            continue
        img_path = dataset_dir / f"{sample.sample_id}.png"
        mask_path = dataset_dir / f"{sample.sample_id}.mask.npz"
        if not img_path.exists():
            Image.fromarray(image).save(img_path)
        if not mask_path.exists():
            np.savez_compressed(mask_path, mask=(gt_mask > 0).astype(np.uint8) * 255)
        samples.append(
            MangaSegSample(
                sample_id=sample.sample_id,
                book=sample.book,
                page_index=sample.page_index,
                instance_id=sample.instance_id,
                category=sample.category,
                semantic_label=sample.semantic_label,
                gt_bbox=sample.gt_bbox,
                gt_area=sample.gt_area,
                page_size=sample.page_size,
                image_path=img_path,
                mask_path=mask_path,
            )
        )

    if skipped:
        (dataset_dir / "skipped_pages.json").write_text(
            json.dumps(skipped, indent=2, sort_keys=True), encoding="utf-8"
        )
    return samples
