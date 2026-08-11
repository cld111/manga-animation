"""Fetch sample manga pages from the MangaDex API for local benchmarking/QA use.

Per project policy: only colored, interior pages (never covers/posters). MangaDex's "Full
Color" tag identifies colored manga; individual page images from a chapter's `/at-home/server`
response are always interior pages (cover art is a separate relationship MangaDex never mixes
into chapter page data), so no separate cover-filtering is needed once a full-color manga is
picked.

Downloaded images are not committed to git (see .gitignore) — copyrighted third-party content
stays local-only, re-fetchable on demand. This is a plain stdlib script deliberately: fetching
a handful of sample pages doesn't need a new project dependency.

Usage: uv run python scripts/fetch_sample_pages.py [--count N] [--out-dir examples]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.mangadex.org"
USER_AGENT = "manga-animation-project/0.1 (research/dev sample fetch)"
FULL_COLOR_TAG_NAME = "Full Color"


def _get(path: str, params: dict[str, object] | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _full_color_tag_id() -> str:
    for tag in _get("/manga/tag")["data"]:
        if tag["attributes"]["name"].get("en") == FULL_COLOR_TAG_NAME:
            return tag["id"]
    raise RuntimeError(f"MangaDex tag '{FULL_COLOR_TAG_NAME}' not found")


def _pick_chapter_with_pages(tag_id: str) -> tuple[str, dict]:
    """First full-color manga (by follower count) with an English chapter that has pages."""
    manga_list = _get(
        "/manga",
        {
            "includedTags[]": [tag_id],
            "contentRating[]": ["safe"],
            "order[followedCount]": "desc",
            "limit": 10,
            "availableTranslatedLanguage[]": ["en"],
        },
    )["data"]

    for manga in manga_list:
        chapters = _get(
            "/chapter",
            {
                "manga": manga["id"],
                "translatedLanguage[]": ["en"],
                "order[chapter]": "asc",
                "limit": 5,
            },
        )["data"]
        for chapter in chapters:
            if chapter["attributes"]["pages"] > 0:
                title = manga["attributes"]["title"].get("en") or next(
                    iter(manga["attributes"]["title"].values())
                )
                return title, chapter
    raise RuntimeError("no full-color manga with a fetchable English chapter found")


def fetch(count: int, out_dir: Path) -> None:
    tag_id = _full_color_tag_id()
    title, chapter = _pick_chapter_with_pages(tag_id)
    chapter_num = chapter["attributes"]["chapter"]

    at_home = _get(f"/at-home/server/{chapter['id']}")
    base_url, chapter_hash = at_home["baseUrl"], at_home["chapter"]["hash"]
    filenames = at_home["chapter"]["data"]  # full-quality interior pages, never covers

    step = max(1, len(filenames) // count)
    picks = filenames[::step][:count]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, filename in enumerate(picks, start=1):
        url = f"{base_url}/data/{chapter_hash}/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        dest = out_dir / f"sample_page_{i:02d}.png"
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        print(f"{dest} <- {title!r} ch.{chapter_num} ({url})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("examples"))
    args = parser.parse_args()
    fetch(args.count, args.out_dir)
