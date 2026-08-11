"""Fetch the specific real manga page used as Phase 3.1's end-to-end vertical-slice input.

Both pages benchmarked in Phase 2 (`docs/phase2-benchmark-results.md`) came back all-STATIC
from the VLM — a known, documented gap: neither page had an actual drawn motion cue (see
ADR 0005's "Open questions"). Phase 3.1 needs a page where a real motion cue is visually
present, so the VLM has a genuine chance to assign `PRIMARY` instead of defaulting everything
to `STATIC` for lack of anything to react to.

This page was chosen by browsing multiple full-color, action-tagged MangaDex series (via the
same "Full Color tag, interior pages only, never covers" policy as `fetch_sample_pages.py`)
and visually reviewing candidates for an unambiguous motion cue, per project policy of not
inventing/assuming one. Selected: *The Skeleton Soldier Failed to Defend the Dungeon*
(MangaDex id `d993f789-e7e5-4832-92fd-37614220b427`), chapter 25 (id
`2cb6c639-033e-4064-a3fd-dacbe0aaaaad`), page 13 of 15 — an arena panel with a knight holding
a raised sword, vertical speed-line rain effects, and hanging cloth banners on both arena
walls (a concrete, segmentable "cloth on a pole"-type object, matching the kind of object the
project's own schema example uses).

Hardcoded (not searched) so this exact page is reproducible on demand — see the "reproducible"
Phase 3.1 acceptance criterion. Not committed to git (see .gitignore's `examples/*.png` rule):
copyrighted third-party content, re-fetchable on demand like every other sample page.

Usage: uv run python scripts/fetch_phase3_sample_page.py [--out examples/phase3_action_page.png]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.mangadex.org"
USER_AGENT = "manga-animation-project/0.1 (research/dev sample fetch)"

CHAPTER_ID = "2cb6c639-033e-4064-a3fd-dacbe0aaaaad"
PAGE_INDEX = 12  # 0-indexed -> page 13 of 15


def _get(path: str, params: dict[str, object] | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch(out_path: Path) -> None:
    at_home = _get(f"/at-home/server/{CHAPTER_ID}")
    base_url, chapter_hash = at_home["baseUrl"], at_home["chapter"]["hash"]
    filenames = at_home["chapter"]["data"]
    filename = filenames[PAGE_INDEX]

    url = f"{base_url}/data/{chapter_hash}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    print(f"{out_path} <- 'The Skeleton Soldier Failed to Defend the Dungeon' ch.25 p.13 ({url})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("examples/phase3_action_page.png"))
    args = parser.parse_args()
    fetch(args.out)
