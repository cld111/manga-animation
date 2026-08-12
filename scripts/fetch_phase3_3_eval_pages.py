"""Fetch the two additional real pages Phase 3.3's evaluation dataset adds to the three

pages already used by Phase 3.1/3.2 (`sample_page_01.png`, `sample_page_02.png`,
`phase3_action_page.png` -- see `fetch_sample_pages.py` / `fetch_phase3_sample_page.py`).

Those three pages are all one MangaDex series ("The Skeleton Soldier Failed to Defend the
Dungeon") -- Phase 3.2's own "Remaining limitations" flagged cross-series generalization as
untested. These two pages were chosen (by browsing full-color, English-translated MangaDex
series with locally-fetchable chapters and visually reviewing candidate pages, same policy as
`fetch_phase3_sample_page.py`) specifically to add real diversity Phase 3.3's evaluation
dataset needs and did not otherwise have:

- `eval_static_dialogue.png`: *Who Made Me a Princess* (MangaDex id
  `722a45c0-5e55-40f2-929b-ff69b0989edb`), chapter 47 (id
  `5607ccd4-5cc8-4a24-9053-7a0f22cffc46`), page 4 of 17 -- a dialogue-only scene (characters
  talking in a bedroom, no drawn motion lines, no wind/implied force on hair or clothing, no
  mid-action pose). A genuinely STATIC page by visual inspection, and from a different
  genre/series than every previously-used sample (romance/dialogue vs. action).
- `eval_weapon_effects.png`: *Latna Saga: Survival of a Sword King* (MangaDex id
  `478e6926-b8bc-465c-9694-2bae1dbaf32b`), chapter 238 (id
  `98b4f2e6-9703-494f-bd78-7d992aae48ca`), page 26 of 83 -- a clash/impact panel with clear
  radiating motion lines, an energy-burst effect, and a weapon mid-action. A second, real,
  visually distinct weapon/action/effects page from a different series and art style than
  `phase3_action_page.png`.

Hardcoded (not searched) so these exact pages are reproducible on demand, matching
`fetch_phase3_sample_page.py`'s policy. Not committed to git (see `.gitignore`'s
`examples/*.png` rule): copyrighted third-party content, re-fetchable on demand.

Usage: uv run python scripts/fetch_phase3_3_eval_pages.py [--out-dir examples]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.mangadex.org"
USER_AGENT = "manga-animation-project/0.1 (research/dev sample fetch)"

PAGES = [
    {
        "out_name": "eval_static_dialogue.png",
        "chapter_id": "5607ccd4-5cc8-4a24-9053-7a0f22cffc46",
        "page_index": 3,  # 0-indexed -> page 4 of 17
        "citation": "'Who Made Me a Princess' ch.47 p.4",
    },
    {
        "out_name": "eval_weapon_effects.png",
        "chapter_id": "98b4f2e6-9703-494f-bd78-7d992aae48ca",
        "page_index": 25,  # 0-indexed -> page 26 of 83
        "citation": "'Latna Saga: Survival of a Sword King' ch.238 p.26",
    },
]


def _get(path: str, params: dict[str, object] | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        at_home = _get(f"/at-home/server/{page['chapter_id']}")
        base_url, chapter_hash = at_home["baseUrl"], at_home["chapter"]["hash"]
        filename = at_home["chapter"]["data"][page["page_index"]]

        url = f"{base_url}/data/{chapter_hash}/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        out_path = out_dir / page["out_name"]
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        print(f"{out_path} <- {page['citation']} ({url})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("examples"))
    args = parser.parse_args()
    fetch(args.out_dir)
