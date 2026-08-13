"""Fetch the 10 real pages Phase 9's Real-World Evaluation Dataset adds on top of the existing

7-sample golden regression set (`configs/phase3_3_eval_dataset.yaml`) -- see
`docs/decisions/0016-phase9-realworld-evaluation.md` and
`configs/phase9_realworld_eval_dataset.yaml` for the dataset itself.

Deliberately a *different* 10 series/pages from every sample already used by any earlier phase
(all of which are one of three MangaDex series: 'The Skeleton Soldier Failed to Defend the
Dungeon', 'Who Made Me a Princess', 'Latna Saga: Survival of a Sword King') -- chosen by
browsing MangaDex's "Full Color" tag crossed with a spread of genre tags (sports, fantasy
action, mecha/sci-fi, horror, office comedy, slice-of-life, gothic drama, hunter/action) via
the public `/manga` search endpoint, then visually reviewing real candidate pages (same manual
selection policy `fetch_phase3_sample_page.py`/`fetch_phase3_3_eval_pages.py` already
established) for real drawn motion cues per the `manga-analysis` skill's STATIC vs. ANIMATED
checklist -- never guessed or synthesized.

Hardcoded (not searched at fetch time) so these exact pages are reproducible on demand, same
convention as the two existing fetch scripts above. Not committed to git (see `.gitignore`'s
`examples/**/*.png` rule): copyrighted third-party content, re-fetchable on demand.

Usage: uv run python scripts/fetch_phase9_realworld_pages.py [--out-dir examples/realworld]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.mangadex.org"
USER_AGENT = "manga-animation-project/0.1 (research/dev sample fetch)"

# page_index is 0-indexed into the chapter's page list (matches the at-home server's `data`
# array order); citation's "page N of M" is the 1-indexed value a human would cite.
PAGES = [
    {
        "out_name": "wind_breaker_sprint.png",
        "chapter_id": "84a8bccc-591e-4de9-b764-2954fc5b5ad8",
        "page_index": 8,
        "citation": "'Wind Breaker', MangaDex chapter id 84a8bccc-591e-4de9-b764-2954fc5b5ad8 "
        "(ch. 540), page 9 of 32",
    },
    {
        "out_name": "wind_breaker_finish.png",
        "chapter_id": "84a8bccc-591e-4de9-b764-2954fc5b5ad8",
        "page_index": 15,
        "citation": "'Wind Breaker', MangaDex chapter id 84a8bccc-591e-4de9-b764-2954fc5b5ad8 "
        "(ch. 540), page 16 of 32",
    },
    {
        "out_name": "omniscient_reader_blade.png",
        "chapter_id": "d3c06606-4ee7-4454-a3a7-ffdb9ecf5d26",
        "page_index": 30,
        "citation": "'Omniscient Reader's Viewpoint', MangaDex chapter id "
        "d3c06606-4ee7-4454-a3a7-ffdb9ecf5d26 (ch. 255, 'Asmodeus Part 3'), page 31 of 56",
    },
    {
        "out_name": "angels_of_war_fleet.png",
        "chapter_id": "3addf99c-2e3b-4754-addf-3aedae292c86",
        "page_index": 10,
        "citation": "'Tiejī Gāngbīng' ('Angels of War'), MangaDex chapter id "
        "3addf99c-2e3b-4754-addf-3aedae292c86 (ch. 1), page 11 of 24",
    },
    {
        "out_name": "space_monster_creature.png",
        "chapter_id": "3e1fafa2-e831-4337-82d4-ecdccd5e8d6d",
        "page_index": 35,
        "citation": "'Jinhwahaneun Ujugoemul i Doetda' ('I Became an Evolving Space Monster'), "
        "MangaDex chapter id 3e1fafa2-e831-4337-82d4-ecdccd5e8d6d (ch. 1), page 36 of 50",
    },
    {
        "out_name": "space_monster_hypersenses.png",
        "chapter_id": "3e1fafa2-e831-4337-82d4-ecdccd5e8d6d",
        "page_index": 15,
        "citation": "'Jinhwahaneun Ujugoemul i Doetda' ('I Became an Evolving Space Monster'), "
        "MangaDex chapter id 3e1fafa2-e831-4337-82d4-ecdccd5e8d6d (ch. 1), page 16 of 50",
    },
    {
        "out_name": "reality_lie_office.png",
        "chapter_id": "0759f760-102d-4f36-8817-5a3fdd658ab3",
        "page_index": 5,
        "citation": "'Real mo Tama ni wa Uso o Tsuku' ('Sometimes Even Reality Is a Lie!'), "
        "MangaDex chapter id 0759f760-102d-4f36-8817-5a3fdd658ab3 (ch. 1), page 6 of 25",
    },
    {
        "out_name": "marika_love_meter.png",
        "chapter_id": "b905083b-8fb7-4ba8-b2c1-6a02edacaa7d",
        "page_index": 5,
        "citation": "'Marika-chan no Koukando wa Bukkowareteiru' ('Marika's Love Meter "
        "Malfunction'), MangaDex chapter id b905083b-8fb7-4ba8-b2c1-6a02edacaa7d (ch. 1), "
        "page 6 of 26",
    },
    {
        "out_name": "villainess_ending_scuffle.png",
        "chapter_id": "0f01d260-fc5f-4f32-8be0-2b82e5ebecb6",
        "page_index": 10,
        "citation": "'Agyeogui Ending Jugeumppun' ('Death Is the Only Ending for the "
        "Villainess'), MangaDex chapter id 0f01d260-fc5f-4f32-8be0-2b82e5ebecb6 (ch. 64), "
        "page 11 of 18",
    },
    {
        "out_name": "sss_hunter_gladiator.png",
        "chapter_id": "839122d1-4d36-4233-a955-ee9f7ca986cd",
        "page_index": 9,
        "citation": "'SSS-geup Jugeoya Saneun Hunter' ('SSS-Class Suicide Hunter'), MangaDex "
        "chapter id 839122d1-4d36-4233-a955-ee9f7ca986cd (ch. 151), page 10 of 13",
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
    parser.add_argument("--out-dir", type=Path, default=Path("examples/realworld"))
    args = parser.parse_args()
    fetch(args.out_dir)
