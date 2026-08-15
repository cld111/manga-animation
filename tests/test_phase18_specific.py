"""Phase 18.2.1 tests: the instance-specific contrastive prompt (strategy S) -- parsing, the
prompt text, ranking (no DINO score), and selection-accuracy helpers. Uses a fake VLM."""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.benchmarking.phase18.rerank import (
    SPECIFIC_INSTANCE_PROMPT_TEMPLATE,
    SpecificCandidateScore,
    _parse_specific_response,
    rank_of_best_specific,
    rank_specific,
    specific_is_correct,
    specific_score_candidate,
)

GT = (10, 10, 30, 30)


class FakeVLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else "{}"

    def unload(self) -> None:
        pass


def _img():
    return np.zeros((60, 60, 3), dtype=np.uint8)


def _spec(box, is_specific, confidence):
    return SpecificCandidateScore(
        box=box, is_specific=is_specific, confidence=confidence, reason="r"
    )


def test_specific_prompt_is_not_the_production_presence_prompt():
    assert "is_specific" in SPECIFIC_INSTANCE_PROMPT_TEMPLATE
    assert "EXACTLY ONE character body" in SPECIFIC_INSTANCE_PROMPT_TEMPLATE
    assert "does NOT" in SPECIFIC_INSTANCE_PROMPT_TEMPLATE


def test_specific_score_candidate_uses_the_specific_prompt_and_parser():
    client = FakeVLM(['{"is_specific": true, "confidence": 0.8, "reason": "one clean figure"}'])
    score = specific_score_candidate(client, _img(), (10, 10, 30, 30))
    assert score.is_specific is True
    assert score.confidence == pytest.approx(0.8)
    assert "is_specific" in client.prompts[0]
    assert "character body" in client.prompts[0]


def test_specific_parse_fail_closed_on_unparseable():
    assert _parse_specific_response("nope") is None
    assert _parse_specific_response('{"is_specific": "yes"}') is None


def test_specific_parse_tolerates_doubled_braces():
    # Real VLM behavior (Phase 18.2.1): it can echo the prompt's JSON placeholder format.
    raw = '{{\n    "is_specific": false,\n    "confidence": 0.8,\n    "reason": "partial figure"}}'
    parsed = _parse_specific_response(raw)
    assert parsed is not None
    assert parsed.is_specific is False
    assert parsed.confidence == pytest.approx(0.8)


def test_rank_specific_no_dino_score():
    scores = [
        _spec((100, 100, 120, 120), True, 0.9),  # specific, high conf, wrong box
        _spec((10, 10, 30, 30), True, 0.6),  # specific, lower conf, correct box
        _spec((200, 200, 220, 220), False, 0.9),
        _spec((50, 50, 70, 70), None, None),  # unparseable -> last
    ]
    ranked = rank_specific(scores)
    expected = [(100, 100, 120, 120), (10, 10, 30, 30), (200, 200, 220, 220), (50, 50, 70, 70)]
    assert [c.box for c in ranked] == expected
    # Strategy S is semantic-only: DINO score is not an input, so a low-confidence correct
    # specific box cannot be boosted by any external score.


def test_specific_selection_accuracy_helpers():
    ranked = rank_specific(
        [_spec((100, 100, 120, 120), True, 0.9), _spec((10, 10, 30, 30), True, 0.5)]
    )
    assert not specific_is_correct(GT, ranked)  # top specific pick is the wrong box
    assert rank_of_best_specific(GT, ranked) == 2
    assert rank_of_best_specific(GT, rank_specific([_spec((10, 10, 30, 30), True, 0.8)])) == 1


def test_specific_absent_candidate():
    ranked = rank_specific([_spec((200, 200, 220, 220), True, 0.9)])
    assert rank_of_best_specific(GT, ranked) is None
    assert not specific_is_correct(GT, ranked)
