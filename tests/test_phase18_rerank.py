"""Phase 18.2 reranking-logic tests: VLM scoring via the production prompt/parser, ranking
strategies A/B/C, and selection-accuracy helpers. Uses a fake VLM client (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from manga_animation.benchmarking.phase18.rerank import (
    VlmCandidateScore,
    production_object_plan,
    rank_of_best_correct,
    rank_scores,
    selected_is_correct,
    vlm_score_candidate,
)

GT = (10, 10, 30, 30)


class FakeVLM:
    """Canned production-format responses (`{"matches": bool, "confidence": float,
    "reason": str}`), returned in call order -- deterministic, no crop introspection."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return "{}"

    def unload(self) -> None:
        pass


def _crop_with_marker(marker: int, size: int = 40) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[0, 0, 0] = marker
    return img


def _score(matches, confidence, box, dino=0.5):
    return VlmCandidateScore(
        box=box, dino_score=dino, matches=matches, confidence=confidence, reason="r"
    )


def test_vlm_score_candidate_uses_production_prompt_and_parser():
    client = FakeVLM(['{"matches": true, "confidence": 0.9, "reason": "yes"}'])
    image = _crop_with_marker(1)
    plan = production_object_plan()
    score = vlm_score_candidate(client, image, plan, (10, 10, 30, 30))
    assert score.matches is True
    assert score.confidence == pytest.approx(0.9)
    # The production verification prompt must be the one used.
    assert "character body" in client.prompts[0]


def test_vlm_score_candidate_fail_closed_on_unparseable():
    client = FakeVLM(["not json at all"])
    score = vlm_score_candidate(
        client, _crop_with_marker(1), production_object_plan(), (0, 0, 10, 10)
    )
    assert score.matches is None and score.confidence is None


def test_strategy_a_ranks_matches_then_confidence():
    scores = [
        _score(False, 0.9, (100, 100, 120, 120)),
        _score(True, 0.5, (10, 10, 30, 30)),
        _score(True, 0.8, (50, 50, 70, 70)),
        _score(None, None, (200, 200, 220, 220)),
    ]
    ranked = rank_scores(scores, "A")
    expected = [(50, 50, 70, 70), (10, 10, 30, 30), (100, 100, 120, 120), (200, 200, 220, 220)]
    assert [c.box for c in ranked] == expected


def test_strategy_b_blends_dino_score_within_matches_group():
    scores = [
        _score(True, 0.4, (10, 10, 30, 30), dino=0.9),  # blend 1.3
        _score(True, 0.7, (50, 50, 70, 70), dino=0.1),  # blend 0.8
        _score(False, 0.9, (100, 100, 120, 120), dino=0.9),
    ]
    ranked = rank_scores(scores, "B")
    assert ranked[0].box == (10, 10, 30, 30)  # higher blend wins within matches=True
    assert ranked[1].box == (50, 50, 70, 70)
    assert ranked[2].box == (100, 100, 120, 120)


def test_strategy_c_filters_implausible_then_ranks():
    # A candidate covering >90% of the page fails the production plausibility gate and is
    # dropped before ranking.
    scores = [
        _score(True, 0.9, (10, 10, 30, 30)),
        _score(True, 0.8, (0, 0, 39, 39)),  # 39x39 of 40x40 page -> 95% -> implausible
    ]
    ranked = rank_scores(scores, "C", image_shape=(40, 40))
    assert [c.box for c in ranked] == [(10, 10, 30, 30)]


def test_selected_is_correct_and_rank_of_best():
    # A high-confidence WRONG candidate is rank 1 after reranking; the correct candidate at
    # rank 2 (both matches=True, so confidence decides).
    scores = [
        _score(True, 0.9, (100, 100, 120, 120)),
        _score(True, 0.6, (10, 10, 30, 30)),
    ]
    ranked = rank_scores(scores, "A")
    assert not selected_is_correct(GT, ranked)
    assert rank_of_best_correct(GT, ranked) == 2
    ranked2 = rank_scores([_score(True, 0.6, (10, 10, 30, 30))], "A")
    assert selected_is_correct(GT, ranked2)
    assert rank_of_best_correct(GT, ranked2) == 1


def test_rank_scores_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        rank_scores([], "Z")


def test_rank_scores_strategy_c_without_image_shape_returns_empty():
    # Without page geometry the geometry gate cannot run -> no candidate survives it.
    assert rank_scores([], "C") == []
    assert rank_scores([_score(True, 0.9, (10, 10, 30, 30))], "C") == []
