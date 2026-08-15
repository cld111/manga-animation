"""Phase 19 prompt generation tests.

Covers the official OMG-LLaVA referring form, the four controlled conditions with their
provenance labels, the production-available description extraction, and the oracle-free
autonomous instruction (must not contain the [SEG] special token or any GT-derived data).
"""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase19.prompts import (
    AUTONOMOUS_INSTRUCTION,
    GT_DERIVED,
    PRODUCTION_AVAILABLE,
    autonomous_prompt,
    category_expression,
    condition_provenance,
    controlled_prompt,
    production_expression,
    referring_prompt,
    spatial_expression,
    spatial_side,
)


def test_referring_prompt_uses_official_phrasing():
    assert referring_prompt("the character on the right") == (
        "Can you please segment the character on the right in the given image"
    )


def test_production_expression_strips_period_and_underscores():
    assert production_expression("character_body") == "character body"
    assert production_expression("flag_cloth") == "flag cloth"


def test_condition_a_category_only_diagnostic():
    p = controlled_prompt("A", semantic_label="character_body")
    assert p.condition == "A"
    assert p.expression == "character"
    assert p.prompt == referring_prompt("character")
    assert p.provenance == GT_DERIVED  # the bare category name is not produced text


def test_condition_b_spatial_gt_derived():
    p = controlled_prompt(
        "B", semantic_label="character_body", gt_bbox=(100, 100, 200, 300), page_size=(1000, 800)
    )
    assert p.provenance == GT_DERIVED
    assert "segment the character in the upper part" in p.prompt


def test_condition_c_semantic_spatial_gt_derived():
    p = controlled_prompt(
        "C", semantic_label="character_body", gt_bbox=(600, 400, 700, 500), page_size=(1000, 800)
    )
    assert p.provenance == GT_DERIVED
    assert "character on the right" in p.prompt


def test_condition_d_is_the_production_available_primary():
    p = controlled_prompt("D", semantic_label="character_body")
    assert p.provenance == PRODUCTION_AVAILABLE
    assert p.expression == "character body"
    assert p.prompt == referring_prompt("character body")


def test_condition_b_c_require_geometry():
    with pytest.raises(ValueError):
        controlled_prompt("B", semantic_label="character_body")
    with pytest.raises(ValueError):
        controlled_prompt("C", semantic_label="character_body")


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        controlled_prompt("X", semantic_label="character_body")


def test_condition_provenance():
    assert condition_provenance("D") == PRODUCTION_AVAILABLE
    for cond in ("A", "B", "C"):
        assert condition_provenance(cond) == GT_DERIVED


def test_spatial_side_deterministic():
    page = (1000, 800)
    assert spatial_side((600, 400, 700, 500), page) == "on the right"
    assert spatial_side((100, 100, 200, 300), page) == "in the upper part"
    assert spatial_side((400, 100, 500, 200), page) == "in the upper part"
    assert spatial_side((400, 600, 500, 700), page) == "in the lower part"


def test_spatial_expression():
    assert spatial_expression("character", (600, 400, 700, 500), (1000, 800)) == (
        "the character on the right"
    )


def test_category_expression():
    assert category_expression() == "character"


def test_autonomous_instruction_is_generic_and_oracle_free():
    text = autonomous_prompt()
    assert "[SEG]" not in text  # [SEG] is a special token -- must not be in the prompt
    assert text == AUTONOMOUS_INSTRUCTION
    for banned in ("bbox", "box", "mask.npz", "sample_id", "character_body", "gt_"):
        assert banned not in text
