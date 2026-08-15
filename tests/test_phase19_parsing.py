"""Phase 19 output-parsing tests: assistant-response isolation, [SEG] extraction, mask
association, and the no-target heuristic used to split taxonomy categories E/F."""

from __future__ import annotations

from manga_animation.benchmarking.phase19.parsing import (
    assistant_response,
    count_seg,
    has_seg,
    no_target_text,
    seg_spans,
    strip_markers,
    target_context,
)

_FULL = (
    "<|im_start|>user\n<image>\nCan you please segment the character body in the given image"
    "<|im_end|>\n<|im_start|>assistant\n[SEG]<|im_end|>"
)


def test_assistant_response_isolates_generation():
    assert assistant_response(_FULL) == "[SEG]"


def test_assistant_response_strips_markers():
    assert strip_markers("<|im_end|><s>hello</s>") == "hello"


def test_seg_spans_order_and_context():
    text = (
        "<|im_start|>assistant\nThe character on the right is running. [SEG] The character on "
        "the left is standing. [SEG]<|im_end|>"
    )
    spans = seg_spans(text)
    assert len(spans) == 2
    assert spans[0].index == 0
    assert spans[0].context == "character on the right is running."
    assert spans[1].index == 1
    assert spans[1].context == "character on the left is standing."


def test_no_seg_returns_empty():
    assert seg_spans("<|im_start|>assistant\nI see no character here.<|im_end|>") == []
    assert count_seg("no tokens here") == 0
    assert not has_seg("no tokens here")


def test_count_and_has():
    assert count_seg(_FULL) == 1
    assert has_seg(_FULL)


def test_target_context_first_seg():
    text = (
        "<|im_start|>assistant\nSo, the flowing hair of the character [SEG] and the weapon "
        "[SEG]<|im_end|>"
    )
    assert target_context(text) == "flowing hair of the character"


def test_target_context_none_when_no_mask():
    assert target_context("no mask emitted") is None


def test_no_target_text_refusal_phrases():
    assert no_target_text("<|im_start|>assistant\nI cannot find a character body.<|im_end|>")
    assert no_target_text("<|im_start|>assistant\nThere is no character here.<|im_end|>")
    assert not no_target_text("<|im_start|>assistant\n[SEG]<|im_end|>")
