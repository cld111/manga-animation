"""Parsing OMG-LLaVA output text into `[SEG]` mask associations and status flags.

OMG-LLaVA emits a `[SEG]` token in its generated text wherever it produces a pixel mask; the
masks come back in the same order as the `[SEG]` occurrences. This module is pure text logic
(no model imports) so the association between generated text and mask indices is independently
testable, and so the failure taxonomy's "no mask / no target" signals are deterministic.

The decoded output contains the full sequence (prompt prefix + assistant response). The
assistant response starts after the internlm2_chat `<|im_start|>assistant\\n` marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEG_TOKEN = "[SEG]"
_ASSISTANT_MARKER = "<|im_start|>assistant\n"
_END_MARKER = "<|im_end|>"
_STRIP_PATTERNS = (
    r"<\|im_end\|>",
    r"<s>",
    r"</s>",
    r"<\|im_start\|>user.*?<\|im_start\|>assistant",
)

# Refusal / no-target phrasing the model may emit when it cannot identify the requested element.
_NO_TARGET_PHRASES = (
    "cannot",
    "can't",
    "unable",
    "does not exist",
    "not found",
    "no character",
    "no person",
    "no obvious",
    "no suitable",
    "cannot find",
    "unable to find",
    "there is no",
    "do not see",
    "don't see",
    "does not appear",
    "no such",
)


@dataclass(frozen=True, slots=True)
class SegSpan:
    """One `[SEG]` occurrence in the assistant response: its order index and the surrounding
    text (the description the model attached to the mask)."""

    index: int  # 0-based mask order
    before: str  # response text immediately before this [SEG]
    after: str  # response text immediately after this [SEG]
    context: str  # the trimmed sentence-like context around the [SEG]

    def as_dict(self) -> dict[str, str | int]:
        return {"index": self.index, "before": self.before, "after": self.after,
                "context": self.context}


def strip_markers(text: str) -> str:
    """Remove special/eos markers from a decoded response while keeping the text readable."""
    out = text
    for pattern in _STRIP_PATTERNS:
        out = re.sub(pattern, "", out)
    return out.strip()


def assistant_response(full_text: str) -> str:
    """Isolate the assistant's response from a full decoded sequence.

    The response is everything after the last `<|im_start|>assistant\\n` marker, trimmed at the
    first `<|im_end|>`; if no marker is found the whole text is returned (defensive -- the
    decode should always contain the template markers).
    """
    idx = full_text.rfind(_ASSISTANT_MARKER)
    if idx == -1:
        return strip_markers(full_text)
    body = full_text[idx + len(_ASSISTANT_MARKER) :]
    end = body.find(_END_MARKER)
    if end != -1:
        body = body[:end]
    return strip_markers(body)


def seg_spans(text: str) -> list[SegSpan]:
    """All `[SEG]` occurrences in order, with their surrounding text.

    The `context` is the trimmed window of the response the model associated with the mask --
    the text on both sides of the marker, collapsed. This is what the autonomous gallery shows
    as "target description" for each mask.
    """
    response = assistant_response(text)
    parts = response.split(SEG_TOKEN)
    if len(parts) < 2:
        return []
    spans: list[SegSpan] = []
    for i in range(len(parts) - 1):
        before = parts[i].strip()
        after = parts[i + 1].strip()
        before_trimmed = _trim_sentence(before)
        context = before_trimmed or after
        spans.append(SegSpan(index=i, before=before, after=after, context=context))
    return spans


def _trim_sentence(text: str) -> str:
    """Trim leading connective words/symbols so the target description reads naturally."""
    stripped = text.lstrip(":.,;- \t\n")
    while True:
        match = re.match(r"^(so|and|the|for|of|that|which)[\s,:]+", stripped, re.IGNORECASE)
        if not match:
            break
        stripped = stripped[match.end():]
    return stripped


def count_seg(text: str) -> int:
    """Number of `[SEG]` tokens in the output (0 = the model emitted no mask)."""
    return len(seg_spans(text))


def has_seg(text: str) -> bool:
    return SEG_TOKEN in assistant_response(text)


def no_target_text(text: str) -> bool:
    """Heuristic: did the model's response indicate it could not identify the target (no mask
    expected / target not found)? Used only to split taxonomy categories E vs F deterministically.
    """
    response = assistant_response(text).lower()
    return any(phrase in response for phrase in _NO_TARGET_PHRASES)


def target_context(text: str) -> str | None:
    """The description the model attached to the FIRST `[SEG]` (the primary target in both the
    controlled and autonomous modes), or None when no mask was emitted."""
    spans = seg_spans(text)
    if not spans:
        return None
    return spans[0].context or None
