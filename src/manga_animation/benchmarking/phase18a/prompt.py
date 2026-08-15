"""Phase 18.2A prompt: ask Qwen2.5-VL to localize ONE specific target instance on a full page.

The target description is whatever the production pipeline actually has (the manifest's
grounding prompt text, e.g. `"character body."`), deliberately NOT a GT-derived description and
NOT a unique per-instance label -- the benchmark measures whether the VLM can resolve the
*specific* instance from production-available semantic information alone, exactly the
information DINO received in Phase 17/18.1.

Output format is the `coords.py` contract: a single JSON object with the box in SOURCE-PIXEL
coordinates (the measured convention Qwen2.5-VL actually reports in -- see `coords.py`), and
`found: false` + `bbox: null` when the model is not sure. The image width/height are stated
explicitly in the prompt so the pixel reference is unambiguous. The prompt is deliberately
strict (no prose, no fences) and the response is parsed leniently by
`coords.convert_prediction` regardless.
"""

from __future__ import annotations

_DIRECT_LOCALIZATION_PROMPT_TEMPLATE = """You are looking at one full page of a manga comic. \
The animation pipeline wants to animate a specific character instance on this page.

The image above is exactly {width} pixels wide and {height} pixels tall.

Target object: "{target_description}"

Look carefully at the whole page. Give the tight bounding box of the ONE specific instance of \
this target (the full figure, head to feet). If there are several similar objects on the page, \
pick the specific instance this target refers to.

If the target instance is not present or not clearly visible on this page, answer with \
"found": false.

Coordinate convention: the box coordinates are PIXEL coordinates in the image above, top-left \
origin (x from 0 to {width}, y from 0 to {height}). The box is [left, top, right, bottom].

Answer with ONLY one JSON object, no prose, no markdown fences, in exactly this shape:
{{"found": true or false, "bbox": [x1, y1, x2, y2]}}
When "found" is false, set "bbox" to null."""


def build_direct_prompt(target_description: str, width: int, height: int) -> str:
    """The full localization prompt for one target, with the page's pixel reference stated."""
    return _DIRECT_LOCALIZATION_PROMPT_TEMPLATE.format(
        target_description=target_description, width=width, height=height
    )
