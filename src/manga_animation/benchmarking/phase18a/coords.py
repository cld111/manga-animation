"""Coordinate contract for Phase 18.2A: Qwen2.5-VL direct bbox localization.

This module exists for the phase brief's "Coordinate handling" requirement: pin down *in what
coordinate space the VLM returns a bbox, what normalization is used, how the bbox is converted
back to source-page pixels, and whether a resize/preprocessing mismatch can be ruled out* --
with unit tests for the conversion, never a silent coordinate mismatch.

Measured contract (established empirically on a real GPU smoke run, not assumed):

- `prompt.py` asks Qwen2.5-VL for the box in PIXEL coordinates of the full source page
  (top-left origin), and states the exact image width/height in the prompt.
- **Qwen2.5-VL reports coordinates in the pixel space of the ORIGINAL input image, NOT in a
  normalized 0..1000 scale.** Evidence (3-sample smoke run): on 1654x1170 pages the model
  returned coordinates with values > 1000 (e.g. x=1254, y=1174 ~= page height 1170), which is
  only possible if the numbers are source-image pixels. The original prompt asked for 0..1000
  and was ignored. A 0..1000-relative assumption would have silently shrunk every box; the
  source-pixel contract matches what the model actually does.
- The page is passed at SOURCE resolution (Qwen's processor bounds the vision tokens
  internally via `max_pixels`, so no manual resize is needed and the coordinate reference
  stays unambiguous).
- Conversion is therefore IDENTITY up to a small edge-tolerance clamp: coordinates are already
  source pixels; values up to 5% beyond the page edge (the model overshooting the page border
  by a few pixels is a normal spatial-estimate artifact, e.g. y=1174 vs H=1170) are clamped
  into `[0, dim]`; anything further out, non-integral, or degenerate is flagged as a
  conversion failure -- never silently scaled or swapped.

Everything here is pure/deterministic and independently unit-tested
(`tests/test_phase18a_coords.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

#: Fraction of a page dimension the model may overshoot and still be treated as a real
#: source-pixel box (clamped into bounds) rather than a coordinate-mismatch failure.
EDGE_TOLERANCE_FRACTION = 0.05

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open pixel box


def extract_json_object(text: str) -> str:
    """Best-effort recovery of a single JSON object from a VLM's raw text.

    Same lenient strategy as `validation.validate._extract_json_object`: strip markdown fences,
    then scan for a balanced `{...}`. Raises `ValueError` when no parseable object exists.
    """
    stripped = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found in Qwen localization output")

    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                json.loads(candidate)  # raises json.JSONDecodeError if still malformed
                return candidate

    raise ValueError("no balanced JSON object found in Qwen localization output")


def _as_int(value: Any) -> int | None:
    """Coerce a numeric JSON value to `int`; `None` for anything non-integral."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_direct_response(raw_text: str) -> dict[str, Any] | None:
    """Parse the model's raw text into its JSON dict. `None` when unparseable."""
    try:
        return json.loads(extract_json_object(raw_text))
    except (ValueError, json.JSONDecodeError):
        return None


def bbox_from_response(parsed: dict[str, Any]) -> tuple[bool, BBox | None, str | None]:
    """Extract `(found, box, error)` from a parsed response.

    `error` is set only for responses that claim `found: true` but cannot provide a usable
    box -- a genuinely `found: false` response is a normal outcome with no error. Returned
    coordinates are validated numerically but NOT yet checked against the page dimensions
    (that is `clamp_box`, kept separate so each check is independently testable).
    """
    if not isinstance(parsed.get("found"), bool):
        return False, None, "response has no boolean 'found' field"

    found = parsed["found"]
    if not found:
        bbox = parsed.get("bbox")
        if bbox is not None:
            return False, None, "response says found=false but carries a non-null bbox"
        return False, None, None

    bbox = parsed.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return True, None, "found=true but 'bbox' is not a 4-element list"

    coords: list[int] = []
    for value in bbox:
        integer = _as_int(value)
        if integer is None:
            return True, None, f"bbox contains a non-integral coordinate: {value!r}"
        coords.append(integer)
    return True, (coords[0], coords[1], coords[2], coords[3]), None


def _in_bounds_with_tolerance(box: BBox, width: int, height: int) -> bool:
    """All four coords within `[0, dim * (1 + EDGE_TOLERANCE_FRACTION)]` -- i.e. they look
    like source-image pixels (possibly overshooting the page edge by a few px)."""
    x0, y0, x1, y1 = box
    x_lo, x_hi = 0, round(width * (1 + EDGE_TOLERANCE_FRACTION))
    y_lo, y_hi = 0, round(height * (1 + EDGE_TOLERANCE_FRACTION))
    return x_lo <= x0 and x_lo <= x1 and x1 <= x_hi and y_lo <= y0 and y1 <= y_hi


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    """Clamp a source-pixel box into `[0, width] x [0, height]` and reject degeneracy.

    Raises `ValueError` if the box is non-degenerate only because of negative coordinates
    (i.e. clamping would collapse it) or if clamping leaves an empty box -- those are
    conversion failures, never silently fabricated.
    """
    x0 = max(0, min(width, box[0]))
    x1 = max(0, min(width, box[2]))
    y0 = max(0, min(height, box[1]))
    y1 = max(0, min(height, box[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"bbox degenerates after clamping ({box}) on {width}x{height} page"
        )
    return (x0, y0, x1, y1)


@dataclass(frozen=True, slots=True)
class QwenBboxPrediction:
    """The fully-converted result of one Qwen direct-localization call.

    `found` reflects the model's own answer. `pixel_box` is set only when `found` AND the
    conversion succeeded; `error` (category 9) is set when the response was unparseable,
    malformed, or outside the source-pixel convention. `convention_ok` records whether the raw
    coordinates looked like source-image pixels (`_in_bounds_with_tolerance`) -- the explicit
    "no silent coordinate mismatch" evidence the brief requires.
    """

    sample_id: str
    found: bool
    box_raw: BBox | None
    pixel_box: BBox | None
    raw_text: str
    error: str | None
    convention_ok: bool
    width: int
    height: int
    clamped: bool = False

    @property
    def usable(self) -> bool:
        """A pixel box usable for IoU/SAM: found, converted, and convention-clean."""
        return self.found and self.pixel_box is not None and self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "found": self.found,
            "box_raw": list(self.box_raw) if self.box_raw is not None else None,
            "pixel_box": list(self.pixel_box) if self.pixel_box is not None else None,
            "raw_text": self.raw_text,
            "error": self.error,
            "convention_ok": self.convention_ok,
            "clamped": self.clamped,
            "width": self.width,
            "height": self.height,
            "usable": self.usable,
        }


def convert_prediction(
    sample_id: str, raw_text: str, width: int, height: int
) -> QwenBboxPrediction:
    """Full parse -> validate -> clamp pipeline for one raw VLM response. Never raises."""
    parsed = parse_direct_response(raw_text)
    if parsed is None:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=False,
            box_raw=None,
            pixel_box=None,
            raw_text=raw_text,
            error="unparseable response",
            convention_ok=False,
            width=width,
            height=height,
        )

    found, box, error = bbox_from_response(parsed)
    if error is not None:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=found,
            box_raw=box,
            pixel_box=None,
            raw_text=raw_text,
            error=error,
            convention_ok=box is not None and _in_bounds_with_tolerance(box, width, height),
            width=width,
            height=height,
        )
    if not found:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=False,
            box_raw=None,
            pixel_box=None,
            raw_text=raw_text,
            error=None,
            convention_ok=False,
            width=width,
            height=height,
        )

    assert box is not None  # found=true path always carries a candidate box
    if not _in_bounds_with_tolerance(box, width, height):
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=True,
            box_raw=box,
            pixel_box=None,
            raw_text=raw_text,
            error=(
                f"coordinates outside the source-pixel convention on a "
                f"{width}x{height} page: {box}"
            ),
            convention_ok=False,
            width=width,
            height=height,
        )

    clamped = any(
        c < 0 or c > limit
        for c, limit in ((box[0], width), (box[1], height), (box[2], width), (box[3], height))
    )
    try:
        pixel_box = clamp_box(box, width, height)
    except ValueError as exc:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=True,
            box_raw=box,
            pixel_box=None,
            raw_text=raw_text,
            error=str(exc),
            convention_ok=True,
            width=width,
            height=height,
        )

    return QwenBboxPrediction(
        sample_id=sample_id,
        found=True,
        box_raw=box,
        pixel_box=pixel_box,
        raw_text=raw_text,
        error=None,
        convention_ok=True,
        width=width,
        height=height,
        clamped=clamped,
    )
