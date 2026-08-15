"""Coordinate contract for Phase 18.2A: Qwen2.5-VL direct bbox localization.

This module exists for the phase brief's "Coordinate handling" requirement: pin down *in what
coordinate space the VLM returns a bbox, what normalization is used, how the bbox is converted
back to source-page pixels, and whether a resize/preprocessing mismatch can be ruled out* --
with unit tests for the conversion, never a silent coordinate mismatch.

Contract:

- `prompt.py` asks Qwen2.5-VL for the box in the model's native normalized convention:
  integers in `[0, 1000]` per axis, relative to the full image it sees (`0` = left/top edge,
  `1000` = right/bottom edge), top-left corner first, JSON
  `{"found": bool, "bbox": [x1, y1, x2, y2]}`. This is how Qwen2.5-VL is trained to ground.
- The page is resized for the VLM preserving aspect ratio (production-faithful; mirrors
  `plan_builder._resized_for_vlm`). Because the requested normalization is *relative*,
  conversion to source-pixel coordinates is exact regardless of the resize:
  `x_px = round(c * W / 1000)`.
- Every conversion RECORDS the raw model text, the raw 0..1000 values, and a
  `convention_ok` flag (all four coords within `[0, 1000]`, box non-degenerate). A model that
  instead emits raw pixels of the processed image, reverses corners, or emits garbage is
  flagged as a conversion failure -- never silently scaled into a plausible-looking box.

Everything here is pure/deterministic and independently unit-tested
(`tests/test_phase18a_coords.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Qwen2.5-VL's native box coordinate scale: integers in [0, 1000] per axis.
COORD_SCALE = 1000

BBox1000 = tuple[int, int, int, int]
BBoxPx = tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open pixel box


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


def bbox_from_response(parsed: dict[str, Any]) -> tuple[bool, BBox1000 | None, str | None]:
    """Extract `(found, box_1000, error)` from a parsed response.

    `error` is set only for responses that claim `found: true` but cannot provide a usable
    box -- a genuinely `found: false` response is a normal outcome with no error. Returned
    coordinates are validated numerically but NOT yet checked against `[0, COORD_SCALE]` (that
    is `coords_in_scale`, kept separate so each check is independently testable).
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


def coords_in_scale(box: BBox1000) -> bool:
    """All four coordinates within `[0, COORD_SCALE]` and the box non-degenerate
    (`x2 > x1`, `y2 > y1`) -- the convention the prompt requested. `False` on a coordinate-
    mismatch (values look like raw pixels of a resized image, reversed corners, or a zero box)."""
    x0, y0, x1, y1 = box
    return all(0 <= c <= COORD_SCALE for c in box) and x1 > x0 and y1 > y0


def scale_to_pixels(box: BBox1000, width: int, height: int) -> BBoxPx:
    """Convert a `[0, 1000]` box to half-open pixel coordinates for a `width x height` page.

    Deterministic: `round(c * dim / COORD_SCALE)`, clamped to `[0, dim]`. Raises `ValueError`
    if the resulting box is degenerate (sub-pixel or fully clamped) -- the caller records this
    as a conversion failure rather than fabricating a box.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid page size {width}x{height}")

    def axis(lo: int, hi: int, dim: int) -> tuple[int, int]:
        p0 = max(0, min(dim, round(lo * dim / COORD_SCALE)))
        p1 = max(0, min(dim, round(hi * dim / COORD_SCALE)))
        return p0, p1

    x0, x1 = axis(box[0], box[2], width)
    y0, y1 = axis(box[1], box[3], height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"bbox degenerates after pixel scaling ({x0},{y0})-({x1},{y1}) on {width}x{height}"
        )
    return (x0, y0, x1, y1)


@dataclass(frozen=True, slots=True)
class QwenBboxPrediction:
    """The fully-converted result of one Qwen direct-localization call.

    `found` reflects the model's own answer. `pixel_box` is set only when `found` AND the
    conversion succeeded; `error` (category 9) is set when the response was unparseable,
    malformed, or outside the coordinate convention. `convention_ok` records whether the raw
    0..1000 values satisfied `coords_in_scale` -- the explicit "no silent coordinate mismatch"
    evidence the brief requires.
    """

    sample_id: str
    found: bool
    box_1000: BBox1000 | None
    pixel_box: BBoxPx | None
    raw_text: str
    error: str | None
    convention_ok: bool
    width: int
    height: int

    @property
    def usable(self) -> bool:
        """A pixel box usable for IoU/SAM: found, converted, and convention-clean."""
        return self.found and self.pixel_box is not None and self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "found": self.found,
            "box_1000": list(self.box_1000) if self.box_1000 is not None else None,
            "pixel_box": list(self.pixel_box) if self.pixel_box is not None else None,
            "raw_text": self.raw_text,
            "error": self.error,
            "convention_ok": self.convention_ok,
            "width": self.width,
            "height": self.height,
            "usable": self.usable,
        }


def convert_prediction(
    sample_id: str, raw_text: str, width: int, height: int
) -> QwenBboxPrediction:
    """Full parse -> validate -> scale pipeline for one raw VLM response. Never raises."""
    parsed = parse_direct_response(raw_text)
    if parsed is None:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=False,
            box_1000=None,
            pixel_box=None,
            raw_text=raw_text,
            error="unparseable response",
            convention_ok=False,
            width=width,
            height=height,
        )

    found, box_1000, error = bbox_from_response(parsed)
    if error is not None:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=found,
            box_1000=box_1000,
            pixel_box=None,
            raw_text=raw_text,
            error=error,
            convention_ok=box_1000 is not None and coords_in_scale(box_1000),
            width=width,
            height=height,
        )
    if not found:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=False,
            box_1000=None,
            pixel_box=None,
            raw_text=raw_text,
            error=None,
            convention_ok=False,
            width=width,
            height=height,
        )

    assert box_1000 is not None  # found=true path always carries a candidate box
    if not coords_in_scale(box_1000):
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=True,
            box_1000=box_1000,
            pixel_box=None,
            raw_text=raw_text,
            error=f"coordinates outside the 0..{COORD_SCALE} convention: {box_1000}",
            convention_ok=False,
            width=width,
            height=height,
        )

    try:
        pixel_box = scale_to_pixels(box_1000, width, height)
    except ValueError as exc:
        return QwenBboxPrediction(
            sample_id=sample_id,
            found=True,
            box_1000=box_1000,
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
        box_1000=box_1000,
        pixel_box=pixel_box,
        raw_text=raw_text,
        error=None,
        convention_ok=True,
        width=width,
        height=height,
    )
