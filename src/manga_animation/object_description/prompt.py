"""Prompt construction and image/bbox preparation for the per-candidate VLM call.

The coordinate contract (the task brief's critical requirement):

- The VLM receives ONE image: the full pipeline image (in panel mode, the panel's scene crop)
  -- never a crop of the candidate region, never a bbox visualization.
- The candidate is communicated as PIXEL COORDINATES in the exact pixel space of that image.
- Because the VLM client may resize the image before inference (`PipelineConfig.resolution`),
  the image this module hands to the client and the coordinates it states in the prompt must
  be in the SAME space. The Qwen2.5-VL image processor (transformers 5.x) additionally rounds
  each side UP to the nearest multiple of the 28px patch grid (verified on the real worker:
  1024x1536 -> 1036x1540, 720x5062 -> 5068x728). `prepare_image_and_bbox` therefore applies
  the exact same two-step geometry itself -- long-edge downscale, then 28-multiple rounding --
  and scales the bbox by precisely the same factors, so the coordinates stated in the prompt
  always match the image the model actually sees. `tests/test_object_description.py` checks
  this contract independently of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from manga_animation.pipeline.types import BBoxPx

# Qwen2.5-VL vision patch grid: patch_size=14, spatial_merge_size=2 -> 28px cells. The
# processor rounds image sides to multiples of this (verified empirically on the Phase 18.3
# worker with transformers 5.0.0).
_PATCH_GRID = 28

_BOX_START = "<|box_start|>"
_BOX_END = "<|box_end|>"


@dataclass(frozen=True, slots=True)
class PreparedVlmInput:
    """The image that will actually be sent to the VLM, plus the bbox in ITS pixel space."""

    image: Image.Image  # full pipeline image, resized for the VLM (never a crop)
    bbox_px: BBoxPx  # candidate bbox in `image`'s pixel coordinates
    scale_x: float  # image.width / original width (original = the caller's input image)
    scale_y: float  # image.height / original height
    resized_from: tuple[int, int]  # (original width, original height)


def _round_to_grid(value: int, factor: int = _PATCH_GRID) -> int:
    return round(value / factor) * factor


def _resize_to_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    """Downscale (never upscale) so the long edge is at most `max_long_edge` -- the same
    convention `analysis.plan_builder._resized_for_vlm` established for the analysis stage."""
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def prepare_image_and_bbox(
    image: Image.Image,
    bbox_px: BBoxPx,
    *,
    max_long_edge: int,
) -> PreparedVlmInput:
    """Prepare `image` and `bbox_px` so that the stated bbox coordinates exactly match the
    returned image's pixel space (see the module docstring for why the geometry is replicated).

    The caller's `bbox_px` is relative to the caller's `image`; the returned
    `PreparedVlmInput.bbox_px` is relative to the returned `PreparedVlmInput.image`.
    """
    original = image.size
    resized = _resize_to_long_edge(image, max_long_edge)
    rw, rh = resized.size
    final_w, final_h = _round_to_grid(rw), _round_to_grid(rh)
    final = resized if (final_w == rw and final_h == rh) else resized.resize(
        (final_w, final_h), Image.Resampling.LANCZOS
    )

    scale_x = final_w / original[0]
    scale_y = final_h / original[1]

    def _scaled(coord: int, scale: float) -> int:
        return round(coord * scale)

    return PreparedVlmInput(
        image=final,
        bbox_px=BBoxPx(
            x0=_scaled(bbox_px.x0, scale_x),
            y0=_scaled(bbox_px.y0, scale_y),
            x1=_scaled(bbox_px.x1, scale_x),
            y1=_scaled(bbox_px.y1, scale_y),
        ),
        scale_x=scale_x,
        scale_y=scale_y,
        resized_from=original,
    )


def _format_box(bbox_px: BBoxPx) -> str:
    return (
        f"{_BOX_START}({bbox_px.x0},{bbox_px.y0}),({bbox_px.x1},{bbox_px.y1}){_BOX_END}"
    )


_PROMPT_TEMPLATE = """You are evaluating ONE proposed animation candidate on a manga/comic \
page for a deterministic animation pipeline. The artwork is the truth; you must judge the \
candidate precisely, using the context AROUND the proposed region -- never in isolation.

THE IMAGE: the image above is exactly {width}x{height} pixels (width x height). Pixel \
coordinates are measured from the TOP-LEFT corner: x increases to the right, y increases \
downward. The image is the full page panel -- it is NOT a crop of the candidate.

THE CANDIDATE REGION: the axis-aligned bounding box with top-left corner (x0,y0) and \
bottom-right corner (x1,y1) in the pixel coordinates of the image above:
{box_tokens}
Plain numbers: x0={x0}, y0={y0}, x1={x1}, y1={y1}.

Locate this box in the full image and examine what is inside it AND what surrounds it: \
adjacent objects, other characters, weapons or props near the target, speech bubbles, text, \
panel borders, and occlusions. The intended animation target is: "{semantic_label}". \
Grounding models often propose a box that is technically a detection but a bad animation \
candidate; your job is to catch that.

STEP 1 - assess the candidate region itself. Answer exactly one of:
- "pass": the box contains exactly ONE coherent instance of the intended object, well \
represented as a single object candidate (a character alone, a single flag, one weapon).
- "ambiguous": the box contains SEVERAL objects, or several visually similar instances, or \
the intended target is unclear (e.g. a character standing next to a weapon, a crowd).
- "partial": the box captures only PART of the object (a limb, a fragment, a cut-off figure).
- "reject": the box is mostly background, or does not contain a coherent object at all.
- "not_animatable": the box does contain an identifiable object, but animating it is not \
safe (rigid scene element, text-like content, heavy occlusion, or its motion would collide \
with other objects/background).

STEP 2 - describe the object actually inside the box and its animation potential, considering \
the surrounding context. Identify what may move and what must stay absolutely still, the \
fitted motion category, its direction, relative amplitude, speed, and the constraints that \
must not be violated (e.g. "keep the face static", "do not move the speech bubble", "motion \
must not cross the panel border"). Report any potential problems with neighboring objects, \
background or overlaps in "neighbor_conflicts".

Answer with ONLY ONE JSON object, no prose, no markdown fences, in exactly this shape:
{{"bbox_assessment": "pass" | "ambiguous" | "partial" | "reject" | "not_animatable", \
"object_identity": "short snake_case name of the object actually inside the box, or null", \
"matches_semantic_label": true or false, "animatable": true or false, "movable_parts": \
["short labels"], "static_parts": ["short labels"], "motion_kind": null or one of \
"sway"|"flow"|"drift"|"rotate"|"pulse"|"breathe"|"flicker", "direction": null or one of \
"up"|"down"|"left"|"right"|"up_left"|"up_right"|"down_left"|"down_right", "amplitude_band": \
"subtle"|"moderate"|"pronounced", "speed_band": "slow"|"normal"|"fast", "pivot_hint": \
"top"|"center"|"bottom", "constraints": ["must-not-violate rules"], "neighbor_conflicts": \
["problems with neighbors/background/occlusion"], "confidence": a float 0-1, "reason": "one \
short sentence grounded in what you actually see"}}

Rules: "motion_kind" is required iff "animatable" is true; "direction" is required iff \
"motion_kind" is "drift" (the direction of the steady movement); otherwise both are null. \
"confidence" must reflect genuine uncertainty -- be conservative with low confidence and \
write the doubts into "neighbor_conflicts" or "reason". Text, speech bubbles, lettering, \
rigid background and panel borders must never be animated: if the box is such content, \
assess "not_animatable". If the box contains several objects or the target is ambiguous, \
"bbox_assessment" must be "ambiguous", never "pass". """

# Unique marker the test fake clients use to dispatch this stage's prompt (same convention as
# `_VALIDATION_PROMPT_MARKER`/`_MASK_SEMANTICS_PROMPT_MARKER` in tests/test_pipeline.py).
PROMPT_MARKER = "evaluating ONE proposed animation candidate"


def build_prompt(
    *,
    prepared: PreparedVlmInput,
    semantic_label: str,
) -> str:
    """Build the prompt for one candidate. Coordinates are those of `prepared.bbox_px`,
    which `prepare_image_and_bbox` guarantees to be in `prepared.image`'s pixel space."""
    bbox = prepared.bbox_px
    return _PROMPT_TEMPLATE.format(
        width=prepared.image.width,
        height=prepared.image.height,
        box_tokens=_format_box(bbox),
        x0=bbox.x0,
        y0=bbox.y0,
        x1=bbox.x1,
        y1=bbox.y1,
        semantic_label=semantic_label.replace("_", " "),
    )
