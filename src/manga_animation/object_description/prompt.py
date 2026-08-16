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

from collections.abc import Sequence
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

STEP 1 - READ THE ACTION. Look at the WHOLE scene and determine what is HAPPENING with the \
objects: what action, event, or physical force involves the area around the box. Examples: \
a character swinging a sword, a character sprinting with hair and cloth streaming behind, \
wind blowing through the panel, an explosion with radiating speed lines, an energy field \
pulsing, rain falling, a flag whipping in a gale, an arm raised to strike. A quiet scene \
where nothing is happening is also a valid read. Base this ONLY on what is actually drawn \
(deformation, speed/motion lines, pose mid-action, implied force) -- never on speculation.

STEP 2 - JUDGE THE CANDIDATE with the action in mind. Answer exactly one of:
- "pass": the box contains EXACTLY ONE coherent instance of the intended object, well \
represented as a single object candidate (a character alone, a single flag, one weapon).
- "ambiguous": the box contains SEVERAL objects or SEVERAL instances -- even two identical \
characters, even a character plus a nearby prop. One box = one object, never a group.
- "partial": the box captures only PART of the object (a limb, a fragment, a cut-off figure).
- "reject": the box is mostly background, or does not contain a coherent object at all.
- "not_animatable": the box does contain an identifiable object, but animating it is not \
safe (rigid scene element, text-like content, heavy occlusion, or its motion would collide \
with other objects/background).

STEP 3 - DESCRIBE THE MOTION THE ACTION GIVES THIS OBJECT. Given the action read in STEP 1, \
what should this specific object do in the animation? Identify what may move and what must \
stay absolutely still, the fitted motion category, its direction, relative amplitude, speed, \
and the constraints that must not be violated (e.g. "keep the face static", "do not move \
the speech bubble", "motion must not cross the panel border"). If STEP 1 found a real \
action but it does not involve THIS object, the object has no motion and is best judged \
not_animatable. Report any potential problems with neighboring objects, background or \
overlaps in "neighbor_conflicts".

CONTEXT: a still manga drawing of a person, weapon, flag, hair or cloth is NORMAL and \
remains perfectly animatable when the scene gives it an action -- subtle life (hair swaying \
in a sprint, cloth moving in wind, a weapon swinging) is the pipeline's whole purpose, and \
it does not require visible motion lines. "animatable": false is reserved for content that \
CANNOT be moved safely: lettering/text, rigid background structures, heavily occluded or \
cut-off regions, or objects the scene's action does not involve.

Answer with ONLY ONE JSON object, no prose, no markdown fences, in exactly this shape:
{{"bbox_assessment": "pass" | "ambiguous" | "partial" | "reject" | "not_animatable", \
"object_identity": "short snake_case name of the object actually inside the box", \
"matches_semantic_label": true or false, "animatable": true or false, "movable_parts": \
["short labels"], "static_parts": ["short labels"], "motion_kind": null or one of \
"sway"|"flow"|"drift"|"rotate"|"pulse"|"breathe"|"flicker", "direction": null or one of \
"up"|"down"|"left"|"right"|"up_left"|"up_right"|"down_left"|"down_right", "amplitude_band": \
"subtle"|"moderate"|"pronounced", "speed_band": "slow"|"normal"|"fast", "pivot_hint": \
"top"|"center"|"bottom", "constraints": ["real must-not-violate rules, or empty list"], \
"neighbor_conflicts": ["real problems with neighbors/background/occlusion, or empty list"], \
"confidence": a float 0-1, "reason": "one short sentence naming the ACTION that drives the \
motion (or the lack of one)"}}

Rules (violating any of these is a wrong answer):
1. "bbox_assessment" must be EXACTLY one of the five values "pass", "ambiguous", "partial", \
"reject", "not_animatable" -- never anything else.
2. "pass" requires EXACTLY ONE instance of the intended object in the box. A box with two or \
more characters, or a character plus a weapon/prop, or several visually similar instances, is \
"ambiguous" -- never "pass". When genuinely uncertain between "pass" and a stricter verdict, \
choose the stricter one: this pipeline prefers a clean rejection over animating the wrong \
region. But do NOT reject an ordinary, well-isolated object just because the drawing is \
stylized or blocky -- "matches_semantic_label" means "the box contains the same KIND of \
object as the label" (a stylized person is still a character), not a literal name match.
3. Lettering is NEVER animatable, no matter what the semantic_label says (even if the label \
literally names it, e.g. "text_banner"): speech bubbles, dialogue, sound effects, captions, \
banners of text -- any box whose content is text-like must be assessed "not_animatable" (or \
"reject" if the box is mostly background).
4. "motion_kind" is required iff "animatable" is true; "direction" is required iff \
"motion_kind" is "drift"; otherwise both are null. "direction" is ONLY for drift -- for sway, \
flow, rotate, pulse, breathe, flicker it is always null (never invent values like "up_down" \
or "left_right"). "amplitude_band", "speed_band" and "pivot_hint" always carry one of their \
listed values -- never null. "object_identity", "movable_parts", "static_parts", \
"constraints", "neighbor_conflicts" are never null and never placeholder text: name what you \
actually see, use empty lists where nothing applies, and make "constraints" list the specific \
rules that follow from THIS box (e.g. which part must stay still, what must not be crossed).
5. "confidence" must reflect genuine uncertainty -- write the doubts into \
"neighbor_conflicts" or "reason". """

# Unique marker the test fake clients use to dispatch this stage's prompt (same convention as
# `_VALIDATION_PROMPT_MARKER`/`_MASK_SEMANTICS_PROMPT_MARKER` in tests/test_pipeline.py).
# Shared by the single-candidate and the batch prompt so fakes can dispatch on one substring.
PROMPT_MARKER = "proposed animation candidate"


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


@dataclass(frozen=True, slots=True)
class PromptCandidate:
    """One candidate box within a batch prompt: its index, the prepared image it lives in
    (multi-panel batches have several images), and its prepared (image-space) bbox."""

    index: int
    image_index: int  # index into the prompt's image list
    semantic_label: str
    bbox_px: BBoxPx
    image_size: tuple[int, int] = (0, 0)  # (width, height) of the prepared image the box lives in


_MULTI_PROMPT_TEMPLATE = """You are evaluating {n_candidates} proposed animation candidates on \
manga/comic page(s) for a deterministic animation pipeline. The artwork is the truth; you must \
judge every candidate precisely, using the context AROUND each proposed region -- never in \
isolation.

THE IMAGES: {n_images} image(s) are shown above. Pixel coordinates are measured from the \
TOP-LEFT corner of each image: x increases to the right, y increases downward. Each image is a \
full page panel -- NOT a crop of a candidate.

THE CANDIDATES: {n_candidates} candidate boxes are listed below. Each box is given in the \
pixel coordinates of the image it belongs to (each entry names its image index), using the \
format [index] <|box_start|>(x0,y0),(x1,y1)<|box_end|>:
{box_list}

Locate every box in its image and examine what is inside each box AND what surrounds it: \
adjacent objects, other characters, weapons or props near the target, speech bubbles, text, \
panel borders, and occlusions. Each candidate carries its intended label; grounding models \
often propose a box that is technically a detection but a bad animation candidate -- your job \
is to catch that, for EVERY candidate independently.

FOR EVERY CANDIDATE, IN ONE PASS, do the following three steps:
STEP 1 - READ THE ACTION. Look at the WHOLE scene and determine what is HAPPENING with the \
objects: what action, event, or physical force involves the area around the box. Examples: a \
character swinging a sword, a character sprinting with hair and cloth streaming behind, wind \
blowing through the panel, an explosion with radiating speed lines, an energy field pulsing, \
rain falling, a flag whipping in a gale, an arm raised to strike. A quiet scene where nothing \
is happening is also a valid read. Base this ONLY on what is actually drawn (deformation, \
speed/motion lines, pose mid-action, implied force) -- never on speculation.
STEP 2 - JUDGE THE CANDIDATE with the action in mind:
- "pass": the box contains EXACTLY ONE coherent instance of the intended object, well \
represented as a single object candidate (a character alone, a single flag, one weapon).
- "ambiguous": the box contains SEVERAL objects or SEVERAL instances -- even two identical \
characters, even a character plus a nearby prop. One box = one object, never a group.
- "partial": the box captures only PART of the object (a limb, a fragment, a cut-off figure).
- "reject": the box is mostly background, or does not contain a coherent object at all.
- "not_animatable": the box does contain an identifiable object, but animating it is not \
safe (rigid scene element, text-like content, heavy occlusion, or its motion would collide \
with other objects/background).
STEP 3 - DESCRIBE THE MOTION THE ACTION GIVES THIS OBJECT. Identify what may move and what \
must stay absolutely still, the fitted motion category, its direction, relative amplitude, \
speed, and the constraints that must not be violated (e.g. "keep the face static", "do not \
move the speech bubble", "motion must not cross the panel border"). If STEP 1 found a real \
action but it does not involve THIS object, the object has no motion and is best judged \
not_animatable. Report any potential problems with neighboring objects, background or \
overlaps in "neighbor_conflicts".

CONTEXT: a still manga drawing of a person, weapon, flag, hair or cloth is NORMAL and \
remains perfectly animatable when the scene gives it an action. "animatable": false is \
reserved for content that CANNOT be moved safely: lettering/text, rigid background \
structures, heavily occluded or cut-off regions, or objects the scene's action does not \
involve.

Answer with ONLY ONE JSON ARRAY, no prose, no markdown fences. The array has EXACTLY \
{n_candidates} objects, one per candidate (in any order), each with "box_index" set to the \
candidate's [index]. Each object has exactly this shape:
{{"box_index": the candidate index, "bbox_assessment": "pass" | "ambiguous" | "partial" | \
"reject" | "not_animatable", "object_identity": "short snake_case name of the object actually \
inside the box", "matches_semantic_label": true or false, "animatable": true or false, \
"movable_parts": ["short labels"], "static_parts": ["short labels"], "motion_kind": null or \
one of "sway"|"flow"|"drift"|"rotate"|"pulse"|"breathe"|"flicker", "direction": null or one \
of "up"|"down"|"left"|"right"|"up_left"|"up_right"|"down_left"|"down_right", \
"amplitude_band": "subtle"|"moderate"|"pronounced", "speed_band": "slow"|"normal"|"fast", \
"pivot_hint": "top"|"center"|"bottom", "constraints": ["real must-not-violate rules, or empty \
list"], "neighbor_conflicts": ["real problems with neighbors/background/occlusion, or empty \
list"], "confidence": a float 0-1, "reason": "one short sentence naming the ACTION that \
drives the motion (or the lack of one)"}}

Rules (violating any of these is a wrong answer):
1. "bbox_assessment" must be EXACTLY one of the five values "pass", "ambiguous", "partial", \
"reject", "not_animatable" -- never anything else.
2. "pass" requires EXACTLY ONE instance of the intended object in the box. A box with two or \
more characters, or a character plus a weapon/prop, or several visually similar instances, is \
"ambiguous" -- never "pass". When genuinely uncertain between "pass" and a stricter verdict, \
choose the stricter one: this pipeline prefers a clean rejection over animating the wrong \
region. "matches_semantic_label" means "the box contains the same KIND of object as the \
label" (a stylized person is still a character), not a literal name match.
3. Lettering is NEVER animatable, no matter what the semantic_label says: speech bubbles, \
dialogue, sound effects, captions, banners of text -- any box whose content is text-like must \
be assessed "not_animatable" (or "reject" if the box is mostly background).
4. "motion_kind" is required iff "animatable" is true; "direction" is required iff \
"motion_kind" is "drift"; otherwise both are null. "direction" is ONLY for drift -- for sway, \
flow, rotate, pulse, breathe, flicker it is always null (never invent values like "up_down" \
or "left_right"). "amplitude_band", "speed_band" and "pivot_hint" always carry one of their \
listed values -- never null. "object_identity", "movable_parts", "static_parts", \
"constraints", "neighbor_conflicts" are never null and never placeholder text. Every candidate \
must appear EXACTLY once -- never omit one, never invent a box_index that was not listed.
5. "confidence" must reflect genuine uncertainty -- write the doubts into \
"neighbor_conflicts" or "reason"."""


def build_multi_prompt(
    candidates: Sequence[PromptCandidate],
) -> str:
    """Build the batch prompt for several candidates across one or several prepared images.

    Each `PromptCandidate` must carry `bbox_px` in its own image's pixel space (the
    `prepare_image_and_bbox` contract), and `image_index` pointing at the image the box
    belongs to within the single multi-image call.
    """
    lines = [
        (
            f"[{c.index}] image {c.image_index} ({c.image_size[0]}x{c.image_size[1]} px) "
            f"{_format_box(c.bbox_px)} x0={c.bbox_px.x0} y0={c.bbox_px.y0} "
            f"x1={c.bbox_px.x1} y1={c.bbox_px.y1} label=\"{c.semantic_label.replace('_', ' ')}\""
        )
        for c in candidates
    ]
    return _MULTI_PROMPT_TEMPLATE.format(
        n_candidates=len(candidates),
        n_images=len({c.image_index for c in candidates}),
        box_list="\n".join(lines),
    )
