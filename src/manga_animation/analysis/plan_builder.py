"""Turns a VLM's raw text output into a schema-valid `AnimationPlan`.

This is the "VLM -> structured Animation Plan" step from `docs/pipeline.md`. Per
`.claude/agents/vision-agent.md`, semantic decisions (STATIC vs. PRIMARY/SECONDARY/MICRO,
`semantic_label`, `confidence`) are this stage's call; per `.claude/agents/animation-agent.md`,
picking concrete `MotionSpec` numbers is normally a separate specialist's job. Phase 3.1's
vertical slice is deliberately constrained to exactly one animated object, so this module
folds a small, documented heuristic (see `_MOTION_HEURISTICS` below, informed by the
`animation-planning` skill) into plan construction rather than standing up a second stage for
a single object's parameters — a real multi-object animation-agent stage is future work, not
something this module should quietly grow into.

Nothing here imports `torch`/`transformers` — see `client.py` for why.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import (
    AnimationPlan,
    BBox,
    Easing,
    LoopSpec,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PanelPlan,
    PivotSpec,
    SourceImage,
    TransformKind,
    Vector2,
)

logger = get_logger(__name__)

_PANEL_ID = "panel_1"

ANALYSIS_PROMPT = """You are analyzing a single manga/comic page image to build a structured \
Animation Plan.

For every distinct object or character element on the page that could plausibly be animated \
(e.g. hair, cloth, a weapon, a banner/flag, an eye, a hand, an object in motion), output one \
JSON entry with these exact fields:
- "semantic_label": short lowercase snake_case name, e.g. "character_hair", "flag_banner", \
"raised_sword"
- "motion_type": one of "static", "primary", "secondary", "micro" (lowercase exactly)
  - "static": no visually justified reason to move
  - "primary": this object carries the page's action -- the one thing a reader's eye should \
read as moving on purpose
  - "secondary": motion that follows from a primary mover (e.g. cloth attached to a moving \
character)
  - "micro": small motion that adds life without carrying narrative weight (a blink, subtle \
sway)
- "confidence": a float 0-1, your confidence in this STATIC/ANIMATED decision
- "reason": one sentence grounded in what's actually drawn -- not speculation
- "motion_description": (only if motion_type is not "static") one short sentence describing \
the physical motion in plain terms, e.g. "banner sways left and right in the wind"

A visually justified reason for motion can come from ANY of these, not only deformation \
drawn on the object itself:
1. Deformation/distortion drawn directly on the object (wavy linework, a bent/curved shape).
2. Motion/speed lines drawn on or immediately touching the object.
3. Panel-level or page-level effect lines (speed lines, impact bursts, radiating focus \
lines) layered over the scene near the object, even if the object's own outline is drawn \
clean -- a raised weapon inside a field of speed lines is still visually justified motion \
for that weapon, even though the lines aren't drawn ON the blade itself.
4. The object's drawn pose/position implying it is mid-action (a sword raised for a swing, \
an outstretched arm, a foot lifted mid-stride) -- an implied trajectory is a real, drawable \
cue, not speculation about what happens next.
5. An implied physical force that would plausibly act on the object given the scene (wind \
blowing through the panel, an impact that would shake attached cloth).

Only mark an object "primary"/"secondary"/"micro" if you can point to at least one of these \
five categories actually present in the drawing. If nothing on the page has such a cue for \
any object, mark everything "static" -- that is a valid, correct answer, not a failure. But \
do not default to "static" just because the cue isn't drawn on the object's own outline -- \
check the whole panel around each candidate object, not only the object's own silhouette, \
before deciding.

Return ONLY a JSON array of these objects, no prose, no markdown code fences."""

_RECOVERY_PROMPT_TEMPLATE = """Your previous response could not be parsed/validated as the \
required JSON array. Error: {error}

Return ONLY a corrected JSON array of objects with fields "semantic_label", "motion_type" \
(one of "static"/"primary"/"secondary"/"micro"), "confidence" (0-1), "reason", and \
"motion_description" (only if motion_type is not "static"). No prose, no markdown fences."""


class _RawObjectDecision(BaseModel):
    """The VLM's raw semantic read for one object -- not yet an `ObjectPlan`.

    Deliberately separate from `ObjectPlan`: the VLM doesn't (and per the schema's design,
    shouldn't) supply `object_id`/`panel_id`/pixel-free-but-still-structural fields like
    `parent_id` -- those are assigned when building the real `ObjectPlan`, not asked of the
    model.
    """

    semantic_label: str = Field(min_length=1)
    motion_type: MotionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    motion_description: str | None = None


_RawObjectList = TypeAdapter(list[_RawObjectDecision])


def _extract_json_array(text: str) -> str:
    """Best-effort recovery of a JSON array from a VLM's raw text response.

    VLMs frequently wrap valid JSON in markdown fences or a sentence of prose despite being
    asked not to -- this is deliberately still "the model's actual output," not a fabricated
    plan, so it is tried before falling back to the recovery re-prompt.
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

    start = stripped.find("[")
    if start == -1:
        raise ValueError("no JSON array found in VLM output")

    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "[":
            depth += 1
        elif stripped[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                json.loads(candidate)  # raises json.JSONDecodeError if still malformed
                return candidate

    raise ValueError("no balanced JSON array found in VLM output")


def _parse_and_validate(raw_text: str) -> list[_RawObjectDecision]:
    """Raises `ValueError`/`json.JSONDecodeError`/`pydantic.ValidationError` on failure --

    callers decide whether/how to recover; this function never invents a fallback.
    """
    array_text = _extract_json_array(raw_text)
    data = json.loads(array_text)
    return _RawObjectList.validate_python(data)


def _decisions_from_vlm(
    client: VLMClient, image: Image.Image, image_ref: str
) -> list[_RawObjectDecision]:
    """Parse + validate the VLM's structured output, with exactly one recovery attempt.

    Per the Phase 3.1 brief: "If the VLM produces invalid JSON: (1) attempt the project's
    defined structured-output/recovery mechanism; (2) validate again; (3) if still invalid,
    fail explicitly." This is that mechanism.
    """
    raw_text = client.generate(image, ANALYSIS_PROMPT)
    try:
        return _parse_and_validate(raw_text)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_error:
        logger.info(
            "analysis stage: initial VLM output failed to parse/validate (%s); "
            "attempting one recovery pass",
            first_error,
        )
        recovery_prompt = _RECOVERY_PROMPT_TEMPLATE.format(error=first_error)
        recovery_text = client.generate(image, recovery_prompt)
        try:
            return _parse_and_validate(recovery_text)
        except (json.JSONDecodeError, ValueError, ValidationError) as second_error:
            raise PipelineStageError(
                stage="analysis",
                input_ref=image_ref,
                detail=(
                    "VLM output remained invalid JSON / schema-invalid after one recovery "
                    f"attempt: {second_error}"
                ),
                root_cause=f"first attempt: {first_error}; after recovery: {second_error}",
                architectural=False,
                proposed_fix=(
                    "improve prompt engineering, or add constrained/structured-output "
                    "decoding on the VLM call rather than free-text JSON"
                ),
            ) from second_error


# Keyword -> (transform_kind, amplitude, easing, pivot, direction_hint) heuristic, per the
# animation-planning skill's transform-kind table. This is a deliberately small, documented
# stand-in for a full animation-agent parameter-filling stage -- Phase 3.1 animates exactly
# one object, so a keyword lookup is proportionate; a multi-object plan would need a real
# animation-agent pass instead of growing this table further.
_MOTION_HEURISTICS: list[tuple[tuple[str, ...], MotionSpec]] = [
    (
        ("flag", "banner", "cloth", "cape", "cloak", "drape", "curtain"),
        MotionSpec(
            transform_kind=TransformKind.MESH_WARP,
            amplitude=0.12,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    ),
    (
        ("hair",),
        MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.03,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    ),
    (
        ("sword", "blade", "weapon", "hammer", "spear", "arm", "hand"),
        MotionSpec(
            transform_kind=TransformKind.ROTATE,
            amplitude=8.0,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=1.0, reference="object_bbox"),
        ),
    ),
    (
        ("eye",),
        MotionSpec(
            transform_kind=TransformKind.OPACITY,
            amplitude=0.3,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
        ),
    ),
]

_DEFAULT_MOTION = MotionSpec(
    transform_kind=TransformKind.TRANSLATE,
    direction=Vector2(x=1.0, y=0.0),
    amplitude=0.02,
    speed=1.0,
    easing=Easing.EASE_IN_OUT,
    pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
)


def _motion_spec_for(decision: _RawObjectDecision) -> MotionSpec:
    haystack = f"{decision.semantic_label} {decision.motion_description or ''}".lower()
    for keywords, template in _MOTION_HEURISTICS:
        if any(kw in haystack for kw in keywords):
            return template.model_copy(deep=True)
    return _DEFAULT_MOTION.model_copy(deep=True)


def _slugify(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"obj_{slug or 'object'}_{index}"


# motion_type -> sort priority. Strictly dominates confidence in `_rank_candidates`'s sort key,
# so an existing "primary" decision always outranks any "secondary"/"micro" one, regardless of
# confidence -- this preserves Phase 3.1's original "highest-confidence primary wins" behavior
# whenever a real primary exists (see tests/test_analysis.py); it only changes the outcome when
# NO primary exists at all.
_MOTION_TYPE_RANK: dict[MotionType, int] = {
    MotionType.PRIMARY: 2,
    MotionType.SECONDARY: 1,
    MotionType.MICRO: 0,
}


def _rank_candidates(
    decisions: list[_RawObjectDecision], image_ref: str
) -> list[_RawObjectDecision]:
    """Rank every non-STATIC decision best-to-worst as a candidate for the plan's single
    animated object -- the "ranked animation candidates" analysis representation Phase 3.2
    adds (see the Phase 3.2 brief's VLM investigation section).

    Phase 3.1's original selector (`_select_single_primary`) only ever considered decisions
    the VLM explicitly labeled "primary", discarding "secondary"/"micro" reads entirely when
    deciding whether a plan was usable at all -- a real page where the VLM saw *some*
    motion-worthy object but wasn't confident enough to call it "primary" would incorrectly
    report "no PRIMARY object" even though it had real, usable signal. Ranking by
    `(motion_type priority, confidence)` instead of requiring a literal "primary" label fixes
    that without changing what gets picked whenever a real primary exists.

    Still raises `PipelineStageError` when every decision is genuinely STATIC -- see "Static
    Is a Valid Result" in docs/architecture.md; this ranking does not loosen that case, only
    the case where non-STATIC signal existed but wasn't labeled "primary".
    """
    candidates = [d for d in decisions if d.motion_type != MotionType.STATIC]
    if not candidates:
        raise PipelineStageError(
            stage="analysis",
            input_ref=image_ref,
            detail=(
                "VLM marked every object STATIC -- an all-STATIC read is a valid model "
                "output but an unusable one for this task (the pipeline requires one real "
                "animated object)"
            ),
            root_cause=(
                "either no drawn motion cue was present on the page, or the model failed to "
                "recognize one that was (see ADR 0005's documented all-STATIC gap)"
            ),
            architectural=False,
            proposed_fix=(
                "use a page with an unambiguous drawn motion cue, or fall back to a clearly "
                "labeled test fixture rather than fabricating a plan"
            ),
        )

    candidates.sort(key=lambda d: (_MOTION_TYPE_RANK[d.motion_type], d.confidence), reverse=True)
    primaries = [d for d in candidates if d.motion_type == MotionType.PRIMARY]
    if len(primaries) > 1:
        logger.info(
            "analysis stage: VLM proposed %d PRIMARY objects; keeping highest-confidence "
            "'%s' (%.2f), deferring the rest to STATIC per the single-object pipeline scope",
            len(primaries),
            candidates[0].semantic_label,
            candidates[0].confidence,
        )
    elif not primaries:
        logger.info(
            "analysis stage: VLM proposed no PRIMARY object but %d SECONDARY/MICRO "
            "candidate(s); using the highest-ranked one ('%s', %s, confidence=%.2f) as the "
            "plan's single animated object",
            len(candidates),
            candidates[0].semantic_label,
            candidates[0].motion_type.value,
            candidates[0].confidence,
        )
    return candidates


def _checksum(image_path: Path) -> str:
    return "sha256:" + hashlib.sha256(image_path.read_bytes()).hexdigest()


def build_plan(
    decisions: list[_RawObjectDecision],
    image: Image.Image,
    image_path: Path,
    config: PipelineConfig,
) -> AnimationPlan:
    """Assemble the final schema-valid `AnimationPlan` from validated VLM decisions.

    Every object other than the chosen single PRIMARY is forced to STATIC with no motion
    spec, even if the VLM proposed SECONDARY/MICRO motion for it -- Phase 3.1's pipeline only
    grounds/segments/animates the one PRIMARY object (see `.claude/agents/segmentation-agent.md`
    task scope for this phase), so leaving other objects marked as animated in the plan would
    make the plan lie about what the rendered video will actually contain.
    """
    ranked = _rank_candidates(decisions, str(image_path))
    chosen = ranked[0]
    rest = [d for d in decisions if d is not chosen]

    objects: list[ObjectPlan] = []
    for index, decision in enumerate(rest):
        objects.append(
            ObjectPlan(
                object_id=_slugify(decision.semantic_label, index),
                panel_id=_PANEL_ID,
                semantic_label=decision.semantic_label,
                confidence=decision.confidence,
                motion_type=MotionType.STATIC,
                motion=None,
            )
        )

    primary_object_id = _slugify(chosen.semantic_label, len(rest))
    try:
        objects.append(
            ObjectPlan(
                object_id=primary_object_id,
                panel_id=_PANEL_ID,
                semantic_label=chosen.semantic_label,
                confidence=chosen.confidence,
                motion_type=MotionType.PRIMARY,
                motion=_motion_spec_for(chosen),
            )
        )
    except ValueError as exc:
        # A schema-cross-validation failure here (e.g. the seamless-loop integer-speed rule)
        # is OUR heuristic's bug, not the VLM's -- flag it as architectural/internal.
        raise PipelineStageError(
            stage="analysis",
            input_ref=str(image_path),
            detail=f"internal MotionSpec heuristic produced a schema-invalid object: {exc}",
            root_cause="the built-in transform_kind/amplitude/speed heuristic table",
            architectural=True,
            proposed_fix="fix the offending entry in _MOTION_HEURISTICS/_DEFAULT_MOTION",
        ) from exc

    width, height = image.size
    source = SourceImage(
        path=str(image_path), width=width, height=height, checksum=_checksum(image_path)
    )
    panel = PanelPlan(panel_id=_PANEL_ID, bbox=BBox(x=0.0, y=0.0, width=1.0, height=1.0))
    loop = LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True)

    return AnimationPlan(source=source, panels=[panel], objects=objects, loop=loop)


def _resized_for_vlm(image: Image.Image, max_long_edge: int) -> Image.Image:
    """Downscale (never upscale) so the VLM sees `config.resolution`'s long edge, not the raw

    source pixels. Real finding (Phase 3.1 first Kaggle run): a tall manga page (720x5062, a
    ~7:1 aspect ratio, far taller than anything ADR 0005's benchmarking passes used) fed at
    full resolution produced enough vision tokens to OOM a single T4 during `.generate()`
    even under `device_map="auto"` sharding — `PipelineConfig.resolution` exists exactly to
    bound this (see docs/architecture.md's "GPU Awareness") but `analyze_page` wasn't applying
    it. The `AnimationPlan` this produces is unaffected by the resize: every spatial field is
    normalized to [0, 1] (see docs/animation-plan-schema.md), and `SourceImage.width/height`
    below are still the true source dimensions, not this resized copy's.
    """
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def analyze_page(image_path: Path, client: VLMClient, *, config: PipelineConfig) -> AnimationPlan:
    """Public entry point: real manga page -> validated `AnimationPlan`.

    Raises `PipelineStageError` (never silently invents a plan) if the VLM's output can't be
    turned into a usable, schema-valid plan with exactly one PRIMARY object.
    """
    image = Image.open(image_path).convert("RGB")
    vlm_image = _resized_for_vlm(image, config.resolution)
    decisions = _decisions_from_vlm(client, vlm_image, str(image_path))
    return build_plan(decisions, image, image_path, config)
