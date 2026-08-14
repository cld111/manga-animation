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

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from manga_animation.analysis.client import VLMClient
from manga_animation.analysis.panels import detect_panels
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import PipelineStageError, bbox_px_to_normalized
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

ALSO, and this is important: treat ALREADY-DRAWN visual effects as first-class animation \
targets, exactly like objects. Speed lines, motion strokes, impact bursts, radiating focus \
lines, energy fields, glow, sparks, smoke, water splashes, and similar drawn effect artwork \
are legitimate candidates: list each one as its own JSON entry with a descriptive \
semantic_label like "speed_lines", "impact_burst", "energy_field", "smoke_cloud", \
"water_splash", "spark_shower", "glow_effect". For an effect, "motion_type" describes how \
it should move: "primary" only if the effect itself is the page's main action; otherwise \
"secondary"/"micro" (e.g. rain is usually "micro"). Give its "motion_description" in \
effect-specific terms that name the motion model, e.g. "speed lines streak outward from the \
impact point", "the energy field pulses and radiates", "smoke drifts upward", "water \
splashes and flows". Do NOT invent an effect that is not actually drawn, and do NOT list \
speech bubbles, dialogue text, or panel borders as effects -- those must stay static.

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
    (
        (
            "impact",
            "burst",
            "explosion",
            "shockwave",
            "energy",
            "glow",
            "pulse",
            "aura",
            "radiat",
            "flash",
            "shock wave",
        ),
        MotionSpec(
            transform_kind=TransformKind.RADIAL_EXPAND,
            amplitude=0.08,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        ),
    ),
    (
        ("smoke", "steam"),
        MotionSpec(
            transform_kind=TransformKind.MESH_WARP,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        ),
    ),
    (
        ("water", "splash", "fluid", "liquid", "wave", "spray"),
        MotionSpec(
            transform_kind=TransformKind.MESH_WARP,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        ),
    ),
    (
        ("rain",),
        MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=0.0, y=1.0),
            amplitude=0.03,
            speed=2.0,
            easing=Easing.EASE_IN_OUT,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        ),
    ),
    (
        ("spark", "spark_shower", "debris", "particle", "shard"),
        MotionSpec(
            transform_kind=TransformKind.OPACITY,
            amplitude=0.35,
            speed=2.0,
            easing=Easing.EASE_IN_OUT,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
        ),
    ),
    (
        ("speed_line", "speed line", "motion_line", "motion line", "streak", "slash", "flow line"),
        MotionSpec(
            transform_kind=TransformKind.MESH_WARP,
            amplitude=0.12,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.5, reference="object_bbox"),
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
    decisions: list[_RawObjectDecision], image_ref: str, *, allow_all_static: bool = False
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
    if not candidates and allow_all_static:
        return []
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


def _non_primary_object_plan(
    decision: _RawObjectDecision, index: int, panel_id: str, image_path: Path
) -> ObjectPlan:
    """Build the `ObjectPlan` for one decision that isn't this plan's chosen PRIMARY.

    Phase 4 (see `docs/decisions/0010-multi-object-layer-decomposition.md`): a decision the VLM
    itself marked SECONDARY or MICRO keeps that real motion_type and gets a real `MotionSpec` --
    the pipeline can now animate more than one object per page (`pipeline/orchestrator.py` loops
    grounding/validation/segmentation/animation over every non-STATIC `ObjectPlan`, not just the
    PRIMARY). Before this phase, EVERY non-chosen decision was forced to STATIC regardless of
    its own label -- a deliberate, documented Phase 3.1-3.3.x scope limit (see
    docs/phase3.2-results.md's "kept, by design" note), not a bug being fixed here.

    Two cases still fall back to STATIC, unchanged from before: a decision the VLM itself
    labeled STATIC (obviously), and a decision labeled PRIMARY that lost to a
    higher-confidence PRIMARY (`_rank_candidates`'s existing "keeping highest-confidence...
    deferring the rest to STATIC" policy for that specific edge case) -- this function does not
    invent a new policy for demoting an unchosen PRIMARY to SECONDARY; it only stops
    overriding an already-real SECONDARY/MICRO read.
    """
    if decision.motion_type in (MotionType.SECONDARY, MotionType.MICRO):
        try:
            return ObjectPlan(
                object_id=_slugify(decision.semantic_label, index),
                panel_id=panel_id,
                semantic_label=decision.semantic_label,
                confidence=decision.confidence,
                motion_type=decision.motion_type,
                motion=_motion_spec_for(decision),
            )
        except ValueError as exc:
            raise PipelineStageError(
                stage="analysis",
                input_ref=str(image_path),
                detail=(
                    f"internal MotionSpec heuristic produced a schema-invalid "
                    f"{decision.motion_type.value} object: {exc}"
                ),
                root_cause="the built-in transform_kind/amplitude/speed heuristic table",
                architectural=True,
                proposed_fix="fix the offending entry in _MOTION_HEURISTICS/_DEFAULT_MOTION",
            ) from exc
    return ObjectPlan(
        object_id=_slugify(decision.semantic_label, index),
        panel_id=panel_id,
        semantic_label=decision.semantic_label,
        confidence=decision.confidence,
        motion_type=MotionType.STATIC,
        motion=None,
    )


def build_plan(
    decisions: list[_RawObjectDecision],
    image: Image.Image,
    image_path: Path,
    config: PipelineConfig,
    *,
    allow_all_static: bool = False,
) -> AnimationPlan:
    """Assemble the final schema-valid `AnimationPlan` from validated VLM decisions.

    The chosen PRIMARY always gets a real `MotionSpec`; every other decision the VLM itself
    marked SECONDARY/MICRO also keeps its real motion (Phase 4 -- see
    `_non_primary_object_plan`'s docstring). Only a decision the VLM marked STATIC, or an extra
    PRIMARY that lost to a higher-confidence one, still becomes STATIC in the emitted plan.
    """
    ranked = _rank_candidates(decisions, str(image_path), allow_all_static=allow_all_static)
    if not ranked:
        width, height = image.size
        source = SourceImage(
            path=str(image_path), width=width, height=height, checksum=_checksum(image_path)
        )
        panel = PanelPlan(panel_id=_PANEL_ID, bbox=BBox(x=0.0, y=0.0, width=1.0, height=1.0))
        static_objects = [
            _non_primary_object_plan(decision, index, _PANEL_ID, image_path)
            for index, decision in enumerate(decisions)
        ]
        return AnimationPlan(
            source=source,
            panels=[panel],
            objects=static_objects,
            loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
        )
    chosen = ranked[0]
    rest = [d for d in decisions if d is not chosen]

    objects: list[ObjectPlan] = [
        _non_primary_object_plan(decision, index, _PANEL_ID, image_path)
        for index, decision in enumerate(rest)
    ]

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


def analyze_page(
    image_path: Path,
    client: VLMClient,
    *,
    config: PipelineConfig,
    allow_all_static: bool = False,
) -> AnimationPlan:
    """Public entry point: real manga page -> validated `AnimationPlan`.

    Raises `PipelineStageError` (never silently invents a plan) if the VLM's output can't be
    turned into a usable, schema-valid plan with exactly one PRIMARY object.
    """
    image = Image.open(image_path).convert("RGB")
    vlm_image = _resized_for_vlm(image, config.resolution)
    decisions = _decisions_from_vlm(client, vlm_image, str(image_path))
    return build_plan(
        decisions, image, image_path, config, allow_all_static=allow_all_static
    )


# --- Phase 3.3: panel-aware analysis --------------------------------------------------------
#
# Everything below analyzes a page panel-by-panel instead of as one whole-page VLM call (see
# docs/decisions/0007-panel-aware-analysis.md). `analyze_page` above is completely untouched --
# it remains the always-available page-level path, no longer `run_pipeline`'s default as of
# Phase 10 (see docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md;
# Phase 3.3's original acceptance criterion #2, "default stays page-level," was superseded by
# real Phase 9/10 evidence that panel-level analysis is substantially more reliable).
# The same `ANALYSIS_PROMPT` is deliberately reused unmodified for panel-level calls: its five
# evidence categories (deformation, motion lines, panel/page-level effect lines, pose, implied
# force) are already phrased relative to "the scene"/"the panel", not specifically "the whole
# page" -- there is no real evidence yet that panel-level analysis needs different wording, and
# per the brief, prompt changes should follow evidence, not be made speculatively.


def _rank_panel_candidates(
    decisions: list[tuple[str, _RawObjectDecision]],
    image_ref: str,
    *,
    allow_all_static: bool = False,
) -> list[tuple[str, _RawObjectDecision]]:
    """Panel-tagged counterpart of `_rank_candidates` -- identical `(motion_type priority,
    confidence)` ranking (see that function's docstring for the full rationale, unchanged
    here), operating over `(panel_id, decision)` pairs pooled from potentially several
    panel-level VLM calls instead of decisions from one page-level call.

    Still raises `PipelineStageError` when every decision across every analyzed panel is
    genuinely STATIC -- an honest "no motion cue anywhere on this page" read, which
    `analyze_page_panels` deliberately does NOT retry at the page level (see its docstring):
    that would let VLM nondeterminism silently overrule a real panel-level finding.
    """
    candidates = [(pid, d) for pid, d in decisions if d.motion_type != MotionType.STATIC]
    if not candidates and allow_all_static:
        return []
    if not candidates:
        raise PipelineStageError(
            stage="analysis",
            input_ref=image_ref,
            detail=(
                "VLM marked every object STATIC across every analyzed panel -- an all-STATIC "
                "read is a valid model output but an unusable one for this task (the pipeline "
                "requires one real animated object)"
            ),
            root_cause=(
                "either no drawn motion cue was present on any detected panel, or the model "
                "failed to recognize one that was"
            ),
            architectural=False,
            proposed_fix=(
                "use a page with an unambiguous drawn motion cue, or fall back to a clearly "
                "labeled test fixture rather than fabricating a plan"
            ),
        )
    candidates.sort(
        key=lambda pair: (_MOTION_TYPE_RANK[pair[1].motion_type], pair[1].confidence), reverse=True
    )
    return candidates


def _build_plan_from_panels(
    decisions: list[tuple[str, _RawObjectDecision]],
    panels: list[PanelPlan],
    image: Image.Image,
    image_path: Path,
    config: PipelineConfig,
    *,
    allow_all_static: bool = False,
) -> AnimationPlan:
    """Panel-aware counterpart of `build_plan` -- identical object-construction/heuristic logic
    (see that function's docstring), the only differences being that each `ObjectPlan.panel_id`
    is the real panel it was analyzed from (not a single implicit whole-page panel), and
    `panels` reflects every panel `analyze_page_panels` actually detected -- so downstream
    stages (grounding/validation/segmentation/animation) get real per-panel `PanelPlan.bbox`
    geometry (used e.g. by `pivot.reference="panel"`) instead of the page-level path's
    always-`(0, 0, 1, 1)` placeholder.
    """
    ranked = _rank_panel_candidates(
        decisions, str(image_path), allow_all_static=allow_all_static
    )
    if not ranked:
        width, height = image.size
        source = SourceImage(
            path=str(image_path), width=width, height=height, checksum=_checksum(image_path)
        )
        static_objects = [
            _non_primary_object_plan(decision, index, panel_id, image_path)
            for index, (panel_id, decision) in enumerate(decisions)
        ]
        loop = LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True)
        return AnimationPlan(source=source, panels=panels, objects=static_objects, loop=loop)
    chosen_panel_id, chosen = ranked[0]
    rest = [(pid, d) for pid, d in decisions if d is not chosen]

    objects: list[ObjectPlan] = [
        _non_primary_object_plan(decision, index, panel_id, image_path)
        for index, (panel_id, decision) in enumerate(rest)
    ]

    primary_object_id = _slugify(chosen.semantic_label, len(rest))
    try:
        objects.append(
            ObjectPlan(
                object_id=primary_object_id,
                panel_id=chosen_panel_id,
                semantic_label=chosen.semantic_label,
                confidence=chosen.confidence,
                motion_type=MotionType.PRIMARY,
                motion=_motion_spec_for(chosen),
            )
        )
    except ValueError as exc:
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
    loop = LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True)
    return AnimationPlan(source=source, panels=panels, objects=objects, loop=loop)


def analyze_page_panels(
    image_path: Path,
    client: VLMClient,
    *,
    config: PipelineConfig,
    allow_all_static: bool = False,
) -> AnimationPlan:
    """Panel-aware entry point: real manga page -> validated `AnimationPlan`, analyzed panel by
    panel (via `analysis/panels.py::detect_panels`) instead of as one whole-page VLM call.

    Falls back to the unchanged, page-level `analyze_page` when panel detection itself provides
    no usable signal to analyze:
    - `detect_panels` returns zero candidates (a genuine detector failure -- an image too
      small/degenerate for gutter analysis to mean anything), or
    - detection did return panels, but not a single one's VLM call produced any parseable
      decision at all (every panel's VLM response was unparseable JSON, even after the one
      built-in recovery attempt each `_decisions_from_vlm` call already gets).

    Deliberately does NOT fall back to the page level when every analyzed panel's VLM read was
    genuinely all-STATIC (see `_rank_panel_candidates`) -- that is itself a usable, informative
    result, not a detection failure, and silently re-trying at the page level would let VLM
    nondeterminism (see docs/phase3.2-results.md) quietly overrule a real per-panel finding
    rather than reporting it.

    A single panel's VLM call failing to parse does not abort the whole run -- it is logged and
    skipped, and analysis continues with the remaining panels; this is a real reliability
    advantage of the panel-aware path over the page-level one, where a single VLM parse failure
    fails the entire page (see `analyze_page` / `_decisions_from_vlm`).

    Raises `PipelineStageError` (never silently invents a plan) if neither the panel-aware path
    nor its page-level fallback can produce a usable plan.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    panel_candidates = detect_panels(np.asarray(image))

    if not panel_candidates:
        logger.warning(
            "analysis stage (panel-aware): detect_panels found zero usable panels for %s -- "
            "falling back to page-level analysis",
            image_path,
        )
        return analyze_page(
            image_path, client, config=config, allow_all_static=allow_all_static
        )

    tagged_decisions: list[tuple[str, _RawObjectDecision]] = []
    any_panel_produced_decisions = False
    for candidate in panel_candidates:
        panel_image = Image.fromarray(candidate.crop)
        vlm_image = _resized_for_vlm(panel_image, config.resolution)
        try:
            decisions = _decisions_from_vlm(client, vlm_image, f"{image_path}#{candidate.id}")
        except PipelineStageError as exc:
            logger.warning(
                "analysis stage (panel-aware): panel %s produced no usable VLM decisions (%s) "
                "-- continuing with the remaining panels",
                candidate.id,
                exc.detail,
            )
            continue
        any_panel_produced_decisions = True
        tagged_decisions.extend((candidate.id, decision) for decision in decisions)

    if not any_panel_produced_decisions:
        logger.warning(
            "analysis stage (panel-aware): no detected panel produced a parseable VLM response "
            "for %s -- falling back to page-level analysis",
            image_path,
        )
        return analyze_page(
            image_path, client, config=config, allow_all_static=allow_all_static
        )

    panels = [
        PanelPlan(
            panel_id=candidate.id,
            bbox=bbox_px_to_normalized(candidate.bbox, width, height),
            description=(
                f"panel-aware analysis ({candidate.source}, confidence={candidate.confidence:.2f})"
            ),
        )
        for candidate in panel_candidates
    ]
    return _build_plan_from_panels(
        tagged_decisions,
        panels,
        image,
        image_path,
        config,
        allow_all_static=allow_all_static,
    )
