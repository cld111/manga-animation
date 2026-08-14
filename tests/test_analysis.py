from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from manga_animation.analysis.plan_builder import analyze_page, analyze_page_panels
from manga_animation.core.config import PipelineConfig
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import MotionType


class FakeVLMClient:
    """A `VLMClient` double: returns canned strings in order, records prompts it was given.

    Mirrors `tests/test_benchmarking.py`'s `FakeAdapter` pattern -- no torch/transformers/
    network required.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.image_sizes: list[tuple[int, int]] = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        self.image_sizes.append(image.size)
        if not self._responses:
            raise AssertionError("FakeVLMClient ran out of canned responses")
        return self._responses.pop(0)


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 200), color=(255, 255, 255)).save(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


def _decision(
    label: str, motion_type: str, confidence: float = 0.8, motion_description: str | None = None
) -> dict:
    d = {
        "semantic_label": label,
        "motion_type": motion_type,
        "confidence": confidence,
        "reason": "test fixture reason",
    }
    if motion_description is not None:
        d["motion_description"] = motion_description
    return d


def test_valid_json_single_primary_produces_valid_plan(sample_image_path, config):
    decisions = [
        _decision("background", "static"),
        _decision("flag_banner", "primary", confidence=0.9, motion_description="sways in the wind"),
    ]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    assert len(client.prompts) == 1
    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "flag_banner"
    assert primary_objects[0].motion is not None
    static_objects = [o for o in plan.objects if o.motion_type == MotionType.STATIC]
    assert len(static_objects) == 1
    assert static_objects[0].motion is None
    # schema validated this already (AnimationPlan construction would have raised), but
    # double check the loop config was actually threaded through from PipelineConfig
    assert plan.loop.fps == config.fps
    assert plan.loop.duration_s == config.duration_s


def test_multiple_primaries_forces_the_losing_one_static_but_keeps_a_real_secondary(
    sample_image_path, config
):
    """Phase 4 (docs/decisions/0010-multi-object-layer-decomposition.md): the losing extra

    "primary" candidate is still forced to STATIC -- unchanged, `_rank_candidates`'
    single-object-among-primaries policy isn't being generalized here -- but a real
    "secondary" read is no longer collapsed to STATIC just because it wasn't chosen as
    PRIMARY; it keeps its own motion_type and gets a real MotionSpec (the pipeline can now
    animate more than one object per page).
    """
    decisions = [
        _decision("hair", "primary", confidence=0.6, motion_description="blows sideways"),
        _decision("cape", "primary", confidence=0.95, motion_description="flutters behind"),
        _decision("sword", "secondary", confidence=0.7, motion_description="trails the swing"),
    ]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "cape"

    hair = next(o for o in plan.objects if o.semantic_label == "hair")
    assert hair.motion_type == MotionType.STATIC
    assert hair.motion is None

    sword = next(o for o in plan.objects if o.semantic_label == "sword")
    assert sword.motion_type == MotionType.SECONDARY
    assert sword.motion is not None


def test_all_static_raises_pipeline_stage_error_not_a_fabricated_plan(sample_image_path, config):
    decisions = [_decision("background", "static"), _decision("character_face", "static")]
    client = FakeVLMClient([json.dumps(decisions)])

    with pytest.raises(PipelineStageError) as excinfo:
        analyze_page(sample_image_path, client, config=config)

    assert excinfo.value.stage == "analysis"
    assert not excinfo.value.architectural


def test_all_static_can_be_recorded_as_a_valid_panel_outcome(sample_image_path, config):
    decisions = [_decision("background", "static"), _decision("character_face", "static")]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config, allow_all_static=True)

    assert plan.objects
    assert all(obj.motion_type == MotionType.STATIC for obj in plan.objects)


def test_recovery_pass_fixes_malformed_json(sample_image_path, config):
    valid = [_decision("banner", "primary", confidence=0.9, motion_description="waves")]
    client = FakeVLMClient(["this is not json at all {{{", json.dumps(valid)])

    plan = analyze_page(sample_image_path, client, config=config)

    assert len(client.prompts) == 2
    # the recovery prompt should reference the failure so the model has something to fix
    assert "error" in client.prompts[1].lower() or "corrected" in client.prompts[1].lower()
    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1


def test_recovery_pass_still_invalid_raises_and_is_not_swallowed(sample_image_path, config):
    client = FakeVLMClient(["not json", "still not valid json {{{"])

    with pytest.raises(PipelineStageError) as excinfo:
        analyze_page(sample_image_path, client, config=config)

    assert len(client.prompts) == 2
    assert excinfo.value.stage == "analysis"
    assert "recovery" in excinfo.value.detail.lower() or "invalid" in excinfo.value.detail.lower()


def test_json_wrapped_in_markdown_fences_still_parses(sample_image_path, config):
    decisions = [_decision("flag", "primary", confidence=0.9, motion_description="ripples")]
    wrapped = (
        f"Sure, here is the analysis:\n```json\n{json.dumps(decisions)}\n```\nHope that helps!"
    )
    client = FakeVLMClient([wrapped])

    plan = analyze_page(sample_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "flag"


def test_invalid_motion_type_value_triggers_recovery(sample_image_path, config):
    bad = [
        {
            "semantic_label": "hair",
            "motion_type": "kinda_moving",
            "confidence": 0.5,
            "reason": "x",
        }
    ]
    valid = [_decision("hair", "primary", confidence=0.8, motion_description="sways")]
    client = FakeVLMClient([json.dumps(bad), json.dumps(valid)])

    plan = analyze_page(sample_image_path, client, config=config)

    assert len(client.prompts) == 2
    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1


def test_tall_page_is_downscaled_to_config_resolution_before_reaching_the_vlm(tmp_path):
    """Real bug found on the first Kaggle run: a 720x5062 (~7:1) page fed at full resolution

    produced enough vision tokens to OOM a T4 during generate() -- config.resolution exists
    to bound this (see docs/architecture.md's "GPU Awareness") and analyze_page must apply it
    to what the VLM actually sees, without changing SourceImage's recorded true dimensions.
    """
    tall_page = tmp_path / "tall_page.png"
    Image.new("RGB", (720, 5062), color=(255, 255, 255)).save(tall_page)
    config = PipelineConfig(resolution=1024)
    decisions = [_decision("banner", "primary", motion_description="waves")]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(tall_page, client, config=config)

    assert len(client.image_sizes) >= 1
    seen_w, seen_h = client.image_sizes[0]
    assert max(seen_w, seen_h) <= 1024
    assert seen_h / seen_w == pytest.approx(5062 / 720, rel=0.01)  # aspect ratio preserved
    # SourceImage keeps the TRUE source dimensions, not the resized copy's
    assert plan.source.width == 720
    assert plan.source.height == 5062


def test_page_within_resolution_is_not_upscaled(sample_image_path, config):
    """sample_image_path is 100x200 -- well under the default resolution -- must pass through

    unchanged (never upscaled).
    """
    client = FakeVLMClient([json.dumps([_decision("banner", "primary", motion_description="x")])])

    analyze_page(sample_image_path, client, config=config)

    assert client.image_sizes[0] == (100, 200)


# --- Phase 3.2: ranked candidates (no longer discards SECONDARY/MICRO reads) ---------------


def test_no_primary_but_a_secondary_candidate_still_produces_a_usable_plan(
    sample_image_path, config
):
    """Real Phase 3.1-era gap: a VLM read with real SECONDARY/MICRO motion signal but no

    literal "primary" label used to be treated identically to an all-STATIC read (unusable) --
    even though the model DID identify a motion-worthy object, just not confidently enough to
    call it primary. `_rank_candidates` now considers the whole non-STATIC pool, not only
    objects labeled "primary" (see docs/decisions/0006-grounding-target-validation.md).
    """
    decisions = [
        _decision("background", "static"),
        _decision("character_hair", "secondary", confidence=0.7, motion_description="sways"),
    ]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "character_hair"
    assert primary_objects[0].motion is not None


def test_only_a_micro_candidate_also_produces_a_usable_plan(sample_image_path, config):
    decisions = [_decision("eye", "micro", confidence=0.6, motion_description="blinks")]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "eye"


def test_primary_secondary_and_micro_all_get_real_motion_in_one_plan(sample_image_path, config):
    """Phase 4's core new capability: a plan can carry more than one animated object at once --

    PRIMARY, SECONDARY, and MICRO decisions each keep their own real MotionSpec, not just the
    single chosen PRIMARY. `object_id`s stay distinct so downstream stages can address each
    object independently.
    """
    decisions = [
        _decision("cape", "primary", confidence=0.9, motion_description="flutters behind"),
        _decision("hair", "secondary", confidence=0.7, motion_description="sways"),
        _decision("eye", "micro", confidence=0.5, motion_description="blinks"),
        _decision("background", "static"),
    ]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    by_label = {o.semantic_label: o for o in plan.objects}
    assert by_label["cape"].motion_type == MotionType.PRIMARY
    assert by_label["cape"].motion is not None
    assert by_label["hair"].motion_type == MotionType.SECONDARY
    assert by_label["hair"].motion is not None
    assert by_label["eye"].motion_type == MotionType.MICRO
    assert by_label["eye"].motion is not None
    assert by_label["background"].motion_type == MotionType.STATIC
    assert by_label["background"].motion is None

    animated = [o for o in plan.objects if o.motion is not None]
    assert len({o.object_id for o in animated}) == len(animated)  # distinct object_ids


def test_a_real_primary_still_outranks_a_more_confident_secondary(sample_image_path, config):
    """motion_type strictly dominates confidence in the ranking -- a "secondary" read must

    never bump out an actual "primary" one just because the model was more confident about it.
    """
    decisions = [
        _decision("cape", "primary", confidence=0.5, motion_description="flutters"),
        _decision("hair", "secondary", confidence=0.99, motion_description="sways"),
    ]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page(sample_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert primary_objects[0].semantic_label == "cape"


def test_analysis_prompt_broadens_evidence_beyond_deformation_on_the_object_itself():
    """Real Phase 3.1 finding: the original prompt only recognized motion cues drawn directly

    on an object, which structurally could not justify motion for a page whose only cue was a
    page-level speed-line effect (see docs/phase3-results.md finding #2). Guard against
    silently reverting the broadened prompt.
    """
    from manga_animation.analysis.plan_builder import ANALYSIS_PROMPT

    lowered = ANALYSIS_PROMPT.lower()
    assert "page-level" in lowered or "panel-level" in lowered
    assert "pose" in lowered


def test_drawn_effects_get_effect_specific_motion_specs():
    """Phase 16 Drawn Effect Track: effect labels (`impact_burst`, `energy_field`, `smoke`,
    `rain`, `speed_lines`) must each get a natural effect-specific MotionSpec -- NOT the
    generic rigid translate that every effect received before this phase (the locally-proven
    gap: `rain`/`green_fluid`/`speed_lines`/`impact_effect` all collapsed to
    `_DEFAULT_MOTION` = TRANSLATE amplitude 0.02)."""
    from manga_animation.analysis.plan_builder import _motion_spec_for, _RawObjectDecision
    from manga_animation.schemas.animation_plan import MotionType

    cases = {
        # label, motion_description -> expected transform_kind
        "impact_burst": ("impact burst radiates outward from the hit point", "radial_expand"),
        "energy_field": ("the energy field pulses and glows", "radial_expand"),
        "glow_effect": ("the glow expands in a ring", "radial_expand"),
        "smoke_cloud": ("smoke drifts upward", "mesh_warp"),
        "water_splash": ("water splashes outward", "mesh_warp"),
        "green_fluid": ("fluid flows and ripples", "mesh_warp"),
        "rain": ("rain falls downward", "translate"),
        "speed_lines": ("speed lines streak along the motion direction", "mesh_warp"),
        "spark_shower": ("sparks flicker", "opacity"),
    }
    for label, (desc, expected_kind) in cases.items():
        decision = _RawObjectDecision(
            semantic_label=label,
            motion_type=MotionType.SECONDARY,
            confidence=0.7,
            reason="a drawn effect present in the panel",
            motion_description=desc,
        )
        spec = _motion_spec_for(decision)
        assert spec.transform_kind.value == expected_kind, (
            f"{label} should map to {expected_kind}, got {spec.transform_kind.value}"
        )
    # The radial class anchors at the object's own center -- the natural burst origin.
    decision = _RawObjectDecision(
        semantic_label="impact_burst",
        motion_type=MotionType.SECONDARY,
        confidence=0.7,
        reason="x",
        motion_description="impact burst radiates outward",
    )
    spec = _motion_spec_for(decision)
    assert spec.pivot.reference == "object_bbox"
    assert spec.pivot.x == pytest.approx(0.5)
    assert spec.pivot.y == pytest.approx(0.5)


def test_effect_label_dominates_object_words_in_its_description():
    """Phase 16 regression (real GPU finding on `villainess_ending_scuffle`): an effect's
    motion_description routinely names the object it is attached to ("bursts outward from
    the weapon clash"), and matching on label+description let the earlier object heuristics
    (`sword/blade/weapon` -> ROTATE) steal the effect and give it a rigid rotation instead
    of its natural pulse. The effect's own label must decide the effect class; object words
    inside its description must not override it."""
    from manga_animation.analysis.plan_builder import _motion_spec_for, _RawObjectDecision
    from manga_animation.schemas.animation_plan import MotionType

    cases = [
        ("impact_burst", "impact burst radiates around the sword swing", "radial_expand"),
        ("impact_burst", "bursts outward from the weapon clash", "radial_expand"),
        ("speed_lines", "speed lines streak behind the character's hand", "mesh_warp"),
        ("smoke_cloud", "smoke drifts from the burning cloth", "mesh_warp"),
        ("energy_field", "energy pulses around the character's arm", "radial_expand"),
    ]
    for label, desc, expected_kind in cases:
        decision = _RawObjectDecision(
            semantic_label=label,
            motion_type=MotionType.SECONDARY,
            confidence=0.7,
            reason="x",
            motion_description=desc,
        )
        spec = _motion_spec_for(decision)
        assert spec.transform_kind.value == expected_kind, (
            f"{label!r} with description {desc!r} should map to {expected_kind}, "
            f"got {spec.transform_kind.value}"
        )


def test_analysis_prompt_explicitly_asks_for_drawn_effects_as_animation_targets():
    """Phase 16: the analysis prompt must instruct the VLM to list ALREADY-DRAWN effects
    (speed lines, impact bursts, energy fields, smoke, water, glow) as first-class animation
    candidates, not just objects -- while keeping speech bubbles/text/panel borders static."""
    from manga_animation.analysis.plan_builder import ANALYSIS_PROMPT

    lowered = ANALYSIS_PROMPT.lower()
    assert "speed lines" in lowered or "speed_lines" in lowered
    assert "impact" in lowered
    assert "energy" in lowered
    assert "smoke" in lowered
    # Artwork preservation is explicit: text/bubbles/borders must stay static.
    assert "speech bubbles" in lowered
    assert "panel borders" in lowered
    assert "must stay static" in lowered


# --- Phase 3.3: panel-aware analysis --------------------------------------------------------


def _noise_block(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def two_panel_image_path(tmp_path):
    """A real, detectable two-panel page: two textured (non-uniform) blocks separated by a

    wide blank gutter -- same construction style proven against `analysis/panels.py` in
    `tests/test_panels.py`.
    """
    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = _noise_block(300, 300, seed=1)  # top panel: rows 0-300
    page[500:900, 0:300] = _noise_block(400, 300, seed=2)  # bottom panel: rows 500-900
    path = tmp_path / "two_panel_page.png"
    Image.fromarray(page).save(path)
    return path


def test_analyze_page_panels_detects_and_labels_multiple_panel_plans(two_panel_image_path, config):
    top_decision = _decision(
        "hair", "primary", confidence=0.8, motion_description="sways"
    )
    bottom_decision = _decision("background", "static")
    client = FakeVLMClient([json.dumps([top_decision]), json.dumps([bottom_decision])])

    plan = analyze_page_panels(two_panel_image_path, client, config=config)

    assert len(plan.panels) == 2
    # panels are real, distinct sub-regions of the page (not the page-level path's single
    # (0, 0, 1, 1) whole-page placeholder) -- each covers less than the full page height.
    for panel in plan.panels:
        assert panel.bbox.height < 1.0
    assert plan.panels[0].panel_id != plan.panels[1].panel_id

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "hair"
    # the PRIMARY object's panel_id must reference a real panel this plan actually declares
    assert primary_objects[0].panel_id in {p.panel_id for p in plan.panels}


def test_analyze_page_panels_falls_back_to_page_level_when_detection_finds_no_panels(
    sample_image_path, config
):
    """`sample_image_path` (100x200, blank) is well above the detector's minimum-size floor,

    but a real fallback trigger (zero panels) is exercised directly against a degenerate,
    below-minimum-size image instead -- see `analysis/panels.py::_MIN_IMAGE_DIM_PX`.
    """
    import numpy as np

    tiny_path = sample_image_path.parent / "tiny.png"
    Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8)).save(tiny_path)
    decisions = [_decision("banner", "primary", motion_description="waves")]
    client = FakeVLMClient([json.dumps(decisions)])

    plan = analyze_page_panels(tiny_path, client, config=config)

    # exactly one VLM call -- the page-level fallback path, not a panel-level one
    assert len(client.prompts) == 1
    assert len(plan.panels) == 1
    assert plan.panels[0].bbox.x == 0.0 and plan.panels[0].bbox.width == 1.0


def test_analyze_page_panels_falls_back_to_page_level_when_every_panel_response_is_unparseable(
    two_panel_image_path, config
):
    """Detection succeeds (2 real panels), but every panel-level VLM call (including each

    panel's one built-in recovery attempt) returns garbage -- `analyze_page_panels` must fall
    back to a real page-level VLM call rather than raising, per its documented fallback trigger.
    """
    valid_page_level = [_decision("banner", "primary", motion_description="waves")]
    client = FakeVLMClient(
        [
            "not json at all",  # panel 0, first attempt
            "still not json",  # panel 0, recovery attempt
            "also not json",  # panel 1, first attempt
            "nope",  # panel 1, recovery attempt
            json.dumps(valid_page_level),  # page-level fallback call
        ]
    )

    plan = analyze_page_panels(two_panel_image_path, client, config=config)

    assert len(client.prompts) == 5
    assert len(plan.panels) == 1  # the page-level fallback's single implicit panel
    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert primary_objects[0].semantic_label == "banner"


def test_analyze_page_panels_one_panels_vlm_failure_does_not_abort_the_others(
    two_panel_image_path, config
):
    """A single panel's VLM response being unparseable must not sink the whole panel-aware

    run -- this is a real reliability advantage over the page-level path (see
    `analyze_page_panels`'s docstring): analysis continues with whatever other panels did
    produce usable decisions.
    """
    valid = [_decision("hair", "secondary", confidence=0.7, motion_description="sways")]
    client = FakeVLMClient(
        [
            "garbage",  # panel 0, first attempt
            "still garbage",  # panel 0, recovery attempt (both fail -- panel 0 contributes nothing)
            json.dumps(valid),  # panel 1, first attempt (succeeds)
        ]
    )

    plan = analyze_page_panels(two_panel_image_path, client, config=config)

    assert len(client.prompts) == 3
    assert len(plan.panels) == 2  # detection still recorded both real panels
    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert len(primary_objects) == 1
    assert primary_objects[0].semantic_label == "hair"


def test_analyze_page_panels_ranking_preserves_motion_type_over_confidence_across_panels(
    two_panel_image_path, config
):
    """`_rank_panel_candidates` must apply the same "motion_type strictly dominates confidence"

    rule as the page-level `_rank_candidates`, across panels -- a low-confidence PRIMARY in one
    panel must still outrank a high-confidence SECONDARY in another.
    """
    panel_0_secondary = _decision(
        "cape", "secondary", confidence=0.99, motion_description="flutters"
    )
    panel_1_primary = _decision("sword", "primary", confidence=0.5, motion_description="swings")
    client = FakeVLMClient([json.dumps([panel_0_secondary]), json.dumps([panel_1_primary])])

    plan = analyze_page_panels(two_panel_image_path, client, config=config)

    primary_objects = [o for o in plan.objects if o.motion_type == MotionType.PRIMARY]
    assert primary_objects[0].semantic_label == "sword"


def test_analyze_page_panels_all_static_across_every_panel_raises_and_does_not_fall_back(
    two_panel_image_path, config
):
    """An honest "no motion cue on any detected panel" read is a valid, informative result --

    `analyze_page_panels` must raise `PipelineStageError(stage="analysis")` exactly like the
    page-level path, and must NOT silently retry at the page level (that would let VLM
    nondeterminism quietly overrule a real per-panel finding -- see the function's docstring).
    """
    client = FakeVLMClient(
        [json.dumps([_decision("background", "static")]), json.dumps([_decision("wall", "static")])]
    )

    with pytest.raises(PipelineStageError) as excinfo:
        analyze_page_panels(two_panel_image_path, client, config=config)

    assert excinfo.value.stage == "analysis"
    # exactly 2 calls (one per real panel) -- no third, page-level fallback call was made
    assert len(client.prompts) == 2


def test_analyze_page_panels_coordinates_map_correctly_to_the_page(two_panel_image_path, config):
    """Each detected panel's `PanelPlan.bbox` (normalized) must correspond to the real

    page-space region it was actually detected at -- the top panel (rows 0-300 of a 900px-tall
    page) must land near normalized y=0, and the bottom one (rows 500-900) must land near the
    page's bottom half. This is the "no downstream component should need to know whether a
    target came from page-level or panel-level analysis" guarantee, checked concretely.
    """
    client = FakeVLMClient(
        [json.dumps([_decision("a", "static")]), json.dumps([_decision("b", "static", 0.1)])]
    )
    with pytest.raises(PipelineStageError):
        # all-STATIC still raises, but the plan's *panels* aren't built in that path -- so
        # instead directly exercise detection + the panel list this function assembles, via a
        # real (non-static) call, below.
        analyze_page_panels(two_panel_image_path, client, config=config)

    client2 = FakeVLMClient(
        [
            json.dumps([_decision("hair", "primary", motion_description="x")]),
            json.dumps([_decision("bg", "static")]),
        ]
    )
    plan = analyze_page_panels(two_panel_image_path, client2, config=config)
    panels_by_y = sorted(plan.panels, key=lambda p: p.bbox.y)

    assert panels_by_y[0].bbox.y < 0.1  # top panel starts near the page's top edge
    assert panels_by_y[1].bbox.y > 0.4  # bottom panel starts around/after the page's midpoint
