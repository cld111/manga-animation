from __future__ import annotations

import json

import pytest
from PIL import Image

from manga_animation.analysis.plan_builder import analyze_page
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


def test_multiple_primaries_keeps_highest_confidence_forces_rest_static(sample_image_path, config):
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

    non_primary = [o for o in plan.objects if o.semantic_label != "cape"]
    assert len(non_primary) == 2
    for obj in non_primary:
        assert obj.motion_type == MotionType.STATIC
        assert obj.motion is None


def test_all_static_raises_pipeline_stage_error_not_a_fabricated_plan(sample_image_path, config):
    decisions = [_decision("background", "static"), _decision("character_face", "static")]
    client = FakeVLMClient([json.dumps(decisions)])

    with pytest.raises(PipelineStageError) as excinfo:
        analyze_page(sample_image_path, client, config=config)

    assert excinfo.value.stage == "analysis"
    assert not excinfo.value.architectural


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
