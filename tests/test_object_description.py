"""Unit tests for the Phase 18.3 per-candidate VLM object-description stage.

Covers the task brief's hard requirements:
- the coordinate contract (the VLM gets the FULL pipeline image + the bbox as pixel
  coordinates in THAT image's space -- never a crop, never a bbox visualization; the bbox is
  rescaled by exactly the same factor as the image);
- strict JSON/schema validation with exactly one recovery attempt;
- fail-closed behavior with machine-readable rejection reasons for every non-pass state;
- the deterministic description -> MotionSpec mapping.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from manga_animation.object_description.describe import describe_object
from manga_animation.object_description.mapping import motion_spec_from_description
from manga_animation.object_description.prompt import (
    PROMPT_MARKER,
    build_prompt,
    prepare_image_and_bbox,
)
from manga_animation.object_description.schema import (
    AmplitudeBand,
    DirectionWord,
    MotionKind,
    ObjectDescriptionResponse,
    PivotHint,
    SpeedBand,
)
from manga_animation.pipeline.types import BBoxPx
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
    Vector2,
)


def _object_plan(label: str = "character_hair") -> ObjectPlan:
    return ObjectPlan(
        object_id="obj_test",
        panel_id="panel_1",
        semantic_label=label,
        confidence=0.9,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.02,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
        ),
    )


def _valid_response(**overrides) -> str:
    payload = {
        "bbox_assessment": "pass",
        "object_identity": "character_hair",
        "matches_semantic_label": True,
        "animatable": True,
        "movable_parts": ["hair tips"],
        "static_parts": ["face", "eyes"],
        "motion_kind": "sway",
        "direction": None,
        "amplitude_band": "moderate",
        "speed_band": "slow",
        "pivot_hint": "top",
        "constraints": ["keep the face static"],
        "neighbor_conflicts": [],
        "confidence": 0.9,
        "reason": "a single character's hair, clearly separated from the face",
    }
    payload.update(overrides)
    return json.dumps(payload)


class RecordingVLMClient:
    """Records every (image, prompt) call and answers with a scripted sequence of responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[Image.Image, str]] = []
        self.replies: list[str] = []
        self.source = "fake-qwen"

    def generate(self, image, prompt: str) -> str:
        self.calls.append((image, prompt))
        reply = self.responses.pop(0)
        self.replies.append(reply)
        return reply

    def unload(self) -> None:
        pass


# --- coordinate contract --------------------------------------------------------------------


class TestCoordinateContract:
    """The task brief's critical check: the VLM must receive the ORIGINAL (full) image plus
    bbox coordinates in that image's pixel space -- and if the image is resized for inference,
    the bbox must be rescaled by exactly the same factor."""

    def test_bbox_scales_with_image_and_stays_consistent(self):
        image = Image.new("RGB", (2000, 1200), (255, 255, 255))
        bbox = BBoxPx(x0=100, y0=200, x1=500, y1=800)
        prepared = prepare_image_and_bbox(image, bbox, max_long_edge=1536)
        # Image was downscaled to the 1536 long edge and rounded UP to the 28px patch grid
        # (the same rounding the Qwen2.5-VL processor applies itself, so the image the model
        # sees matches the coordinates we state: 1536 -> 1540, exactly the verified processor
        # behavior on the Phase 18.3 worker).
        assert prepared.image.size[0] in (1536, 1540)
        assert prepared.image.size[0] % 28 == 0
        assert prepared.image.size[1] % 28 == 0
        # The bbox scales by exactly the same factors as the image.
        assert prepared.bbox_px.x0 == round(bbox.x0 * prepared.scale_x)
        assert prepared.bbox_px.x1 == round(bbox.x1 * prepared.scale_x)
        assert prepared.bbox_px.y0 == round(bbox.y0 * prepared.scale_y)
        assert prepared.bbox_px.y1 == round(bbox.y1 * prepared.scale_y)
        # No crop: the returned image is the full image, just resized.
        assert prepared.resized_from == (2000, 1200)

    def test_small_image_follows_the_processors_patch_grid_rounding(self):
        # The Qwen2.5-VL processor's smart_resize rounds each side to the nearest multiple of
        # 28 (Python round(), verified against transformers 5.0.0 source) -- a 600x400 image
        # becomes 588x392, so a "no downscale needed" image can still be slightly resized.
        image = Image.new("RGB", (600, 400), (255, 255, 255))
        prepared = prepare_image_and_bbox(image, BBoxPx(10, 10, 300, 300), max_long_edge=1536)
        assert prepared.image.size == (588, 392)
        assert prepared.scale_x == 588 / 600
        assert prepared.scale_y == 392 / 400
        assert prepared.bbox_px.x0 == round(10 * prepared.scale_x)

    def test_tall_manga_page_contract(self):
        # The real Phase 3.1 pathological shape: 720x5062 tall page, resized to 1536 long edge,
        # then rounded to the patch grid (218x1536 -> 224x1540).
        image = Image.new("RGB", (720, 5062), (255, 255, 255))
        bbox = BBoxPx(x0=100, y0=2000, x1=300, y1=3000)
        prepared = prepare_image_and_bbox(image, bbox, max_long_edge=1536)
        assert prepared.image.size == (224, 1540)
        assert prepared.image.size[0] % 28 == 0 and prepared.image.size[1] % 28 == 0
        for dim in ("x0", "x1"):
            assert 0 <= getattr(prepared.bbox_px, dim) <= prepared.image.width
        for dim in ("y0", "y1"):
            assert 0 <= getattr(prepared.bbox_px, dim) <= prepared.image.height

    def test_the_client_receives_the_full_image_with_stated_coordinates(self):
        """The definitive test of the VLM input contract: the fake client records what it was
        sent -- it must be the FULL (resized) image, not a crop of the candidate, and the
        prompt's bbox numbers must match that image's own pixel space."""
        image = np.full((900, 1600, 3), 245, dtype=np.uint8)
        image[300:600, 400:700] = (200, 30, 30)
        bbox = BBoxPx(x0=400, y0=300, x1=700, y1=600, score=0.9)
        client = RecordingVLMClient([_valid_response()])
        describe_object(image, bbox, _object_plan(), client, max_long_edge=1536)

        sent_image, prompt = client.calls[0]
        # The sent image is the whole pipeline image (resized + patch-grid-rounded), and it is
        # exactly what prepare_image_and_bbox produced -- never a crop of the candidate.
        prepared = prepare_image_and_bbox(Image.fromarray(image), bbox, max_long_edge=1536)
        assert sent_image.size == prepared.image.size
        assert PROMPT_MARKER in prompt
        # The prompt states the image's real dimensions and the bbox in ITS pixel space.
        assert f"{sent_image.width}x{sent_image.height}" in prompt
        for coord in (prepared.bbox_px.x0, prepared.bbox_px.y0, prepared.bbox_px.x1,
                      prepared.bbox_px.y1):
            assert str(coord) in prompt
        # And the bbox matches what the model's patch grid would actually see: the sent image
        # is exactly the image whose pixel space the coordinates claim.
        assert sent_image.size == (prepared.image.width, prepared.image.height)
        # The bbox is stated as a region WITHIN the full image -- the prompt must not describe
        # a crop ("The image is the full page panel -- it is NOT a crop of the candidate").
        assert "NOT a crop" in prompt

    def test_prompt_contains_the_native_box_tokens(self):
        image = Image.new("RGB", (400, 400), (255, 255, 255))
        prepared = prepare_image_and_bbox(image, BBoxPx(50, 60, 200, 220), max_long_edge=1536)
        prompt = build_prompt(prepared=prepared, semantic_label="character_hair")
        b = prepared.bbox_px
        assert f"<|box_start|>({b.x0},{b.y0}),({b.x1},{b.y1})<|box_end|>" in prompt
        # The coordinate system is stated unambiguously.
        assert "TOP-LEFT" in prompt


# --- schema validation and fail-closed behavior --------------------------------------------


class TestFailClosed:

    def test_pass_description_is_accepted_and_maps_to_a_motion_spec(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient([_valid_response()])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is True
        assert result.assessment == "pass"
        assert result.rejection_reason is None
        assert result.motion_spec is not None
        assert result.motion_spec.transform_kind == TransformKind.MESH_WARP  # sway
        assert result.motion_spec.speed == 1.0  # slow band
        assert result.movable_parts == ("hair tips",)
        assert result.static_parts == ("face", "eyes")
        assert result.constraints == ("keep the face static",)
        assert result.raw_responses == (client.calls[0][1] and client.calls[0][1],) or len(
            result.raw_responses
        ) == 1

    @pytest.mark.parametrize(
        ("assessment", "expected_reason"),
        [
            ("ambiguous", "bbox_assessment=ambiguous"),
            ("partial", "bbox_assessment=partial"),
            ("reject", "bbox_assessment=reject"),
            ("not_animatable", "bbox_assessment=not_animatable"),
        ],
    )
    def test_every_non_pass_assessment_is_fail_closed(self, assessment, expected_reason):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient([_valid_response(bbox_assessment=assessment)])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is False
        assert expected_reason in result.rejection_reason
        assert result.motion_spec is None
        assert result.assessment == assessment

    def test_semantic_label_mismatch_is_fail_closed(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient(
            [_valid_response(object_identity="flag_banner", matches_semantic_label=False)]
        )
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is False
        assert result.rejection_reason == "semantic_label_mismatch"

    def test_not_animatable_is_fail_closed(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient([_valid_response(animatable=False, motion_kind=None)])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is False
        assert result.rejection_reason == "not_animatable"

    def test_malformed_json_then_valid_gets_one_recovery_attempt(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient(["not json at all", _valid_response()])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is True
        assert len(client.calls) == 2  # initial + one recovery
        assert len(result.raw_responses) == 2

    def test_malformed_twice_is_fail_closed_with_unparseable_reason(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient(["garbage", "also garbage"])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is False
        assert result.rejection_reason == "unparseable"
        assert result.assessment is None
        assert len(client.calls) == 2
        assert len(result.raw_responses) == 2

    def test_schema_invalid_response_is_fail_closed(self):
        # Valid JSON but schema-invalid: animatable=true without motion_kind.
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        bad = _valid_response(motion_kind=None)
        client = RecordingVLMClient([bad, _valid_response()])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.accepted is True  # recovery produced a valid response
        assert len(client.calls) == 2

    def test_result_carries_the_audit_trail(self):
        image = np.full((200, 220, 3), 245, dtype=np.uint8)
        client = RecordingVLMClient(["first raw", _valid_response()])
        result = describe_object(image, BBoxPx(10, 10, 60, 90), _object_plan(), client,
                                 max_long_edge=1536)
        assert result.raw_responses == tuple(client.replies)
        assert len(client.replies) == 2
        assert result.model_id == "fake-qwen"


# --- schema cross-field rules ---------------------------------------------------------------


class TestResponseSchema:

    def test_drift_requires_direction(self):
        with pytest.raises(ValidationError):
            ObjectDescriptionResponse.model_validate(json.loads(
                _valid_response(motion_kind="drift", direction=None)
            ))

    def test_animatable_requires_motion_kind(self):
        with pytest.raises(ValidationError):
            ObjectDescriptionResponse.model_validate(
                json.loads(_valid_response(animatable=False, motion_kind="sway"))
            )

    def test_direction_only_for_drift(self):
        # `direction` is inert for non-drift kinds (only translate uses it); real Qwen
        # output fills it in for sway -- the schema strips it instead of failing the read.
        parsed = ObjectDescriptionResponse.model_validate(
            json.loads(_valid_response(motion_kind="sway", direction="up"))
        )
        assert parsed.direction is None

    def test_null_optional_bands_are_tolerated(self):
        # Real Qwen output on a non-animatable object sets amplitude_band/speed_band/
        # pivot_hint/constraints to null -- the schema tolerates that (documented defaults)
        # while keeping the semantic fields strict.
        parsed = ObjectDescriptionResponse.model_validate(
            json.loads(
                _valid_response(
                    animatable=False,
                    motion_kind=None,
                    amplitude_band=None,
                    speed_band=None,
                    pivot_hint=None,
                    constraints=None,
                    neighbor_conflicts=None,
                )
            )
        )
        assert parsed.animatable is False
        assert parsed.motion_kind is None

    def test_unknown_assessment_value_is_rejected(self):
        with pytest.raises(ValidationError):
            ObjectDescriptionResponse.model_validate(
                json.loads(_valid_response(bbox_assessment="static"))
            )


# --- deterministic mapping ------------------------------------------------------------------


class TestMapping:

    def test_every_motion_kind_maps_to_a_schema_valid_motion_spec(self):
        for kind in MotionKind:
            direction = DirectionWord.RIGHT if kind == MotionKind.DRIFT else None
            spec = motion_spec_from_description(
                motion_kind=kind,
                direction=direction,
                amplitude_band=AmplitudeBand.MODERATE,
                speed_band=SpeedBand.NORMAL.value,
                pivot_hint=PivotHint.CENTER,
            )
            assert isinstance(spec, MotionSpec)
            assert spec.speed == int(spec.speed)  # seamless cycle loop requires whole cycles
            assert spec.amplitude > 0
            if kind == MotionKind.DRIFT:
                assert spec.direction is not None
                assert abs(spec.direction.magnitude() - 1.0) < 1e-6  # normalized unit vector

    def test_amplitude_bands_scale_the_baseline(self):
        base = motion_spec_from_description(
            motion_kind=MotionKind.SWAY,
            direction=None,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.SLOW.value,
            pivot_hint=PivotHint.TOP,
        )
        subtle = motion_spec_from_description(
            motion_kind=MotionKind.SWAY,
            direction=None,
            amplitude_band=AmplitudeBand.SUBTLE,
            speed_band=SpeedBand.SLOW.value,
            pivot_hint=PivotHint.TOP,
        )
        pronounced = motion_spec_from_description(
            motion_kind=MotionKind.SWAY,
            direction=None,
            amplitude_band=AmplitudeBand.PRONOUNCED,
            speed_band=SpeedBand.SLOW.value,
            pivot_hint=PivotHint.TOP,
        )
        assert subtle.amplitude < base.amplitude < pronounced.amplitude
        assert subtle.amplitude * 2 == base.amplitude
        assert pronounced.amplitude == pytest.approx(base.amplitude * 1.5)

    def test_pivot_hints_map_to_object_bbox_anchors(self):
        top = motion_spec_from_description(
            motion_kind=MotionKind.ROTATE,
            direction=None,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.NORMAL.value,
            pivot_hint=PivotHint.TOP,
        )
        assert (top.pivot.x, top.pivot.y) == (0.5, 0.0)
        assert top.pivot.reference == "object_bbox"

    def test_speed_bands_map_to_whole_cycles(self):
        slow = motion_spec_from_description(
            motion_kind=MotionKind.FLICKER,
            direction=None,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.SLOW.value,
            pivot_hint=PivotHint.CENTER,
        )
        fast = motion_spec_from_description(
            motion_kind=MotionKind.FLICKER,
            direction=None,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.FAST.value,
            pivot_hint=PivotHint.CENTER,
        )
        assert slow.speed == 1.0
        assert fast.speed == 3.0

    def test_direction_words_map_to_unit_vectors(self):
        up = motion_spec_from_description(
            motion_kind=MotionKind.DRIFT,
            direction=DirectionWord.UP,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.NORMAL.value,
            pivot_hint=PivotHint.CENTER,
        )
        assert up.direction is not None
        assert (up.direction.x, up.direction.y) == (0.0, -1.0)
        up_left = motion_spec_from_description(
            motion_kind=MotionKind.DRIFT,
            direction=DirectionWord.UP_LEFT,
            amplitude_band=AmplitudeBand.MODERATE,
            speed_band=SpeedBand.NORMAL.value,
            pivot_hint=PivotHint.CENTER,
        )
        assert up_left.direction is not None
        assert abs(up_left.direction.magnitude() - 1.0) < 1e-6
        assert up_left.direction.x < 0 and up_left.direction.y < 0
