"""Behavioral tests for src/manga_animation/pipeline/orchestrator.py.

Exercises the real stage-wiring code with fake model clients (no torch/transformers/GPU
needed, mirroring every other stage's test style -- see tests/test_analysis.py,
tests/test_grounding.py, tests/test_segmentation.py, tests/test_reconstruction.py) plus the
REAL cv2/animation/compositing code and (when available) the REAL ffmpeg encode -- this is the
one place that proves the six stages actually fit together, not just that each one works in
isolation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_animation.core.config import PipelineConfig, load_config
from manga_animation.grounding.client import Detection
from manga_animation.pipeline.orchestrator import (
    PipelineRunResult,
    _candidate_source,
    _select_primary,
    build_default_clients,
    run_pipeline,
)
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import MotionType
from manga_animation.segmentation.client import MaskCandidate

pytestmark = pytest.mark.filterwarnings("ignore")


def _resolve_test_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return None


requires_ffmpeg = pytest.mark.skipif(
    _resolve_test_ffmpeg() is None,
    reason="no ffmpeg binary resolvable (neither system nor imageio-ffmpeg)",
)


# --- fakes --------------------------------------------------------------------------------


_VALIDATION_PROMPT_MARKER = "Does the image above show"
"""Distinctive substring of validation/validate.py's verification prompt -- lets these fakes

tell "analysis stage asking for the page's object list" apart from "Phase 3.2 validation stage
asking whether one cropped candidate matches its target" without needing two separate client
parameters threaded through every test.
"""

_MASK_SEMANTICS_PROMPT_MARKER = "Does the bright region show"
"""Distinctive substring of validation/mask_semantics.py's verification prompt (Phase 12) --

same role as `_VALIDATION_PROMPT_MARKER`, but for the post-segmentation semantic mask gate.
Every fake VLM client in this module defaults to ACCEPTing this prompt (the gate is enabled by
default -- `PipelineConfig.enable_semantic_mask_validation=True` -- so every pre-existing test
in this file that doesn't care about mask_semantics still needs a passing response for it, not
just for the pre-segmentation `_VALIDATION_PROMPT_MARKER` check); dedicated mask_semantics tests
override this via each fake's `mask_semantics_matches` parameter.
"""


def _fake_mask_semantics_response(matches: bool, *, confidence: float | None = None) -> str:
    resolved_confidence = confidence if confidence is not None else (0.9 if matches else 0.1)
    return json.dumps(
        {
            "mask_matches_object": matches,
            "confidence": resolved_confidence,
            "unexpected_content": [] if matches else ["fake unrelated content"],
            "reason": "fake mask semantics response",
        }
    )


class FakeVLMClient:
    """Answers the analysis-stage prompt (returns canned `decisions`), the Phase 3.2

    pre-segmentation validation-stage prompt, and the Phase 12 post-segmentation mask_semantics
    prompt -- see `_VALIDATION_PROMPT_MARKER`/`_MASK_SEMANTICS_PROMPT_MARKER`.
    `verification_matches=False`/`mask_semantics_matches=False` let a test make either gate
    reject instead, without needing a second fake class.
    """

    def __init__(
        self,
        decisions: list[dict],
        *,
        verification_matches: bool = True,
        mask_semantics_matches: bool = True,
        mask_semantics_confidence: float | None = None,
    ):
        self._decisions = decisions
        self._verification_matches = verification_matches
        self._mask_semantics_matches = mask_semantics_matches
        self._mask_semantics_confidence = mask_semantics_confidence
        self.mask_semantics_prompts: list[str] = []
        """Every prompt this fake answered as a mask_semantics-stage call, in call order --

        lets a test assert exactly which/how many objects reached this stage (e.g. proving an
        object dropped by an earlier guard, such as the cross-object overlap check, never wastes
        a mask_semantics call at all -- an independent adversarial QA review finding)."""

    def generate(self, image, prompt: str) -> str:
        if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
            self.mask_semantics_prompts.append(prompt)
            return _fake_mask_semantics_response(
                self._mask_semantics_matches, confidence=self._mask_semantics_confidence
            )
        if _VALIDATION_PROMPT_MARKER in prompt:
            return json.dumps(
                {
                    "matches": self._verification_matches,
                    "confidence": 0.9 if self._verification_matches else 0.1,
                    "reason": "fake validation response",
                }
            )
        return json.dumps(self._decisions)


class ValidationSequenceVLMClient:
    """Like `FakeVLMClient`, but rejects the first `reject_first_n` validation calls and

    accepts every one after -- exercises the orchestrator's "try the next ranked grounding
    candidate" retry path deterministically. Always ACCEPTs mask_semantics prompts (that stage
    is not what these tests exercise).
    """

    def __init__(self, decisions: list[dict], *, reject_first_n: int = 0):
        self._decisions = decisions
        self._reject_first_n = reject_first_n
        self._validation_calls = 0

    def generate(self, image, prompt: str) -> str:
        if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
            return _fake_mask_semantics_response(True)
        if _VALIDATION_PROMPT_MARKER in prompt:
            self._validation_calls += 1
            matches = self._validation_calls > self._reject_first_n
            return json.dumps(
                {
                    "matches": matches,
                    "confidence": 0.9 if matches else 0.1,
                    "reason": f"fake validation response #{self._validation_calls}",
                }
            )
        return json.dumps(self._decisions)


class RejectFromNthValidationVLMClient:
    """Like `ValidationSequenceVLMClient`, but rejects starting from the `reject_from_call`'th

    validation call onward (1-based) instead of the first N -- lets a Phase 4 multi-object test
    make an EARLIER object's (e.g. PRIMARY's) validation succeed while a LATER object's (e.g. a
    SECONDARY's) fails, deterministically, by call order (`objects_to_animate` always processes
    PRIMARY first). Always ACCEPTs mask_semantics prompts (not what these tests exercise).
    """

    def __init__(self, decisions: list[dict], *, reject_from_call: int):
        self._decisions = decisions
        self._reject_from_call = reject_from_call
        self._validation_calls = 0

    def generate(self, image, prompt: str) -> str:
        if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
            return _fake_mask_semantics_response(True)
        if _VALIDATION_PROMPT_MARKER in prompt:
            self._validation_calls += 1
            matches = self._validation_calls < self._reject_from_call
            return json.dumps(
                {
                    "matches": matches,
                    "confidence": 0.9 if matches else 0.1,
                    "reason": f"fake validation response #{self._validation_calls}",
                }
            )
        return json.dumps(self._decisions)


class FailingGroundingClient:
    model_id = "fake-grounding-dino"

    def load(self) -> None:
        pass

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return []  # nothing above threshold

    def unload(self) -> None:
        pass


class FakeGroundingClient:
    """Returns one or more ranked candidate boxes -- `boxes` (highest score first) supersedes

    the single-box `box` shorthand when given, letting a test exercise the validator's
    grounding-candidate retry loop deterministically.
    """

    model_id = "fake-grounding-dino"

    def __init__(
        self,
        box: tuple[int, int, int, int] = (10, 10, 60, 90),
        *,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ):
        self.boxes = boxes if boxes is not None else [box]
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return [
            Detection(label="banner", score=0.9 - 0.1 * i, box=b)
            for i, b in enumerate(self.boxes)
        ]

    def unload(self) -> None:
        self.unloaded = True


class MultiObjectFakeGroundingClient:
    """Returns a different box per semantic_label -- matched by substring against the

    grounding prompt `grounding/ground.py::_prompt_from_label` builds (`"hanging banner."` for
    `semantic_label="hanging_banner"`) -- lets a Phase 4 multi-object test give PRIMARY and
    SECONDARY/MICRO objects distinct, non-overlapping regions instead of
    `FakeGroundingClient`'s single shared box. A label with no entry in `boxes_by_label`
    detects nothing (mirrors `FailingGroundingClient`, for that one object only).
    """

    model_id = "fake-grounding-dino"

    def __init__(self, boxes_by_label: dict[str, tuple[int, int, int, int]]):
        self._boxes_by_label = {
            label.replace("_", " "): box for label, box in boxes_by_label.items()
        }
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        for label, box in self._boxes_by_label.items():
            if label in text_prompt:
                return [Detection(label=label, score=0.9, box=box)]
        return []

    def unload(self) -> None:
        self.unloaded = True


class RecordingGroundingClient:
    """Like `MultiObjectFakeGroundingClient`, but also records the actual `(image.shape,

    prompt)` of every `detect()` call it received -- lets a Phase 5.1 test assert exactly what
    region (full page vs. a specific panel's crop) grounding ran against per object, not just
    that the final render succeeded. `boxes_by_label` values are CROP-LOCAL coordinates,
    matching the real `GroundingDinoClient.detect`'s contract.
    """

    model_id = "fake-grounding-dino"

    def __init__(self, boxes_by_label: dict[str, tuple[int, int, int, int]]):
        self._boxes_by_label = {
            label.replace("_", " "): box for label, box in boxes_by_label.items()
        }
        self.calls: list[tuple[tuple[int, ...], str]] = []
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.calls.append((image.shape, text_prompt))
        for label, box in self._boxes_by_label.items():
            if label in text_prompt:
                return [Detection(label=label, score=0.9, box=box)]
        return []

    def unload(self) -> None:
        self.unloaded = True


def _region_mask(h: int, w: int, box) -> np.ndarray:
    """A full-image-shape uint8 0/255 mask: a diamond inscribed in `box`, touching each of its

    4 edges at exactly its midpoint -- same tight-bbox-equals-`box` property a solid rectangle
    fill has, but without a solid rectangle's 100%-of-every-edge touch fraction. Phase 8.3:
    `segmentation/segment.py::_validate_mask_shape` now rejects real masks shaped like that (see
    docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md) -- a raw rectangle fill here
    would trip that same real check, so every fake segmentation mask in this test module needs a
    shape an actual SAM output could plausibly have.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (box.x0 + box.x1 - 1) / 2.0, (box.y0 + box.y1 - 1) / 2.0
    ax, ay = max((box.x1 - box.x0) / 2.0, 1e-9), max((box.y1 - box.y0) / 2.0, 1e-9)
    yy, xx = np.mgrid[box.y0 : box.y1, box.x0 : box.x1]
    local = (np.abs(xx - cx) / ax + np.abs(yy - cy) / ay) <= 1.0
    mask[box.y0 : box.y1, box.x0 : box.x1] = local.astype(np.uint8) * 255
    return mask


class FakeSegmentationClient:
    model_id = "fake-sam2.1"

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[0], image.shape[1]
        return [MaskCandidate(mask=_region_mask(h, w, box), iou_score=0.9)]

    def unload(self) -> None:
        pass


class MaskShapeControlledSegmentationClient:
    """Like `FakeSegmentationClient`, but returns a one-sided mask (Phase 8.3's

    `segmentation/segment.py::_validate_mask_shape` real, evidenced "hugs one bbox edge, but not
    the opposite one" rejection case -- a diamond silhouette unioned with a solid strip along
    the box's own left third, matching the real downloaded SAM mask's LEFT=45.5%/RIGHT=0.56%
    asymmetry) for any box in `bad_boxes`, and a realistic diamond mask (`_region_mask`) for
    everything else -- lets a test control exactly which object gets a defective real mask
    shape.
    """

    model_id = "fake-sam2.1"

    def __init__(self, bad_boxes: set[tuple[int, int, int, int]]):
        self._bad_boxes = bad_boxes

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[0], image.shape[1]
        mask = _region_mask(h, w, box)
        if box.as_xyxy() in self._bad_boxes:
            strip_x1 = box.x0 + max(1, (box.x1 - box.x0) // 3)
            mask[box.y0 : box.y1, box.x0 : strip_x1] = 255
        return [MaskCandidate(mask=mask, iou_score=0.9)]

    def unload(self) -> None:
        pass


class FakeReconstructionClient:
    def load(self) -> None:
        pass

    def inpaint(self, image, hole_mask):
        return image

    def unload(self) -> None:
        pass


def _primary_decision(label: str = "hanging_banner") -> dict:
    return {
        "semantic_label": label,
        "motion_type": "primary",
        "confidence": 0.9,
        "reason": "test fixture",
        "motion_description": "sways left and right",
    }


def _static_decision(label: str = "background") -> dict:
    return {
        "semantic_label": label,
        "motion_type": "static",
        "confidence": 0.9,
        "reason": "test fixture",
    }


def _secondary_decision(label: str = "trailing_cloth") -> dict:
    return {
        "semantic_label": label,
        "motion_type": "secondary",
        "confidence": 0.8,
        "reason": "test fixture",
        "motion_description": "trails behind",
    }


@pytest.fixture
def page_path(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    img = Image.new("RGB", (120, 160), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 60, 90], fill=(180, 30, 30))
    img.save(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(duration_s=0.5, fps=8)


def _noise_block(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def two_panel_page_path(tmp_path: Path) -> Path:
    """A real, detectably-two-panel page (see tests/test_panels.py's construction style) for

    exercising `run_pipeline(..., analysis_mode="panel")` end to end.
    """
    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = _noise_block(300, 300, seed=11)
    page[500:900, 0:300] = _noise_block(400, 300, seed=12)
    path = tmp_path / "two_panel_page.png"
    Image.fromarray(page).save(path)
    return path


# --- happy path -----------------------------------------------------------------------------


@requires_ffmpeg
def test_run_pipeline_end_to_end_produces_a_playable_video(page_path: Path, config, tmp_path: Path):
    out_dir = tmp_path / "out"
    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision(), _static_decision()]),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
    )

    assert isinstance(result, PipelineRunResult)
    assert result.primary_object.motion_type == MotionType.PRIMARY
    assert result.grounding.object_id == result.primary_object.object_id
    assert result.segmentation.object_id == result.primary_object.object_id
    assert result.render.output_path.exists()
    assert result.render.frame_count == config.fps * config.duration_s

    # artifact creation: the frame sequence is kept per the Phase 3.1 brief
    frame_files = sorted((out_dir / "frames").glob("frame_*.png"))
    assert len(frame_files) == result.render.frame_count


@requires_ffmpeg
def test_run_pipeline_loads_and_unloads_grounding_and_segmentation_clients(
    page_path: Path, config, tmp_path: Path
):
    grounding_client = FakeGroundingClient()
    run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision()]),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    # GPU-memory hygiene ("GPU Awareness" in docs/architecture.md): the orchestrator must
    # release each model-backed client right after its stage runs, not hold it for the whole
    # pipeline run.
    assert grounding_client.loaded is True
    assert grounding_client.unloaded is True


# --- Phase 3.2: grounding-candidate validation ---------------------------------------------


@requires_ffmpeg
def test_run_pipeline_accepts_the_only_candidate_when_it_passes_validation(
    page_path: Path, config, tmp_path: Path
):
    """The plain "correct candidate accepted" case, made explicit at the orchestrator level

    (see tests/test_validation.py for the underlying `validate_target` unit behavior).
    """
    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision()]),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert len(result.validation_attempts) == 1
    assert result.validation_attempts[0].accepted is True
    assert result.validation_attempts[0].candidate_rank == 0


@requires_ffmpeg
def test_run_pipeline_tries_the_next_ranked_grounding_candidate_when_the_first_fails_validation(
    page_path: Path, config, tmp_path: Path
):
    """"Attempt another ranked grounding candidate if available" (Phase 3.2 failure policy) --

    the first-ranked (highest-score) box fails semantic validation, so the orchestrator must
    fall through to the second-ranked box from the SAME `detect()` call, not fail the run.
    """
    grounding_client = FakeGroundingClient(boxes=[(10, 10, 60, 90), (15, 15, 55, 85)])
    vlm_client = ValidationSequenceVLMClient([_primary_decision()], reject_first_n=1)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert len(result.validation_attempts) == 2
    assert result.validation_attempts[0].accepted is False
    assert result.validation_attempts[0].candidate_rank == 0
    assert result.validation_attempts[1].accepted is True
    assert result.validation_attempts[1].candidate_rank == 1
    assert result.grounding.bbox.as_xyxy() == (15, 15, 55, 85)
    assert result.render.output_path.exists()


def test_run_pipeline_raises_stage_validation_when_every_grounding_candidate_fails(
    page_path: Path, config, tmp_path: Path
):
    """"Never silently animate an unvalidated candidate" -- when every ranked grounding

    candidate fails validation, the run must fail outright (stage="validation"), not fall
    back to the best-scoring-but-rejected one.
    """
    grounding_client = FakeGroundingClient(boxes=[(10, 10, 60, 90), (15, 15, 55, 85)])
    vlm_client = ValidationSequenceVLMClient([_primary_decision()], reject_first_n=99)

    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "validation"
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_rejects_semantically_wrong_candidate_even_at_high_grounding_score(
    page_path: Path, config, tmp_path: Path
):
    """Real Phase 3.1 finding this stage exists to catch: a high-scoring, in-bounds detection

    is not automatically trusted -- a candidate the VLM says does not depict the target is
    rejected regardless of its grounding score.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_primary_decision()], verification_matches=False),
            grounding_client=FakeGroundingClient(),  # single box, score 0.9 -- high confidence
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "validation"


def test_run_pipeline_flag_banner_historical_regression_still_rejected_by_semantics(
    page_path: Path, config, tmp_path: Path
):
    """Explicit, named regression guard for the real Phase 3.1 historical failure (a

    "flag_banner" candidate whose crop is actually a face/dialogue box, see
    docs/decisions/0006-grounding-target-validation.md) -- must remain REJECTed by the
    semantic check, unaffected by adding the new Phase 3.3.1 geometry check alongside it.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient(
                [_primary_decision("flag_banner")], verification_matches=False
            ),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "validation"


# --- Phase 3.3.1: transform-aware geometric validation --------------------------------------
#
# Real motivating defect (Phase 3.3 real E2E run, eval_weapon_effects.png): a candidate passed
# semantic validation ("yes, this crop shows a weapon") but its bbox covered nearly the entire
# action panel; the plan's `rotate` transform then visibly swung the whole panel, not the
# weapon, producing torn black-wedge artifacts. See validation/transform_geometry.py and
# docs/decisions/0008-transform-aware-target-validation.md.


@requires_ffmpeg
def test_run_pipeline_tries_next_candidate_when_first_fails_geometry_not_semantics(
    page_path: Path, config, tmp_path: Path
):
    """"Attempt another ranked grounding candidate if available" (Phase 3.2 failure policy)

    must also apply when the FIRST candidate is rejected by the NEW transform-geometry check,
    not only a semantic mismatch -- the retry loop is agnostic to WHY a candidate was rejected.
    """
    oversized_box = (5, 5, 115, 155)  # ~86% of the 120x160 test image -- fails ROTATE's 15% cap
    small_box = (20, 20, 50, 50)  # ~4.7% of the image, well clear of every edge -- passes
    grounding_client = FakeGroundingClient(boxes=[oversized_box, small_box])
    vlm_client = FakeVLMClient([_primary_decision("weapon")])  # verification_matches=True default

    result = run_pipeline(
        page_path,
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert len(result.validation_attempts) == 2
    assert result.validation_attempts[0].accepted is False
    assert result.validation_attempts[0].semantic_match is True
    assert result.validation_attempts[0].transform_compatible is False
    assert result.validation_attempts[1].accepted is True
    assert result.validation_attempts[1].transform_compatible is True
    assert result.grounding.bbox.as_xyxy() == small_box
    assert result.render.output_path.exists()


def test_run_pipeline_fallback_plan_can_be_rejected_by_geometry_check(
    page_path: Path, config, tmp_path: Path
):
    """"Never silently animate an unvalidated candidate" (Phase 3.2) now also covers geometric

    safety, not just semantic correctness -- a human-authored fallback plan whose grounded
    region is semantically right but geometrically unsafe for its transform_kind must still
    fail the run, exactly like a semantic mismatch already does
    (test_run_pipeline_fallback_plan_can_still_be_rejected_by_validation).
    """
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
    )

    w, h = Image.open(page_path).size
    plan = AnimationPlan(
        source=SourceImage(path=str(page_path), width=w, height=h),
        panels=[PanelPlan(panel_id="panel_1", bbox=BBox(x=0, y=0, width=1, height=1))],
        objects=[
            ObjectPlan(
                object_id="obj_weapon",
                panel_id="panel_1",
                semantic_label="weapon",
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.ROTATE,
                    amplitude=8.0,
                    speed=1.0,
                    easing=Easing.SINE,
                    pivot=PivotSpec(x=0.5, y=1.0, reference="object_bbox"),
                ),
            )
        ],
        loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
    )
    oversized_box = (5, 5, 115, 155)  # ~86% of the 120x160 test image -- fails ROTATE's 15% cap
    grounding_client = FakeGroundingClient(box=oversized_box)
    # verification_matches=True -- the fallback candidate IS semantically correct; only its
    # geometry is unsafe, isolating this test from the already-covered semantic-rejection case.
    vlm_client = FakeVLMClient([], verification_matches=True)

    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
            plan=plan,
        )
    assert excinfo.value.stage == "validation"
    assert not (tmp_path / "out" / "output.mp4").exists()


# --- Phase 3.3: panel-aware analysis_mode ---------------------------------------------------


@requires_ffmpeg
def test_run_pipeline_analysis_mode_panel_produces_a_valid_run(
    two_panel_page_path: Path, config, tmp_path: Path
):
    """`analysis_mode="panel"` must wire real panel detection into the same orchestrator path

    -- grounding/validation/segmentation/animation/rendering are all identical to the
    page-level path (see docs/decisions/0007-panel-aware-analysis.md's explicit decoupling).
    """
    result = run_pipeline(
        two_panel_page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision(), _static_decision()]),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        analysis_mode="panel",
    )

    assert isinstance(result, PipelineRunResult)
    assert len(result.plan.panels) >= 1
    assert result.primary_object.motion_type == MotionType.PRIMARY
    assert result.render.output_path.exists()


def test_run_pipeline_analysis_mode_panel_still_rejects_semantically_wrong_candidate(
    two_panel_page_path: Path, config, tmp_path: Path
):
    """Regression guard: the panel-aware analysis path must not bypass Phase 3.2's

    grounding-target validation gate -- this mirrors
    `test_run_pipeline_rejects_semantically_wrong_candidate_even_at_high_grounding_score`
    (the real Phase 3.1 historical false-grounding failure this stage exists to catch, see
    docs/decisions/0006-grounding-target-validation.md), but through `analysis_mode="panel"`.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            two_panel_page_path,
            config,
            vlm_client=FakeVLMClient(
                [_primary_decision(), _static_decision()], verification_matches=False
            ),
            grounding_client=FakeGroundingClient(),  # single box, score 0.9 -- high confidence
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
            analysis_mode="panel",
        )
    assert excinfo.value.stage == "validation"
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_analysis_mode_defaults_to_panel_level(
    page_path: Path, config, tmp_path: Path
):
    """`analysis_mode` defaults to `"panel"` as of Phase 10 (see

    docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md) -- real Phase 9
    evidence (docs/phase9-results.md section 5.3: end_to_end_completion_rate 20%->60%,
    grounding_success_rate 50%->100%, ERROR outcomes 5->0) and a real Phase 9/10 mid-cycle
    visual defect traced to page-level analysis specifically (`realworld_marika_love_meter`,
    docs/phase10-results.md) together superseded Phase 3.3's original "default stays
    page-level" acceptance criterion. Every pre-existing caller/test that explicitly passes
    `analysis_mode="page"` is unaffected (see the sibling test below).
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_static_decision(), _static_decision("other")]),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    # the panel-aware all-STATIC error message ("...across every analyzed panel", not the plain
    # page-level one) proves the default path actually taken was panel-aware analysis
    assert excinfo.value.stage == "analysis"
    assert "every object STATIC across every analyzed panel" in excinfo.value.detail


def test_run_pipeline_analysis_mode_page_still_available_explicitly(
    page_path: Path, config, tmp_path: Path
):
    """`analysis_mode="page"` (Phase 3.1/3.2's original default) remains fully available and

    behaviorally unchanged for any caller that asks for it explicitly -- only the *default*
    changed in Phase 10, not the page-level path itself.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_static_decision(), _static_decision("other")]),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
            analysis_mode="page",
        )
    assert excinfo.value.stage == "analysis"
    assert excinfo.value.detail.startswith("VLM marked every object STATIC --")


# --- controlled-fallback plan override (Phase 3.1 failure policy escape hatch) -------------


class ExplodingVLMClient:
    """Proves the fallback path genuinely skips the ANALYSIS stage's VLM call.

    Phase 3.2's validation stage and Phase 12's mask_semantics stage still legitimately call
    the VLM (cheap crop-verification checks, not a full-page analysis call, see
    `_VALIDATION_PROMPT_MARKER`/`_MASK_SEMANTICS_PROMPT_MARKER`) even on the fallback path --
    "never silently animate an unvalidated candidate" applies to a human-authored fallback plan
    too, not only to automatic analysis output. This fake accepts both of those and only
    explodes on an analysis-stage call, so it still proves what its name says (analysis is
    skipped) without a false failure from either deliberate validation call.
    """

    def generate(self, image, prompt: str) -> str:
        if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
            return _fake_mask_semantics_response(True)
        if _VALIDATION_PROMPT_MARKER in prompt:
            return json.dumps(
                {"matches": True, "confidence": 0.9, "reason": "fallback target confirmed"}
            )
        raise AssertionError(
            "the analysis-stage VLM call must not happen when an explicit plan is supplied"
        )


@requires_ffmpeg
def test_run_pipeline_with_explicit_plan_skips_analysis_vlm_call(
    page_path: Path, config, tmp_path: Path
):
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

    w, h = Image.open(page_path).size
    plan = AnimationPlan(
        source=SourceImage(path=str(page_path), width=w, height=h),
        panels=[PanelPlan(panel_id="panel_1", bbox=BBox(x=0, y=0, width=1, height=1))],
        objects=[
            ObjectPlan(
                object_id="obj_banner",
                panel_id="panel_1",
                semantic_label="hanging_banner",
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.TRANSLATE,
                    direction=Vector2(x=1.0, y=0.0),
                    amplitude=0.02,
                    speed=1.0,
                    easing=Easing.EASE_IN_OUT,
                    pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
                ),
            )
        ],
        loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
    )

    result = run_pipeline(
        page_path,
        config,
        vlm_client=ExplodingVLMClient(),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        plan=plan,
    )

    assert result.plan is plan
    assert result.primary_object.object_id == "obj_banner"
    assert result.render.output_path.exists()
    # the fallback candidate still went through real Phase 3.2 validation -- it wasn't
    # rubber-stamped just because a human supplied the plan (see ExplodingVLMClient's docstring)
    assert len(result.validation_attempts) == 1
    assert result.validation_attempts[0].accepted is True


@requires_ffmpeg
def test_run_pipeline_fallback_plan_can_still_be_rejected_by_validation(
    page_path: Path, config, tmp_path: Path
):
    """"Preserve the existing controlled fallback" does not mean "skip validation for it" --

    a human-authored fallback plan whose grounded region fails semantic validation must still
    fail the run, not silently animate (see the Phase 3.2 acceptance criterion: "never
    silently animate an unvalidated candidate").
    """
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

    w, h = Image.open(page_path).size
    plan = AnimationPlan(
        source=SourceImage(path=str(page_path), width=w, height=h),
        panels=[PanelPlan(panel_id="panel_1", bbox=BBox(x=0, y=0, width=1, height=1))],
        objects=[
            ObjectPlan(
                object_id="obj_banner",
                panel_id="panel_1",
                semantic_label="hanging_banner",
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.TRANSLATE,
                    direction=Vector2(x=1.0, y=0.0),
                    amplitude=0.02,
                    speed=1.0,
                    easing=Easing.EASE_IN_OUT,
                    pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
                ),
            )
        ],
        loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
    )

    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([], verification_matches=False),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
            plan=plan,
        )
    assert excinfo.value.stage == "validation"
    assert not (tmp_path / "out" / "output.mp4").exists()


# --- failure propagation (no ffmpeg needed -- these fail before rendering) -----------------


def test_run_pipeline_propagates_grounding_failure(page_path: Path, config, tmp_path: Path):
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_primary_decision()]),
            grounding_client=FailingGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "grounding"
    # a failed stage must not leave a video behind that looks like a successful run
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_propagates_analysis_all_static_failure(
    page_path: Path, config, tmp_path: Path
):
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_static_decision(), _static_decision("other")]),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "analysis"


# --- Phase 4: multi-object layer decomposition --------------------------------------------


def test_run_pipeline_animates_primary_and_a_successful_secondary_object(
    page_path: Path, config, tmp_path: Path
):
    """The core new Phase 4 capability: a plan with a real SECONDARY candidate alongside the

    PRIMARY animates both, not just the PRIMARY -- see
    docs/decisions/0010-multi-object-layer-decomposition.md.
    """
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": (10, 10, 60, 90), "trailing_cloth": (70, 100, 110, 150)}
    )

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.primary_object.semantic_label == "hanging_banner"
    assert len(result.secondary_objects) == 1
    secondary = result.secondary_objects[0]
    assert secondary.object_plan.semantic_label == "trailing_cloth"
    assert secondary.object_plan.motion_type == MotionType.SECONDARY
    assert secondary.segmentation.mask.any()
    assert secondary.object_plan.object_id != result.primary_object.object_id


def test_run_pipeline_drops_a_secondary_that_fails_grounding_without_failing_the_run(
    page_path: Path, config, tmp_path: Path
):
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("ghost_object")]
    # only "hanging_banner" has a real box -- "ghost_object" detects nothing, like
    # FailingGroundingClient, but only for that one object.
    grounding_client = MultiObjectFakeGroundingClient({"hanging_banner": (10, 10, 60, 90)})

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.primary_object.semantic_label == "hanging_banner"
    assert result.secondary_objects == []
    assert (tmp_path / "out" / "output.mp4").exists()  # the run still completed

    # Phase 7.2.1: the drop is now visible, not just logged -- evaluation reporting needs to
    # see WHICH object was dropped and why, not just that secondary_objects came back empty.
    assert len(result.dropped_objects) == 1
    dropped = result.dropped_objects[0]
    assert dropped.object_plan.semantic_label == "ghost_object"
    assert dropped.failing_stage == "grounding"
    assert dropped.reason  # non-empty, human-readable


def test_run_pipeline_drops_a_secondary_that_fails_validation_without_failing_the_run(
    page_path: Path, config, tmp_path: Path
):
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": (10, 10, 60, 90), "trailing_cloth": (70, 100, 110, 150)}
    )
    # call #1 = PRIMARY's validation (accepted); call #2 = SECONDARY's (rejected, and every
    # later call would be too, but there's only one candidate here).
    vlm_client = RejectFromNthValidationVLMClient(decisions, reject_from_call=2)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.primary_object.semantic_label == "hanging_banner"
    assert result.secondary_objects == []
    assert (tmp_path / "out" / "output.mp4").exists()

    # Phase 7.2.1: same visibility for a validation-stage drop.
    assert len(result.dropped_objects) == 1
    dropped = result.dropped_objects[0]
    assert dropped.object_plan.semantic_label == "trailing_cloth"
    assert dropped.failing_stage == "validation"
    assert "rank=0" in dropped.reason


def test_run_pipeline_still_raises_for_primary_failure_even_with_a_secondary_present(
    page_path: Path, config, tmp_path: Path
):
    """A SECONDARY object succeeding must never mask a PRIMARY failure -- PRIMARY keeps its

    exact pre-Phase-4 failure policy regardless of what else is in the plan.
    """
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    # only the secondary's box exists -- PRIMARY's own grounding finds nothing.
    grounding_client = MultiObjectFakeGroundingClient({"trailing_cloth": (70, 100, 110, 150)})

    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient(decisions),
            grounding_client=grounding_client,
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "grounding"
    assert "hanging banner" in excinfo.value.detail  # PRIMARY's own prompt, not the secondary's
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_multi_object_bbox_and_mask_are_correctly_associated_per_object_id(
    page_path: Path, config, tmp_path: Path
):
    """Phase 5 identity-preservation regression: a PRIMARY + SECONDARY plan must never
    cross-associate one object's grounded bbox or segmented mask with the other's
    object_id. This is the exact failure mode named in
    docs/decisions/0010-multi-object-layer-decomposition.md's contract ("mask(A) ->
    animation(B)") -- the existing multi-object tests only assert `mask.any()`, which
    would not catch a positional/index mixup between two non-STATIC objects.
    """
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    box_primary = (10, 10, 60, 90)
    box_secondary = (70, 100, 110, 150)
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": box_primary, "trailing_cloth": box_secondary}
    )

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    secondary = result.secondary_objects[0]
    assert secondary.object_plan.object_id != result.primary_object.object_id

    # Each object keeps its own grounded box -- never the other's.
    assert result.grounding.bbox.as_xyxy() == box_primary
    assert secondary.grounding.bbox.as_xyxy() == box_secondary

    # Each object's mask is populated only inside ITS OWN box and is exactly zero inside
    # the OTHER object's box. `FakeSegmentationClient` fills a diamond inscribed in its box
    # (Phase 8.3, not a solid rectangle -- see `_region_mask`'s docstring), so "populated
    # inside its own box" is `.any()`, not `.all()`.
    px0, py0, px1, py1 = box_primary
    sx0, sy0, sx1, sy1 = box_secondary
    assert result.segmentation.mask[py0:py1, px0:px1].any()
    assert not result.segmentation.mask[sy0:sy1, sx0:sx1].any()
    assert secondary.segmentation.mask[sy0:sy1, sx0:sx1].any()
    assert not secondary.segmentation.mask[py0:py1, px0:px1].any()


def test_run_pipeline_multi_object_no_color_bleed_between_objects_across_the_loop(
    config, tmp_path: Path
):
    """Practical, visual counterpart to the bbox/mask identity test above, exercised
    through the REAL animation + layer + compositing + rendering path (not a hand-built
    `Layer` fixture, unlike tests/test_compositing.py's multi-layer tests). Two distinct,
    solidly-colored, non-overlapping regions get two distinct real transforms (translate
    vs. rotate, via the same semantic-label heuristics real pages hit --
    analysis/plan_builder.py's `_MOTION_HEURISTICS`).

    What this test actually catches: a GROSS spatial mis-association (e.g. object B's
    segmented region ending up composited near object A's location, or vice versa, or a
    hole-fill painting the wrong object's color). It deliberately does NOT catch a pure
    mask<->motion swap that keeps each mask at its own real, correct location (a rigid
    per-pixel transform stays spatially local to whichever mask array it's given,
    regardless of which object that mask "belongs to" conceptually, so a same-location
    color check can't distinguish "translate applied to the right mask" from "rotate
    applied to the right mask" by color alone) -- see the call-argument-level test below
    for that.
    """
    width, height = 200, 220
    box_primary = (10, 10, 60, 90)  # "character_hair" -> TRANSLATE (_MOTION_HEURISTICS)
    box_secondary = (110, 130, 160, 200)  # "raised_hand" -> ROTATE (_MOTION_HEURISTICS)
    red = (200, 30, 30)
    blue = (30, 30, 200)

    image = np.full((height, width, 3), (240, 240, 245), dtype=np.uint8)
    image[box_primary[1] : box_primary[3], box_primary[0] : box_primary[2]] = red
    image[box_secondary[1] : box_secondary[3], box_secondary[0] : box_secondary[2]] = blue
    page_path = tmp_path / "two_color_page.png"
    Image.fromarray(image).save(page_path)

    decisions = [_primary_decision("character_hair"), _secondary_decision("raised_hand")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"character_hair": box_primary, "raised_hand": box_secondary}
    )

    out_dir = tmp_path / "out"
    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
        # Explicit, not the (Phase 10) default: FakeVLMClient returns the same canned
        # `decisions` regardless of which panel it's asked about, so panel-mode's one-call-
        # per-detected-panel behavior would pool duplicate objects across arbitrary panel_ids
        # -- a fixture-simplification mismatch, not something this test is meant to exercise.
        analysis_mode="page",
    )
    primary_motion = result.primary_object.motion
    secondary_motion = result.secondary_objects[0].object_plan.motion
    assert primary_motion is not None and primary_motion.transform_kind == "translate"
    assert secondary_motion is not None and secondary_motion.transform_kind == "rotate"

    frame_paths = sorted((out_dir / "frames").glob("frame_*.png"))
    assert len(frame_paths) == result.plan.loop.frame_count
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths]

    # Generous padding around each box -- comfortably larger than either transform's real
    # peak displacement at this amplitude (a few px for translate, ~10px for rotate at the
    # box's far corner) but still leaving a clear gap between the two padded regions, so a
    # cross-contamination bug (not merely a slightly-imprecise transform) is what this
    # test would catch.
    pad = 15

    def _padded(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
        x0, y0, x1, y1 = box
        return (
            slice(max(0, y0 - pad), min(height, y1 + pad)),
            slice(max(0, x0 - pad), min(width, x1 + pad)),
        )

    def _contains_color(region: np.ndarray, color: tuple[int, int, int], atol: int = 10) -> bool:
        return bool(np.any(np.all(np.abs(region.astype(int) - np.array(color)) <= atol, axis=-1)))

    a_region, b_region = _padded(box_primary), _padded(box_secondary)
    for i, frame in enumerate(frames):
        assert not _contains_color(frame[a_region], blue), f"frame {i}: blue leaked into A's region"
        assert not _contains_color(frame[b_region], red), f"frame {i}: red leaked into B's region"

    # Sanity: both objects actually moved at some point (this test would be vacuous if
    # neither did) -- at least one frame differs from frame 0 within its own padded region.
    assert any(not np.array_equal(f[a_region], frames[0][a_region]) for f in frames[1:])
    assert any(not np.array_equal(f[b_region], frames[0][b_region]) for f in frames[1:])


def test_run_pipeline_drops_a_secondary_object_whose_mask_overlaps_an_already_accepted_one(
    config, tmp_path: Path
):
    """Phase 8.3, Defect A regression: `verified_action_1`'s real "duplicate silhouette"

    ghost was traced to two independently-accepted SECONDARY `character_hair` objects whose
    masks substantially overlapped, each animated with its own MotionSpec -- invisible at
    rest, a visible double-exposure once their motions diverged (see
    docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md). This is the exact invariant
    the fix (`_drop_overlapping_secondary_objects`) protects: two SECONDARY objects whose
    real segmentation masks overlap far above `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION` must
    never both reach the render -- the second one is dropped instead, non-fatally.
    """
    box_primary = (110, 130, 160, 200)  # "raised_hand" -- ROTATE, does not overlap either hair box
    box_hair_left = (10, 10, 60, 90)  # "left_hair" -> TRANSLATE (contains "hair")
    box_hair_right = (15, 15, 58, 88)  # "right_hair" -- nested almost entirely inside the above

    decisions = [
        _primary_decision("raised_hand"),
        _secondary_decision("left_hair"),
        _secondary_decision("right_hair"),
    ]
    grounding_client = MultiObjectFakeGroundingClient(
        {"raised_hand": box_primary, "left_hair": box_hair_left, "right_hair": box_hair_right}
    )

    out_dir = tmp_path / "out"
    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
    )

    # The run itself still succeeds -- an overlap conflict between two SECONDARY objects is a
    # non-fatal drop, exactly like a grounding/validation failure for a non-PRIMARY object.
    assert result.render.output_path.exists()
    assert result.primary_object.semantic_label == "raised_hand"

    # Exactly one of the two overlapping hair objects made it into the render...
    assert len(result.secondary_objects) == 1
    kept_label = result.secondary_objects[0].object_plan.semantic_label
    assert kept_label == "left_hair"  # first-encountered-in-plan-order wins, deterministically

    # ...and the other was dropped, attributed to this exact new reason, not silently lost.
    assert len(result.dropped_objects) == 1
    dropped = result.dropped_objects[0]
    assert dropped.object_plan.semantic_label == "right_hair"
    assert dropped.failing_stage == "segmentation"
    assert "overlaps" in dropped.reason


def test_run_pipeline_keeps_two_secondary_objects_with_genuinely_distinct_masks(
    config, tmp_path: Path
):
    """Negative control for the overlap guard above: two real SECONDARY objects whose masks

    do NOT meaningfully overlap must both survive -- the fix must not become an accidental cap
    on multi-object rendering in general (ADR 0010's core Phase 4 guarantee).
    """
    box_primary = (10, 10, 60, 90)
    box_hair = (70, 10, 110, 60)  # "trailing_cloth" default label -- disjoint from box_hand
    box_hand = (70, 130, 110, 190)  # "raised_hand" -- disjoint from box_hair too

    decisions = [
        _primary_decision("character_hair"),
        _secondary_decision("trailing_cloth"),
        _secondary_decision("raised_hand"),
    ]
    grounding_client = MultiObjectFakeGroundingClient(
        {"character_hair": box_primary, "trailing_cloth": box_hair, "raised_hand": box_hand}
    )

    out_dir = tmp_path / "out"
    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
    )

    assert len(result.secondary_objects) == 2
    assert result.dropped_objects == []


def test_run_pipeline_keeps_two_secondary_objects_whose_bboxes_intersect_but_masks_do_not(
    config, tmp_path: Path
):
    """Stronger negative control than the one above (found by independent review): the

    previous negative control uses fully disjoint bboxes, which short-circuits at
    `_bbox_intersects` and never exercises the actual pixel-overlap arithmetic in
    `_mask_overlap_fraction`. Here `box_hair`/`box_hand`'s bboxes DO intersect (a real 10x10
    corner overlap), but `_region_mask`'s diamond shape tapers to a point at each bbox corner,
    so the actual mask intersection in that corner is 0 -- both objects must still survive.
    """
    box_primary = (10, 10, 60, 90)
    box_hair = (70, 10, 120, 90)  # "trailing_cloth" -- bbox-disjoint from box_hand
    box_hand = (110, 80, 160, 160)  # "raised_hand" -- bbox INTERSECTS box_hair at (110-120, 80-90)

    decisions = [
        _primary_decision("character_hair"),
        _secondary_decision("trailing_cloth"),
        _secondary_decision("raised_hand"),
    ]
    grounding_client = MultiObjectFakeGroundingClient(
        {"character_hair": box_primary, "trailing_cloth": box_hair, "raised_hand": box_hand}
    )

    out_dir = tmp_path / "out"
    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
    )

    assert len(result.secondary_objects) == 2
    assert result.dropped_objects == []


def test_run_pipeline_raises_stage_segmentation_when_primary_mask_hugs_a_bbox_edge(
    page_path: Path, config, tmp_path: Path
):
    """Phase 8.3, Defect B regression: a real SAM mask that hugs one edge of its own tight

    bbox (see `test_segment_object_raises_on_a_mask_that_hugs_one_bbox_edge` for the real
    evidence) is exactly the shape traced to `phase3_action_page`'s real "vertical seam"
    defect -- when that shape lands on the PRIMARY object, the run must fail outright (stage=
    "segmentation"), the same hard-failure policy every other PRIMARY-stage defect already
    gets, rather than silently rendering the defective video.
    """
    box_primary = (10, 10, 60, 90)
    grounding_client = FakeGroundingClient(box=box_primary)
    segmentation_client = MaskShapeControlledSegmentationClient(bad_boxes={box_primary})

    with pytest.raises(PipelineStageError) as exc_info:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_primary_decision()]),
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert exc_info.value.stage == "segmentation"
    assert "hugs" in exc_info.value.detail


def test_run_pipeline_drops_a_secondary_whose_mask_hugs_a_bbox_edge_without_failing_the_run(
    config, tmp_path: Path
):
    """Same real defect shape as above, but on a SECONDARY object -- must be dropped

    non-fatally (this is the pre-existing orchestration gap Phase 8.3 also found and fixed: the
    segmentation stage's per-object loop previously had no try/except at all, so a SECONDARY's
    segmentation failure used to fail the WHOLE run, contradicting this module's own documented
    policy).
    """
    box_primary = (10, 10, 60, 90)
    box_secondary = (70, 100, 110, 150)  # "raised_hand" -- gets the defective mask shape
    grounding_client = MultiObjectFakeGroundingClient(
        {"character_hair": box_primary, "raised_hand": box_secondary}
    )
    segmentation_client = MaskShapeControlledSegmentationClient(bad_boxes={box_secondary})

    decisions = [_primary_decision("character_hair"), _secondary_decision("raised_hand")]
    out_dir = tmp_path / "out"
    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=segmentation_client,
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
    )

    assert result.render.output_path.exists()
    assert result.secondary_objects == []
    assert len(result.dropped_objects) == 1
    dropped = result.dropped_objects[0]
    assert dropped.object_plan.semantic_label == "raised_hand"
    assert dropped.failing_stage == "segmentation"
    assert "hugs" in dropped.reason


def test_run_pipeline_multi_object_mask_and_motion_reach_the_right_object(
    page_path: Path, config, tmp_path: Path, monkeypatch
):
    """Direct wiring-level identity check for the two seams ADR 0010 explicitly names as the
    risk ('mask(A) -> animation(B)', 'reconstruction(A) -> layer(B)'):
    `pipeline.orchestrator.run_pipeline` must call `generate_transformed_layer` and
    `reconstruct_hidden_region` with each object's OWN mask/motion, never the other
    object's. A rigid per-pixel transform stays spatially local to whichever mask array
    it's given regardless of which object that mask conceptually belongs to, so a
    rendered-pixel check (see the color-bleed test above) cannot reliably distinguish a
    pure mask<->motion swap between two objects -- this test instead asserts on the real
    call arguments at the actual stage boundary.
    """
    import manga_animation.pipeline.orchestrator as orch

    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    box_primary = (10, 10, 60, 90)
    box_secondary = (70, 100, 110, 150)
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": box_primary, "trailing_cloth": box_secondary}
    )

    animation_calls: list[tuple[np.ndarray, object]] = []
    real_generate_transformed_layer = orch.generate_transformed_layer

    def spy_generate_transformed_layer(image, mask, motion, panel_bbox, page_shape, t_frac, **kw):
        animation_calls.append((mask.copy(), motion))
        return real_generate_transformed_layer(
            image, mask, motion, panel_bbox, page_shape, t_frac, **kw
        )

    monkeypatch.setattr(orch, "generate_transformed_layer", spy_generate_transformed_layer)

    reconstruction_calls: list[tuple[str, np.ndarray]] = []
    real_reconstruct_hidden_region = orch.reconstruct_hidden_region

    def spy_reconstruct_hidden_region(
        image, original_mask, transformed_masks, client, *, object_id, model_id
    ):
        reconstruction_calls.append((object_id, original_mask.copy()))
        return real_reconstruct_hidden_region(
            image,
            original_mask,
            transformed_masks,
            client,
            object_id=object_id,
            model_id=model_id,
        )

    monkeypatch.setattr(orch, "reconstruct_hidden_region", spy_reconstruct_hidden_region)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    secondary = result.secondary_objects[0]
    frame_count = result.plan.loop.frame_count
    # One call per frame per object -- PRIMARY's whole loop first, then SECONDARY's (the
    # orchestrator's animation stage loops `for obj in animated_objects: for i in
    # range(frame_count): ...`), so the calls split cleanly into two contiguous blocks.
    assert len(animation_calls) == 2 * frame_count
    primary_calls = animation_calls[:frame_count]
    secondary_calls = animation_calls[frame_count:]

    for mask_seen, motion_seen in primary_calls:
        np.testing.assert_array_equal(mask_seen, result.segmentation.mask)
        assert motion_seen is result.primary_object.motion
        assert not np.array_equal(mask_seen, secondary.segmentation.mask)
    for mask_seen, motion_seen in secondary_calls:
        np.testing.assert_array_equal(mask_seen, secondary.segmentation.mask)
        assert motion_seen is secondary.object_plan.motion
        assert not np.array_equal(mask_seen, result.segmentation.mask)

    assert len(reconstruction_calls) == 2
    reconstruction_masks_by_object_id = dict(reconstruction_calls)
    np.testing.assert_array_equal(
        reconstruction_masks_by_object_id[result.primary_object.object_id], result.segmentation.mask
    )
    np.testing.assert_array_equal(
        reconstruction_masks_by_object_id[secondary.object_plan.object_id],
        secondary.segmentation.mask,
    )


# --- Phase 7.1.1: multi-object E2E encode/decode regression --------------------------------


@requires_ffmpeg
def test_run_pipeline_multi_object_e2e_encode_decode_regression(config, tmp_path: Path):
    """Phase 7.1.1: a deterministic multi-object scenario through the REAL render/encode path,
    decoded back from the actual .mp4 on disk (not the intermediate frame PNGs, and not a
    mocked render) -- extends
    `test_run_pipeline_multi_object_no_color_bleed_between_objects_across_the_loop`'s coverage
    (which only checks the pre-encode frame PNGs) with a genuine decode-verification pass,
    matching what a real evaluation/QA consumer would actually observe. Fake VLM/grounding/
    segmentation/reconstruction clients; real animation/compositing/rendering/ffmpeg encode
    and a real `cv2.VideoCapture` decode. No GPU needed.
    """
    import cv2

    width, height = 200, 220
    box_primary = (10, 10, 60, 90)  # "character_hair" -> TRANSLATE (_MOTION_HEURISTICS)
    box_secondary = (110, 130, 160, 200)  # "raised_hand" -> ROTATE (_MOTION_HEURISTICS)
    static_box = (10, 130, 60, 200)  # touched by neither object -- must stay unchanged
    red = (200, 30, 30)
    blue = (30, 30, 200)
    green = (30, 180, 30)

    image = np.full((height, width, 3), (240, 240, 245), dtype=np.uint8)
    image[box_primary[1] : box_primary[3], box_primary[0] : box_primary[2]] = red
    image[box_secondary[1] : box_secondary[3], box_secondary[0] : box_secondary[2]] = blue
    image[static_box[1] : static_box[3], static_box[0] : static_box[2]] = green
    page_path = tmp_path / "two_color_page.png"
    Image.fromarray(image).save(page_path)

    decisions = [_primary_decision("character_hair"), _secondary_decision("raised_hand")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"character_hair": box_primary, "raised_hand": box_secondary}
    )

    out_dir = tmp_path / "out"
    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient(decisions),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=out_dir,
        # Explicit, not the (Phase 10) default -- see the sibling color-bleed test's own
        # comment for why: FakeVLMClient isn't panel-crop-aware.
        analysis_mode="page",
    )

    # -- successful render, expected frame count/resolution -----------------------------------
    assert result.render.output_path.exists()
    assert result.render.frame_count == result.plan.loop.frame_count
    assert result.render.resolution == (width, height)

    # -- seamless loop verification ------------------------------------------------------------
    assert result.render.seamless_loop_verified is True

    # -- object identity preservation ----------------------------------------------------------
    assert result.primary_object.semantic_label == "character_hair"
    assert len(result.secondary_objects) == 1
    assert result.secondary_objects[0].object_plan.semantic_label == "raised_hand"
    assert result.secondary_objects[0].object_plan.object_id != result.primary_object.object_id

    # -- decode the ACTUAL encoded .mp4 for real, independent of render()'s own internal
    # validation, to prove the file on disk really contains what's expected ------------------
    cap = cv2.VideoCapture(str(result.render.output_path))
    decoded_frames = []
    ok, frame = cap.read()
    while ok:
        decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    assert len(decoded_frames) == result.render.frame_count

    pad = 15  # see test_run_pipeline_multi_object_no_color_bleed... for why this is generous
    # but still leaves a clear gap between the two padded regions.

    def _padded(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
        x0, y0, x1, y1 = box
        return (
            slice(max(0, y0 - pad), min(height, y1 + pad)),
            slice(max(0, x0 - pad), min(width, x1 + pad)),
        )

    def _contains_color(region: np.ndarray, color: tuple[int, int, int], atol: int = 12) -> bool:
        return bool(np.any(np.all(np.abs(region.astype(int) - np.array(color)) <= atol, axis=-1)))

    a_region, b_region = _padded(box_primary), _padded(box_secondary)
    sx0, sy0, sx1, sy1 = static_box
    for i, frame in enumerate(decoded_frames):
        # -- static-region preservation (H.264 is lossy -- a flat solid patch stays close to
        # its source color, not bit-exact; a real compositing bug would show a gross drift,
        # not a small quantization delta) ------------------------------------------------------
        static_patch = frame[sy0:sy1, sx0:sx1].astype(int)
        assert np.abs(static_patch - np.array(green)).mean() < 8, (
            f"frame {i}: static region drifted from its source color -- possible "
            "compositing/reconstruction leak into an untouched region"
        )
        # -- no cross-object color/mask contamination -------------------------------------------
        assert not _contains_color(frame[a_region], blue), f"frame {i}: blue leaked into A's region"
        assert not _contains_color(frame[b_region], red), f"frame {i}: red leaked into B's region"

    # Sanity: both objects actually moved (this test would be vacuous otherwise).
    assert any(
        not np.array_equal(f[a_region], decoded_frames[0][a_region]) for f in decoded_frames[1:]
    )
    assert any(
        not np.array_equal(f[b_region], decoded_frames[0][b_region]) for f in decoded_frames[1:]
    )


# --- Phase 7.1.2: whole-pipeline determinism regression -------------------------------------


@requires_ffmpeg
def test_run_pipeline_is_deterministic_for_identical_fake_inputs(
    page_path: Path, config, tmp_path: Path
):
    """Phase 7.1.2: identical deterministic/fake inputs through the WHOLE orchestration path
    (analysis -> grounding -> validation -> segmentation -> animation -> reconstruction ->
    compositing -> rendering) must produce byte-identical composited frames, run to run.

    This tests DETERMINISTIC PIPELINE CODE ONLY -- every client here is a fake, deterministic
    stand-in for a real model. It says nothing about, and does not attempt to claim, real VLM
    determinism -- see docs/decisions/0009-evaluation-ground-truth-integrity.md's real,
    opposite finding (Qwen2.5-VL is NOT reproducibly deterministic run to run on live GPU
    hardware, even with forced-greedy decoding) -- that finding is untouched by this test.
    """
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    boxes_by_label = {"hanging_banner": (10, 10, 60, 90), "trailing_cloth": (70, 100, 110, 150)}

    def _run(out_dir: Path) -> PipelineRunResult:
        return run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient(list(decisions)),
            grounding_client=MultiObjectFakeGroundingClient(dict(boxes_by_label)),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=out_dir,
        )

    result_a = _run(tmp_path / "run_a")
    result_b = _run(tmp_path / "run_b")

    # Plan-level determinism -- not just pixels.
    assert result_a.primary_object.semantic_label == result_b.primary_object.semantic_label
    assert result_a.primary_object.motion == result_b.primary_object.motion
    assert result_a.grounding.bbox.as_xyxy() == result_b.grounding.bbox.as_xyxy()
    assert len(result_a.secondary_objects) == len(result_b.secondary_objects) == 1
    assert (
        result_a.secondary_objects[0].object_plan.motion
        == result_b.secondary_objects[0].object_plan.motion
    )

    # Pixel-level determinism -- the actual composited frame sequence written before encoding.
    frames_a = sorted((tmp_path / "run_a" / "frames").glob("frame_*.png"))
    frames_b = sorted((tmp_path / "run_b" / "frames").glob("frame_*.png"))
    assert len(frames_a) == len(frames_b) == result_a.plan.loop.frame_count
    for fa, fb in zip(frames_a, frames_b, strict=True):
        arr_a = np.asarray(Image.open(fa).convert("RGB"))
        arr_b = np.asarray(Image.open(fb).convert("RGB"))
        np.testing.assert_array_equal(arr_a, arr_b)


# --- Phase 7.1.3: panel-aware regression on real phase3_action_page.png geometry ------------

_PHASE3_ACTION_PAGE = Path(__file__).resolve().parents[1] / "examples" / "phase3_action_page.png"

requires_phase3_action_page = pytest.mark.skipif(
    not _PHASE3_ACTION_PAGE.exists(),
    reason=(
        "examples/phase3_action_page.png is not present locally -- panel-aware real-geometry "
        "regression skipped, not fabricated (see docs/decisions/0002-local-canonical-source.md)"
    ),
)


@requires_ffmpeg
@requires_phase3_action_page
def test_run_pipeline_panel_aware_regression_on_real_action_page_geometry(config, tmp_path: Path):
    """Phase 7.1.3: deterministic regression coverage using `phase3_action_page.png`'s REAL
    720x5062 dimensions and REAL detected panel geometry (`analysis/panels.py::detect_panels`,
    no VLM call) through the panel-aware grounding path (ADR 0011) -- fake VLM/grounding/
    segmentation/reconstruction clients, but real panel detection and real localized CV
    rendering against this project's real extreme-aspect-ratio evaluation page (the same page
    ADR 0011's own live-GPU evidence used).
    """
    from manga_animation.analysis.panels import detect_panels
    from manga_animation.pipeline.types import bbox_px_to_normalized
    from manga_animation.schemas.animation_plan import (
        AnimationPlan,
        Easing,
        LoopSpec,
        MotionSpec,
        MotionType,
        ObjectPlan,
        PanelPlan,
        PivotSpec,
        SourceImage,
        TransformKind,
    )

    image = np.asarray(Image.open(_PHASE3_ACTION_PAGE).convert("RGB"))
    height, width = image.shape[0], image.shape[1]
    # Documents the real, known geometry this regression targets (see ADR 0011's evidence table).
    assert (width, height) == (720, 5062)

    real_panels = detect_panels(image)
    assert len(real_panels) >= 1  # the real detector must find at least the whole-page fallback

    # Prefer a real detected gutter panel over the degenerate whole-page fallback when one
    # exists (matching ADR 0011's own real live-evidence panel, panel_01) -- falls back to
    # whatever detect_panels actually returns if this page's real detector output ever changes.
    gutter_panels = [p for p in real_panels if p.source == "gutter_xy_cut"]
    target_panel = gutter_panels[1] if len(gutter_panels) > 1 else real_panels[0]
    assert target_panel.bbox.width > 0 and target_panel.bbox.height > 0

    panel_bbox_norm = bbox_px_to_normalized(
        target_panel.bbox, page_width=width, page_height=height
    )
    plan = AnimationPlan(
        source=SourceImage(path=str(_PHASE3_ACTION_PAGE), width=width, height=height),
        panels=[PanelPlan(panel_id="real_panel", bbox=panel_bbox_norm)],
        objects=[
            ObjectPlan(
                object_id="obj_weapon",
                panel_id="real_panel",
                semantic_label="weapon",
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.ROTATE,
                    amplitude=8.0,
                    speed=1.0,
                    easing=Easing.SINE,
                    pivot=PivotSpec(x=0.5, y=1.0, reference="object_bbox"),
                ),
            )
        ],
        loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
    )

    # A small, in-bounds local box relative to the real panel's own real crop dimensions --
    # exercises the real local-to-page coordinate translation (ADR 0011) against real panel
    # geometry, not a synthetic placeholder. Sized/margined as fractions of the real panel so
    # it clears validation/transform_geometry.py's ROTATE edge-margin and area-fraction checks
    # regardless of which real panel detect_panels() happens to return.
    panel_w, panel_h = target_panel.bbox.width, target_panel.bbox.height
    margin_x, margin_y = panel_w // 4, panel_h // 4
    box_w, box_h = panel_w // 4, panel_h // 6
    local_box = (margin_x, margin_y, margin_x + box_w, margin_y + box_h)
    grounding_client = RecordingGroundingClient({"weapon": local_box})

    result = run_pipeline(
        _PHASE3_ACTION_PAGE,
        config,
        vlm_client=FakeVLMClient([], verification_matches=True),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        plan=plan,
    )

    # Grounding must have seen the real panel's own real crop, not the real 720x5062 full page.
    assert len(grounding_client.calls) == 1
    called_shape, called_prompt = grounding_client.calls[0]
    assert called_prompt == "weapon."
    assert called_shape == (target_panel.bbox.height, target_panel.bbox.width, 3)
    assert called_shape != (height, width, 3)

    # Translated back to real full-page coordinates -- offset by the real panel's own origin.
    expected_bbox = (
        local_box[0] + target_panel.bbox.x0,
        local_box[1] + target_panel.bbox.y0,
        local_box[2] + target_panel.bbox.x0,
        local_box[3] + target_panel.bbox.y0,
    )
    assert result.grounding.bbox.as_xyxy() == expected_bbox

    assert result.render.output_path.exists()
    assert result.render.resolution == (width, height)
    assert result.render.seamless_loop_verified is True


def test_select_primary_raises_when_plan_has_no_primary_object():
    from manga_animation.schemas.animation_plan import AnimationPlan, PanelPlan, SourceImage

    plan = AnimationPlan(
        source=SourceImage(path="x.png", width=100, height=100),
        panels=[PanelPlan(panel_id="panel_1", bbox={"x": 0, "y": 0, "width": 1, "height": 1})],
        objects=[],
    )
    with pytest.raises(PipelineStageError) as excinfo:
        _select_primary(plan, "x.png")
    assert excinfo.value.stage == "analysis"


# --- model resolution (config -> real source ids, no torch needed) -------------------------


def test_candidate_source_resolves_real_configured_models():
    config = load_config()  # the actual, real configs/default.yaml
    assert _candidate_source("vlm", config) == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert _candidate_source("grounding", config) == "IDEA-Research/grounding-dino-base"
    assert _candidate_source("segmentation", config) == "facebook/sam2.1-hiera-base-plus"


def test_candidate_source_raises_clearly_when_model_variants_missing():
    config = PipelineConfig(model_variants={})
    with pytest.raises(PipelineStageError) as excinfo:
        _candidate_source("vlm", config)
    assert "model_variants" in excinfo.value.detail


def test_build_default_clients_does_not_require_torch_installed():
    # Construction is lazy everywhere (see each client's docstring) -- only calling load()
    # needs torch/transformers, which are intentionally absent on this dev machine.
    config = load_config()
    vlm, grounding, segmentation, reconstruction = build_default_clients(config)
    assert vlm is not None
    assert grounding.model_id == "grounding-dino-swin-l"
    assert segmentation.model_id == "sam2.1-hiera-base"
    assert reconstruction is not None


# --- Phase 5.1: panel-aware grounding (docs/decisions/0011-panel-aware-grounding.md) --------


@requires_ffmpeg
def test_run_pipeline_analysis_mode_panel_grounds_the_object_on_its_own_panel_crop(
    two_panel_page_path: Path, config, tmp_path: Path
):
    """Step 7-E: a panel-associated object's grounding call must receive that panel's real

    crop, not the full page -- verified via a spy on the actual image shape Grounding DINO was
    given, not just on the final render succeeding. `two_panel_page_path` is a real 900x300
    page whose top panel (via real, unmocked `detect_panels()`) is exactly (0,0)-(300,408) --
    a 408x300 crop, strictly smaller than the 900x300 full page.
    """
    grounding_client = RecordingGroundingClient({"hanging_banner": (30, 30, 80, 80)})

    result = run_pipeline(
        two_panel_page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision("hanging_banner"), _static_decision()]),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        analysis_mode="panel",
    )

    full_page_shape = (900, 300, 3)
    assert len(grounding_client.calls) == 1  # the STATIC "background" object is never grounded
    called_shape, called_prompt = grounding_client.calls[0]
    assert called_prompt == "hanging banner."
    assert called_shape != full_page_shape
    assert called_shape == (408, 300, 3)  # the real top panel's own crop, not the full page
    assert result.primary_object.semantic_label == "hanging_banner"
    assert result.render.output_path.exists()


@requires_ffmpeg
def test_run_pipeline_page_mode_still_grounds_the_full_page(
    page_path: Path, config, tmp_path: Path
):
    """Step 7-D regression guard: default (page-level) analysis must keep grounding the whole

    page exactly as before Phase 5.1 -- page-level's synthetic (0,0,1,1) panel resolves to a
    region covering the entire image (see ADR 0011's "Fallback behavior"), so the crop Grounding
    DINO sees, and the returned bbox, must be pixel-identical to the pre-Phase-5.1 behavior.
    """
    grounding_client = RecordingGroundingClient({"hanging_banner": (10, 10, 60, 90)})

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision("hanging_banner"), _static_decision()]),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    full_page_shape = (160, 120, 3)  # page_path is a PIL (120, 160) WxH image -> array (H, W, 3)
    assert len(grounding_client.calls) == 1
    assert grounding_client.calls[0][0] == full_page_shape
    assert result.grounding.bbox.as_xyxy() == (10, 10, 60, 90)  # offset (0, 0) -- unchanged


@requires_ffmpeg
def test_run_pipeline_analysis_mode_panel_falls_back_to_full_page_when_no_real_panels_exist(
    page_path: Path, config, tmp_path: Path
):
    """Step 7-C: when real panel detection itself falls back to a single `fallback_full_page`

    candidate (no internal gutters on this simple synthetic page -- confirmed directly via
    `detect_panels()`), grounding must still see the whole page, not some degenerate crop.
    """
    grounding_client = RecordingGroundingClient({"hanging_banner": (10, 10, 60, 90)})

    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision("hanging_banner"), _static_decision()]),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        analysis_mode="panel",
    )

    full_page_shape = (160, 120, 3)
    assert grounding_client.calls[0][0] == full_page_shape
    assert result.plan.panels[0].bbox.x == 0.0
    assert result.plan.panels[0].bbox.width == 1.0


@requires_ffmpeg
def test_run_pipeline_grounds_two_panel_objects_on_distinct_crops_without_identity_leakage(
    page_path: Path, config, tmp_path: Path
):
    """Steps 7-B/F/G combined: two objects on two different panels must (B) each keep their

    own `object_id`/bbox association, (F) never leak crop-local coordinates downstream -- both
    final bboxes are only sensible as PAGE coordinates, and (G) never swap crops/boxes with each
    other, even though the fake grounding client is given the IDENTICAL local box for both
    labels -- only their different panel offsets can explain their different final positions.
    """
    from manga_animation.schemas.animation_plan import (
        AnimationPlan,
        BBox,
        LoopSpec,
        MotionSpec,
        MotionType,
        ObjectPlan,
        PanelPlan,
        SourceImage,
        TransformKind,
        Vector2,
    )

    w, h = Image.open(page_path).size  # (120, 160)
    plan = AnimationPlan(
        source=SourceImage(path=str(page_path), width=w, height=h),
        panels=[
            PanelPlan(panel_id="panel_top", bbox=BBox(x=0.0, y=0.0, width=1.0, height=0.5)),
            PanelPlan(panel_id="panel_bottom", bbox=BBox(x=0.0, y=0.5, width=1.0, height=0.5)),
        ],
        objects=[
            ObjectPlan(
                object_id="obj_banner",
                panel_id="panel_top",
                semantic_label="hanging_banner",
                confidence=0.9,
                motion_type=MotionType.PRIMARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.TRANSLATE,
                    direction=Vector2(x=0.0, y=1.0),
                    amplitude=0.05,
                ),
            ),
            ObjectPlan(
                object_id="obj_cloth",
                panel_id="panel_bottom",
                semantic_label="trailing_cloth",
                confidence=0.8,
                motion_type=MotionType.SECONDARY,
                motion=MotionSpec(
                    transform_kind=TransformKind.TRANSLATE,
                    direction=Vector2(x=0.0, y=1.0),
                    amplitude=0.05,
                ),
            ),
        ],
        loop=LoopSpec(duration_s=config.duration_s, fps=config.fps, seamless=True),
    )
    # Identical local box for BOTH objects -- only the panel offset can explain different
    # final page positions; a swap or a leaked local coordinate would be immediately visible.
    grounding_client = RecordingGroundingClient(
        {"hanging_banner": (5, 5, 25, 25), "trailing_cloth": (5, 5, 25, 25)}
    )
    vlm_client = FakeVLMClient([], verification_matches=True)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
        plan=plan,
    )

    # (E, incidentally) two distinct 80x120 crops (h/2=80 tall each), not the 160x120 full page.
    assert {call[0] for call in grounding_client.calls} == {(80, 120, 3)}
    assert len(grounding_client.calls) == 2

    # (B) identity: PRIMARY stayed obj_banner, SECONDARY stayed obj_cloth.
    assert result.primary_object.object_id == "obj_banner"
    assert len(result.secondary_objects) == 1
    secondary = result.secondary_objects[0]
    assert secondary.object_plan.object_id == "obj_cloth"

    # (F, G) each bbox is the identical local box translated by ITS OWN panel's page offset --
    # top panel offset (0, 0), bottom panel offset (0, 80) -- never crop-local, never swapped.
    assert result.grounding.bbox.as_xyxy() == (5, 5, 25, 25)
    assert secondary.grounding.bbox.as_xyxy() == (5, 85, 25, 105)


# --- Phase 12: post-segmentation semantic mask validation gate -----------------------------
#
# See docs/decisions/0018-semantic-mask-validation.md and docs/phase11-results.md section 6.4
# for the real defect this stage exists to catch: a mask that passes every existing geometric
# check (bbox coverage, edge-asymmetry, cross-object overlap) but whose actual pixel content
# does not match its semantic_label.


@requires_ffmpeg
def test_run_pipeline_renders_when_mask_semantics_accepts_the_primary_object(
    page_path: Path, config, tmp_path: Path
):
    result = run_pipeline(
        page_path,
        config,
        vlm_client=FakeVLMClient([_primary_decision()], mask_semantics_matches=True),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert result.render.output_path.exists()
    assert result.mask_semantics is not None
    assert result.mask_semantics.verdict == "accept"


def test_run_pipeline_raises_stage_mask_semantics_when_primary_mask_content_is_rejected(
    page_path: Path, config, tmp_path: Path
):
    """The real Phase 11 finding, end to end: a mask that is geometrically fine (passes

    grounding/validation/segmentation's own checks) but whose real content the VLM says does
    not match its label must fail the run, not silently reach the renderer -- the same
    "never silently animate an unvalidated candidate" policy grounding/validation already have,
    extended to the mask's own content instead of just its bounding box.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient([_primary_decision("cloth")], mask_semantics_matches=False),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "mask_semantics"
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_raises_stage_mask_semantics_on_primary_abstain(
    page_path: Path, config, tmp_path: Path
):
    """ABSTAIN is treated identically to REJECT for a PRIMARY object (fail-closed, per

    Workstream 6's "UNKNOWN -> REJECT" conservative default) -- a near-coin-flip VLM read must
    not silently pass a PRIMARY object through to the renderer either.
    """
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            vlm_client=FakeVLMClient(
                [_primary_decision()],
                mask_semantics_matches=True,
                mask_semantics_confidence=0.5,  # inside the [0.4, 0.6] abstain band
            ),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "mask_semantics"
    assert "ABSTAIN" in excinfo.value.detail


@requires_ffmpeg
def test_run_pipeline_drops_a_secondary_whose_mask_semantics_is_rejected_without_failing_the_run(
    config, tmp_path: Path
):
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": (10, 10, 60, 90), "trailing_cloth": (70, 100, 110, 150)}
    )

    class PerObjectMaskSemanticsVLMClient:
        """Accepts every prompt except a mask_semantics check for "trailing_cloth"."""

        def generate(self, image, prompt: str) -> str:
            if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
                matches = "trailing cloth" not in prompt
                return _fake_mask_semantics_response(matches)
            if _VALIDATION_PROMPT_MARKER in prompt:
                return json.dumps({"matches": True, "confidence": 0.9, "reason": "ok"})
            return json.dumps(decisions)

    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=PerObjectMaskSemanticsVLMClient(),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.render.output_path.exists()  # the run still completed
    assert result.primary_object.semantic_label == "hanging_banner"
    assert result.secondary_objects == []
    assert len(result.dropped_objects) == 1
    dropped = result.dropped_objects[0]
    assert dropped.object_plan.semantic_label == "trailing_cloth"
    assert dropped.failing_stage == "mask_semantics"
    assert dropped.reason  # non-empty, human-readable


@requires_ffmpeg
def test_run_pipeline_can_disable_the_mask_semantics_gate_via_config(
    page_path: Path, config, tmp_path: Path
):
    """`PipelineConfig.enable_semantic_mask_validation=False` skips the gate entirely -- a

    caller that has separately characterized this gate's real false-rejection rate for its own
    dataset can opt out deliberately (see docs/decisions/0018-semantic-mask-validation.md).
    """
    disabled_config = config.model_copy(update={"enable_semantic_mask_validation": False})

    result = run_pipeline(
        page_path,
        disabled_config,
        vlm_client=FakeVLMClient([_primary_decision()], mask_semantics_matches=False),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert result.render.output_path.exists()
    assert result.mask_semantics is None


@requires_ffmpeg
def test_run_pipeline_can_disable_the_mask_semantics_gate_for_a_secondary_object(
    config, tmp_path: Path
):
    """Same guarantee as the PRIMARY-object disable test above, extended to a SECONDARY --

    with the gate off, a mask that would otherwise be dropped must reach the render instead
    (independent adversarial QA review finding: the existing disable test only covered PRIMARY).
    """
    disabled_config = config.model_copy(update={"enable_semantic_mask_validation": False})
    decisions = [_primary_decision("hanging_banner"), _secondary_decision("trailing_cloth")]
    grounding_client = MultiObjectFakeGroundingClient(
        {"hanging_banner": (10, 10, 60, 90), "trailing_cloth": (70, 100, 110, 150)}
    )
    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        disabled_config,
        vlm_client=FakeVLMClient(decisions, mask_semantics_matches=False),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.render.output_path.exists()
    assert len(result.secondary_objects) == 1  # would have been dropped with the gate enabled
    assert result.secondary_objects[0].mask_semantics is None
    assert result.dropped_objects == []


def test_run_pipeline_drops_two_simultaneously_bad_secondary_objects(config, tmp_path: Path):
    """Two independent SECONDARY objects both failing mask_semantics in the same run must BOTH

    be dropped, not just the first -- guards against an early-exit/break bug in the per-object
    loop that a single-bad-object test (the one directly above) cannot catch (independent
    adversarial QA review finding).
    """
    decisions = [
        _primary_decision("hanging_banner"),
        _secondary_decision("trailing_cloth"),
        _secondary_decision("raised_hand"),
    ]
    grounding_client = MultiObjectFakeGroundingClient(
        {
            "hanging_banner": (10, 10, 60, 90),
            "trailing_cloth": (70, 10, 110, 60),
            "raised_hand": (70, 130, 110, 190),
        }
    )

    class BothSecondariesBadVLMClient:
        """Accepts PRIMARY; rejects at mask_semantics for BOTH secondary objects."""

        def generate(self, image, prompt: str) -> str:
            if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
                matches = "hanging banner" in prompt  # only PRIMARY passes
                return _fake_mask_semantics_response(matches)
            if _VALIDATION_PROMPT_MARKER in prompt:
                return json.dumps({"matches": True, "confidence": 0.9, "reason": "ok"})
            return json.dumps(decisions)

    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=BothSecondariesBadVLMClient(),
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.render.output_path.exists()  # PRIMARY alone still renders
    assert result.secondary_objects == []
    dropped_labels = {d.object_plan.semantic_label for d in result.dropped_objects}
    assert dropped_labels == {"trailing_cloth", "raised_hand"}
    assert all(d.failing_stage == "mask_semantics" for d in result.dropped_objects)


def test_run_pipeline_overlap_dropped_secondary_never_reaches_mask_semantics(
    config, tmp_path: Path
):
    """The cross-object overlap guard (`_drop_overlapping_secondary_objects`, Phase 8.3) must

    run BEFORE mask_semantics -- an object it already drops should never waste a real VLM call.
    Order is correct by inspection of orchestrator.py, but was previously unpinned by any test:
    the pre-existing overlap test used a `FakeVLMClient` with no call recorder, so a stage-order
    swap would have passed it identically (independent adversarial QA review finding). Uses the
    same real defect shape as
    `test_run_pipeline_drops_a_secondary_object_whose_mask_overlaps_an_already_accepted_one`.
    """
    box_primary = (110, 130, 160, 200)  # "raised_hand" -- ROTATE, doesn't overlap either hair box
    box_hair_left = (10, 10, 60, 90)  # "left_hair" -> TRANSLATE (contains "hair")
    box_hair_right = (15, 15, 58, 88)  # "right_hair" -- nested almost entirely inside the above

    decisions = [
        _primary_decision("raised_hand"),
        _secondary_decision("left_hair"),
        _secondary_decision("right_hair"),
    ]
    grounding_client = MultiObjectFakeGroundingClient(
        {"raised_hand": box_primary, "left_hair": box_hair_left, "right_hair": box_hair_right}
    )
    vlm_client = FakeVLMClient(decisions)  # mask_semantics_matches=True by default

    image = np.full((220, 200, 3), (240, 240, 245), dtype=np.uint8)
    page_path = tmp_path / "page.png"
    Image.fromarray(image).save(page_path)

    result = run_pipeline(
        page_path,
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert result.render.output_path.exists()
    assert len(result.dropped_objects) == 1
    assert result.dropped_objects[0].failing_stage == "segmentation"  # the overlap guard, not
    # mask_semantics -- confirms overlap-dropping happened first

    # Exactly 2 real mask_semantics calls (PRIMARY + the one surviving hair object) -- the
    # overlap-dropped "right_hair" must never have reached this stage at all.
    assert len(vlm_client.mask_semantics_prompts) == 2
    checked_labels = {
        "raised hand" if "raised hand" in p else ("left hair" if "left hair" in p else "?")
        for p in vlm_client.mask_semantics_prompts
    }
    assert checked_labels == {"raised hand", "left hair"}
