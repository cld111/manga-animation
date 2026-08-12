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


class FakeVLMClient:
    """Answers both the analysis-stage prompt (returns canned `decisions`) and the Phase 3.2

    validation-stage prompt (defaults to accepting every candidate) -- see
    `_VALIDATION_PROMPT_MARKER`. `verification_matches=False` lets a test make validation
    reject instead, without needing a second fake class.
    """

    def __init__(self, decisions: list[dict], *, verification_matches: bool = True):
        self._decisions = decisions
        self._verification_matches = verification_matches

    def generate(self, image, prompt: str) -> str:
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
    candidate" retry path deterministically.
    """

    def __init__(self, decisions: list[dict], *, reject_first_n: int = 0):
        self._decisions = decisions
        self._reject_first_n = reject_first_n
        self._validation_calls = 0

    def generate(self, image, prompt: str) -> str:
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
    PRIMARY first).
    """

    def __init__(self, decisions: list[dict], *, reject_from_call: int):
        self._decisions = decisions
        self._reject_from_call = reject_from_call
        self._validation_calls = 0

    def generate(self, image, prompt: str) -> str:
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


class FakeSegmentationClient:
    model_id = "fake-sam2.1"

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[0], image.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[box.y0 : box.y1, box.x0 : box.x1] = 255
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


def test_run_pipeline_analysis_mode_defaults_to_page_level(page_path: Path, config, tmp_path: Path):
    """`analysis_mode` defaults to `"page"` -- every pre-existing Phase 3.1/3.2 caller/test

    (which never passes `analysis_mode`) is unaffected by this phase (acceptance criterion #2).
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
    # the page-level all-STATIC error message (not a panel-aware one) proves the default path
    # was actually taken
    assert excinfo.value.stage == "analysis"
    assert "every object STATIC" in excinfo.value.detail


# --- controlled-fallback plan override (Phase 3.1 failure policy escape hatch) -------------


class ExplodingVLMClient:
    """Proves the fallback path genuinely skips the ANALYSIS stage's VLM call.

    Phase 3.2's validation stage still legitimately calls the VLM (a cheap crop-verification
    check, not a full-page analysis call, see `_VALIDATION_PROMPT_MARKER`) even on the
    fallback path -- "never silently animate an unvalidated candidate" applies to a
    human-authored fallback plan too, not only to automatic analysis output. This fake accepts
    validation-stage calls and only explodes on an analysis-stage one, so it still proves what
    its name says (analysis is skipped) without a false failure from the new, deliberate
    validation call.
    """

    def generate(self, image, prompt: str) -> str:
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
    # the OTHER object's box.
    px0, py0, px1, py1 = box_primary
    sx0, sy0, sx1, sy1 = box_secondary
    assert result.segmentation.mask[py0:py1, px0:px1].all()
    assert not result.segmentation.mask[sy0:sy1, sx0:sx1].any()
    assert secondary.segmentation.mask[sy0:sy1, sx0:sx1].all()
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
