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
