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


class FakeVLMClient:
    def __init__(self, decisions: list[dict]):
        self._decisions = decisions

    def generate(self, image, prompt: str) -> str:
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
    model_id = "fake-grounding-dino"

    def __init__(self, box: tuple[int, int, int, int] = (10, 10, 60, 90)):
        self.box = box
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return [Detection(label="banner", score=0.9, box=self.box)]

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
