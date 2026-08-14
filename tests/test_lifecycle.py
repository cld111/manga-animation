"""Targeted tests for Phase 14's stage-level model lifecycle.

Two layers are protected here, without duplicating the existing behavioral coverage in
tests/test_pipeline.py:

1. `ModelStage` (src/manga_animation/pipeline/lifecycle.py) -- the context manager that owns
   one model client's residency. The GPU evidence this phase records (docs/phase14-results.md)
   showed the old unload path (`set model = None; empty_cache()` with no `gc.collect()`) left
   Qwen's ~16 GiB resident until an opportunistic cyclic-GC, racing the next load into a CUDA
   OOM. `ModelStage` must release deterministically on normal exit AND on exception.
2. Stage-level `run_page_panels` -- each model client is loaded exactly once per page (per
   stage), not once per panel, and a panel failing in an early stage cannot poison later
   panels or prevent the stage's cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_animation.core.config import PipelineConfig
from manga_animation.grounding.client import Detection
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.segmentation.client import MaskCandidate

pytestmark = pytest.mark.filterwarnings("ignore")

# Distinctive substrings of the real stage prompts, so fakes can tell analysis from
# validation from mask_semantics without threading extra parameters (same convention as
# tests/test_pipeline.py).
_VALIDATION_PROMPT_MARKER = "Does the image above show"
_MASK_SEMANTICS_PROMPT_MARKER = "Does the bright region show"


# --- ModelStage unit tests -------------------------------------------------------------------


class TrackingClient:
    """A client with visible load/unload counters (DINO/SAM/LaMa shape: explicit load())."""

    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1


class LazyVLMClient:
    """The VLM's real shape: no `load()` (it lazy-loads inside generate()); only unload()."""

    def __init__(self):
        self.unload_calls = 0

    def generate(self, image, prompt: str) -> str:
        return "{}"

    def unload(self) -> None:
        self.unload_calls += 1


def test_model_stage_loads_on_entry_and_unloads_on_normal_exit():
    client = TrackingClient()
    with ModelStage(client, name="stage"):
        assert client.load_calls == 1
        assert client.unload_calls == 0
    assert client.unload_calls == 1


def test_model_stage_still_unloads_when_the_stage_body_raises():
    client = TrackingClient()
    with pytest.raises(ValueError, match="boom"):
        with ModelStage(client, name="stage"):
            raise ValueError("boom")
    # The exception propagated, but the model was still released -- a failed panel must not
    # leave its model resident to poison the next panel (Phase 14 acceptance criterion 4).
    assert client.unload_calls == 1


def test_model_stage_handles_a_client_with_no_load_method():
    client = LazyVLMClient()
    with ModelStage(client, name="analysis"):
        assert client.unload_calls == 0  # lazy VLM is not force-loaded by the stage
    assert client.unload_calls == 1


def test_model_stage_handles_a_client_with_neither_load_nor_unload():
    with ModelStage(object(), name="trivial"):  # must not raise on enter/exit
        pass


def test_model_stage_rejects_reentrant_entry():
    stage = ModelStage(TrackingClient(), name="stage")
    stage.__enter__()
    try:
        with pytest.raises(RuntimeError):
            stage.__enter__()
    finally:
        stage.__exit__(None, None, None)


# --- stage-level run_page_panels -------------------------------------------------------------


class CountingGroundingClient:
    model_id = "fake-grounding-dino"

    def __init__(self, fail_first_n: int = 0):
        self.load_calls = 0
        self.unload_calls = 0
        self.detect_calls = 0
        self.fail_first_n = fail_first_n

    def load(self) -> None:
        self.load_calls += 1

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.detect_calls += 1
        if self.detect_calls <= self.fail_first_n:
            return []
        h, w = image.shape[:2]
        # Well inset from the crop edges and smaller than 35% of the reference region so the
        # transform-aware geometry check (mesh_warp bounds: 2% edge margin, 35% max area)
        # accepts it.
        return [Detection(label="object", score=0.9, box=(w // 4, h // 4, 3 * w // 4, 3 * h // 4))]

    def unload(self) -> None:
        self.unload_calls += 1


class CountingSegmentationClient:
    model_id = "fake-sam2.1"

    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[box.y0 : box.y1, box.x0 : box.x1] = 255
        return [MaskCandidate(mask=mask, iou_score=0.9)]

    def unload(self) -> None:
        self.unload_calls += 1


class CountingReconstructionClient:
    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def inpaint(self, image, hole_mask):
        return image

    def unload(self) -> None:
        self.unload_calls += 1


class StageLevelVLMClient:
    """Returns an animated PRIMARY decision for analysis, ACCEPTs validation and
    mask_semantics prompts, and counts unloads (its real shape: no load())."""

    def __init__(self):
        self.unload_calls = 0

    def generate(self, image, prompt: str) -> str:
        if _MASK_SEMANTICS_PROMPT_MARKER in prompt:
            return json.dumps(
                {
                    "mask_matches_object": True,
                    "confidence": 0.9,
                    "unexpected_content": [],
                    "reason": "fake mask semantics response",
                }
            )
        if _VALIDATION_PROMPT_MARKER in prompt:
            return json.dumps(
                {"matches": True, "confidence": 0.9, "reason": "fake validation response"}
            )
        return json.dumps(
            [
                {
                    "semantic_label": "hanging_banner",
                    "motion_type": "primary",
                    "confidence": 0.9,
                    "reason": "test fixture",
                    "motion_description": "sways left and right",
                }
            ]
        )

    def unload(self) -> None:
        self.unload_calls += 1


def _noise_block(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def two_panel_page_path(tmp_path: Path) -> Path:
    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = _noise_block(300, 300, seed=21)
    page[500:900, 0:300] = _noise_block(400, 300, seed=22)
    path = tmp_path / "two_panel_page.png"
    Image.fromarray(page).save(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(duration_s=0.5, fps=8)


def _requires_ffmpeg():
    import shutil

    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_loads_each_model_once_for_the_whole_page(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """Phase 14 stage-level lifecycle: a model is loaded once for all eligible panels, not
    once per panel. The old per-panel path loaded Grounding DINO / SAM / LaMa once per panel
    and entered the VLM analysis/validation/mask_semantics stages once per panel.
    """
    grounding = CountingGroundingClient()
    segmentation = CountingSegmentationClient()
    reconstruction = CountingReconstructionClient()
    vlm = StageLevelVLMClient()

    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=segmentation,
        reconstruction_client=reconstruction,
        out_dir=tmp_path / "videos",
    )

    assert len(result.panels) == 2
    assert all(panel.status == "PASS" for panel in result.panels)
    # One load/unload per model for the whole page (the stage owns residency).
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1
    assert segmentation.load_calls == 1
    assert segmentation.unload_calls == 1
    assert reconstruction.load_calls == 1
    assert reconstruction.unload_calls == 1
    # VLM: three stages (analysis, validation, semantic-mask), each releasing once -- not
    # three times per panel as the per-panel path did.
    assert vlm.unload_calls == 3


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_early_panel_failure_does_not_poison_later_panels_or_cleanup(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """Phase 14 acceptance criterion 4: a panel failing in an early stage (panel 1's grounding
    detects nothing) must not prevent panel 2 from processing, and the stage's model must still
    be released exactly once.
    """
    grounding = CountingGroundingClient(fail_first_n=1)
    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=StageLevelVLMClient(),
        grounding_client=grounding,
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )

    assert [panel.status for panel in result.panels] == ["REJECTED", "PASS"]
    assert result.panels[0].failure_stage == "grounding"
    assert grounding.detect_calls == 2  # panel 2 still got grounded
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1
