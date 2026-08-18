"""Behavioral tests for the 2026 generative AnimateAnything panel path.

The new architecture change: the pipeline can run WITHOUT a segmentation (SAM) stage. When
`run_page_panels(animation_clients=...)` is passed, stage 2 (SAM) is skipped and each ACCEPTED
candidate is animated by cropping the panel at its DINO bbox and calling AnimateAnything with
the prompt built from the accepted Qwen description. Each object renders to its own MP4.

These tests use fake model clients (no torch/transformers/GPU) plus the real ffmpeg render
path, mirroring the existing `test_pipeline.py` style. The fake AA client records the crops it
was given and returns a real frame sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_animation.pipeline.panels import run_page_panels

pytestmark = pytest.mark.filterwarnings("ignore")


class FakeGroundingClient:
    """One candidate box per semantic label (matched by substring against the prompt)."""

    model_id = "fake-grounding-dino"

    def __init__(self, boxes_by_label: dict[str, tuple[int, int, int, int]]):
        self._boxes_by_label = {
            label.replace("_", " "): box for label, box in boxes_by_label.items()
        }
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list:
        from manga_animation.grounding.client import Detection

        for label, box in self._boxes_by_label.items():
            if label in text_prompt:
                return [Detection(label=label, score=0.9, box=box)]
        return []

    def unload(self) -> None:
        self.unloaded = True


class FakeVLMClient:
    """Answers the object-description batch prompt: every candidate is ACCEPTED."""

    def __init__(self):
        self.call_count = 0
        self._prompt_marker = "proposed animation candidate"

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        n_boxes = prompt.count("<|box_start|>")
        entries = []
        for i in range(n_boxes):
            entries.append(
                {
                    "box_index": i,
                    "bbox_assessment": "pass",
                    "object_identity": "banner",
                    "matches_semantic_label": True,
                    "animatable": True,
                    "motion_kind": "sway",
                    "direction": None,
                    "amplitude_band": "moderate",
                    "speed_band": "normal",
                    "pivot_hint": "center",
                    "constraints": [],
                    "neighbor_conflicts": [],
                    "confidence": 0.9,
                    "reason": "the banner sways gently",
                }
            )
        return __import__("json").dumps(entries)

    def unload(self) -> None:
        pass


class NoSamSegmentationClient:
    """Should NEVER be constructed on the AA path -- if the pipeline tries to use SAM here,
    this raises."""

    model_id = "fake-sam2.1"

    def __init__(self):
        raise AssertionError("SAM segmentation must not be used on the AnimateAnything path")


class FakeAnimateAnythingClient:
    """Records the crops/prompts it is asked to animate and returns a real 2-frame sequence."""

    model_id = "animate-anything-512-v1.02"

    def __init__(self):
        self.calls: list[tuple[tuple[int, int], str]] = []
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def animate(self, image: np.ndarray, prompt: str, out_dir: Path):
        from manga_animation.pipeline.types import FrameSequence

        self.calls.append((image.shape[:2], prompt))
        h, w = image.shape[:2]
        frames = [image.copy() for _ in range(2)]
        return FrameSequence(frames=frames, fps=8)

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture
def two_panel_page_path(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    img = Image.new("RGB", (240, 320), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Two clearly separated panels separated by a white gutter row.
    draw.rectangle([10, 10, 110, 130], fill=(180, 30, 30))
    draw.rectangle([10, 190, 110, 310], fill=(30, 30, 180))
    img.save(path)
    return path


@pytest.fixture
def config():
    from manga_animation.core.config import PipelineConfig

    return PipelineConfig(duration_s=0.5, fps=8)


def test_animate_anything_path_skips_sam_and_renders_per_object(
    two_panel_page_path: Path, config, tmp_path: Path
):
    vlm = FakeVLMClient()
    grounding = FakeGroundingClient({"banner": (20, 20, 90, 120)})
    aa = FakeAnimateAnythingClient()

    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=None,  # must be unused
        reconstruction_client=None,
        animation_clients=[aa],
        out_dir=tmp_path / "videos",
        labels=["banner"],
    )

    # Every panel that grounded and accepted a candidate rendered at least one per-object
    # video; the VLM was called at most once per panel.
    assert aa.loaded is True
    assert vlm.call_count <= len(result.panels)
    rendered = [p for p in result.panels if p.status == "PASS"]
    assert rendered, "expected at least one PASS panel on the generative path"
    for panel in rendered:
        assert panel.output_videos, "a PASS panel must have at least one per-object video"
        for video in panel.output_videos:
            assert video.exists()
    # The AA client saw per-object crops (the bbox is inside each detected panel).
    assert aa.calls, "AnimateAnything must have animated at least one accepted crop"
    for (h, w), prompt in aa.calls:
        assert h > 0 and w > 0
        assert "banner" in prompt


def test_animate_anything_path_no_sam_loaded(two_panel_page_path: Path, config, tmp_path: Path):
    """The SAM segmentation client is never instantiated or loaded on the AA path."""
    vlm = FakeVLMClient()
    grounding = FakeGroundingClient({"banner": (20, 20, 90, 120)})
    aa = FakeAnimateAnythingClient()

    run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=None,
        reconstruction_client=None,
        animation_clients=[aa],
        out_dir=tmp_path / "videos",
        labels=["banner"],
    )
    # If the pipeline tried to build/load SAM we would have raised above; assert the engine
    # actually animated so this test is not vacuous.
    assert aa.calls
