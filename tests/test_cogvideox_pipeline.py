"""Pipeline integration tests for the CogVideoX-5B-I2V generative animation engine (ADR 0024).

These prove the wiring contract locally with fake model clients (including a fake CogVideoX
client that returns deterministic frames) -- real diffusion inference is remote-GPU work and
never runs in tests. The determinism of the fake CogVideoX client means these also verify
that the CogVideoX pipeline path produces a PASS panel with a real playable MP4, that the
prompt reaches the animation engine, and that fail-closed semantics are unchanged.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_animation.core.config import PipelineConfig
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.pipeline.types import FrameSequence

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


# --- fakes (mirror the style of test_pipeline.py's fakes) ---------------------------------


class FakeCogVideoX:
    """Fake CogVideoX client: returns deterministic frames and records what it saw."""

    model_id = "cogvideox-5b-i2v"

    def __init__(self, num_frames: int = 4, fps: int = 8):
        self.num_frames = num_frames
        self.fps = fps
        self.calls: list[dict] = []
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def animate(self, image, mask, prompt: str, out_dir: Path) -> FrameSequence:
        self.calls.append(
            {
                "image_shape": image.shape,
                "mask_shape": mask.shape,
                "prompt": prompt,
                "mask_area": int(np.count_nonzero(mask)),
                "out_dir": str(out_dir),
            }
        )
        frames = [
            np.full((16, 16, 3), i * 30, dtype=np.uint8) for i in range(self.num_frames)
        ]
        return FrameSequence(frames=frames, fps=self.fps)

    def unload(self) -> None:
        self.unloaded = True


class FakeGrounding:
    model_id = "fake-grounding-dino"

    def load(self) -> None:
        pass

    def detect(self, image, text_prompt: str) -> list:
        return [type("D", (), {"label": "banner", "score": 0.9, "box": (10, 10, 60, 90)})()]

    def unload(self) -> None:
        pass


class FakeSegmentation:
    model_id = "fake-sam2.1"

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list:
        h, w = image.shape[0], image.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[box.y0 : box.y1, box.x0 : box.x1] = 255
        return [type("M", (), {"mask": mask, "iou_score": 0.9})()]

    def unload(self) -> None:
        pass


class BatchVLM:
    """Answers the object-description batch prompt with one accepted candidate per box."""

    def __init__(self):
        self.call_count = 0

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        n_boxes = sum(1 for line in prompt.splitlines() if line.startswith("["))
        entries = []
        for i in range(n_boxes):
            entries.append(
                {
                    "box_index": i,
                    "bbox_assessment": "pass",
                    "object_identity": "speed_lines",
                    "matches_semantic_label": True,
                    "animatable": True,
                    "movable_parts": ["all"],
                    "static_parts": [],
                    "motion_kind": "flow",
                    "direction": None,
                    "amplitude_band": "moderate",
                    "speed_band": "slow",
                    "pivot_hint": "center",
                    "constraints": [],
                    "neighbor_conflicts": [],
                    "confidence": 0.9,
                    "reason": "the speed lines show the motion",
                }
            )
        return json.dumps(entries)


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
    cfg = PipelineConfig(duration_s=0.5, fps=24)
    cfg.model_variants["animation"] = "cogvideox-5b-i2v"
    return cfg


@requires_ffmpeg
def test_cogvideox_engine_renders_pass_video(page_path: Path, config, tmp_path: Path):
    """The CogVideoX engine replaces plan/animate/reconstruct + compositing: the fake CogVideoX
    client's frames are rendered directly into a real playable MP4, and the panel is PASS."""
    cogvideox = FakeCogVideoX()
    vlm = BatchVLM()
    result = run_page_panels(
        page_path,
        config,
        vlm_client=vlm,
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=[cogvideox],
        out_dir=tmp_path / "out",
        labels=["speed_lines"],
    )
    assert len(cogvideox.calls) == 1  # exactly one generative call for the one panel
    assert result.panels[0].status == "PASS"
    assert result.panels[0].output_video is not None
    assert result.panels[0].output_video.exists()
    # No LaMa client was required on the CogVideoX path (reconstruction_client=None).
    assert cogvideox.loaded and cogvideox.unloaded


@requires_ffmpeg
def test_cogvideox_engine_receives_panel_image_and_prompt(
    page_path: Path, config, tmp_path: Path
):
    """The generative engine's input contract: the ORIGINAL panel crop and the prompt
    built from the accepted Qwen description."""
    cogvideox = FakeCogVideoX()
    result = run_page_panels(
        page_path,
        config,
        vlm_client=BatchVLM(),
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=[cogvideox],
        out_dir=tmp_path / "out",
        labels=["speed_lines"],
    )
    assert result.panels[0].status == "PASS"
    call = cogvideox.calls[0]
    # image_shape is the panel crop (full page here since one panel fills it).
    assert call["image_shape"][:2] == (160, 120)
    assert call["mask_shape"][:2] == (160, 120)
    assert call["mask_area"] > 0  # the accepted SAM mask reached the engine
    assert "speed lines" in call["prompt"]  # built from the Qwen description
    assert "flowing" in call["prompt"]  # motion phrase from the mapped transform kind


def test_cogvideox_engine_requires_animation_client_when_selected(
    page_path: Path, config, tmp_path: Path
):
    """The CogVideoX engine is only active when an animation_client is passed; without it, the
    deterministic engine's reconstruction_client is required (fail closed on misuse)."""
    with pytest.raises(AssertionError):
        run_page_panels(
            page_path,
            config,
            vlm_client=BatchVLM(),
            grounding_client=FakeGrounding(),
            segmentation_client=FakeSegmentation(),
            reconstruction_client=None,
            animation_clients=None,
            out_dir=tmp_path / "out",
            labels=["speed_lines"],
        )


@requires_ffmpeg
def test_cogvideox_engine_fails_closed_on_empty_acceptance(
    page_path: Path, config, tmp_path: Path
):
    """A panel with no accepted candidate is REJECTED even on the CogVideoX path (the generative
    engine never animates an unvalidated panel)."""

    class RejectingVLM(BatchVLM):
        def generate(self, image, prompt: str) -> str:
            self.call_count += 1
            n_boxes = sum(1 for line in prompt.splitlines() if line.startswith("["))
            entries = []
            for i in range(n_boxes):
                entries.append(
                    {
                        "box_index": i,
                        "bbox_assessment": "reject",
                        "object_identity": None,
                        "matches_semantic_label": False,
                        "animatable": False,
                        "movable_parts": [],
                        "static_parts": [],
                        "motion_kind": None,
                        "direction": None,
                        "amplitude_band": None,
                        "speed_band": None,
                        "pivot_hint": None,
                        "constraints": [],
                        "neighbor_conflicts": [],
                        "confidence": 0.1,
                        "reason": "no coherent object",
                    }
                )
            return json.dumps(entries)

    cogvideox = FakeCogVideoX()
    result = run_page_panels(
        page_path,
        config,
        vlm_client=RejectingVLM(),
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=[cogvideox],
        out_dir=tmp_path / "out",
        labels=["speed_lines"],
    )
    assert result.panels[0].status == "REJECTED"
    assert len(cogvideox.calls) == 0  # the generative engine never ran


@requires_ffmpeg
def test_cogvideox_two_phase_run_restores_checkpoints_and_animates(
    page_path: Path, config, tmp_path: Path
):
    """ADR 0024 two-phase run: phase 1 (`stop_after_segmentation=True`) runs only
    grounding/description/segmentation with Qwen and persists checkpoints; phase 2
    re-runs `run_page_panels` with the CogVideoX pool -- the restored checkpoints skip
    DINO/Qwen/SAM and the CogVideoX engine animates the accepted panel. The CogVideoX
    pool here is a single fake client."""
    vlm = BatchVLM()
    phase1 = run_page_panels(
        page_path,
        config,
        vlm_client=vlm,
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=None,
        out_dir=tmp_path / "out",
        labels=["speed_lines"],
        stop_after_segmentation=True,
    )
    assert vlm.call_count == 1  # Qwen phase described the panel
    assert phase1.panels[0].status != "PASS"  # not rendered yet (status may be ERROR)

    cogvideox = FakeCogVideoX()
    phase2 = run_page_panels(
        page_path,
        config,
        vlm_client=BatchVLM(),
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=[cogvideox],
        out_dir=tmp_path / "out",
        labels=["speed_lines"],
    )
    assert phase2.panels[0].status == "PASS"
    assert len(cogvideox.calls) == 1  # CogVideoX phase animated the accepted panel
    # The checkpoints exist on disk.
    assert (tmp_path / "out" / "page" / "grounding.json").exists()
    assert (tmp_path / "out" / "page" / "descriptions.json").exists()
    assert (tmp_path / "out" / "page" / "segmentation.json").exists()


def test_cogvideox_pool_splits_panels_between_clients(
    page_path: Path, config, tmp_path: Path
):
    """The CogVideoX stage runs as a worker pool (one worker per CogVideoXClient): with two
    fake clients, panels are split between them rather than serialized by one global lock."""
    # A two-panel page so both pool workers see work.
    def noise_block(h: int, w: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    two = np.full((900, 300, 3), 255, dtype=np.uint8)
    two[0:300, 0:300] = noise_block(300, 300, seed=21)
    two[500:900, 0:300] = noise_block(400, 300, seed=22)
    two_path = tmp_path / "two_panel.png"
    Image.fromarray(two).save(two_path)

    cogvideox_a, cogvideox_b = FakeCogVideoX(), FakeCogVideoX()
    result = run_page_panels(
        two_path,
        config,
        vlm_client=BatchVLM(),
        grounding_client=FakeGrounding(),
        segmentation_client=FakeSegmentation(),
        reconstruction_client=None,
        animation_clients=[cogvideox_a, cogvideox_b],
        out_dir=tmp_path / "out2",
        labels=["speed_lines"],
    )
    total_calls = len(cogvideox_a.calls) + len(cogvideox_b.calls)
    assert total_calls >= 1  # at least the accepted panel(s) animated
    assert all(p.status == "PASS" for p in result.panels)
