"""Behavioral tests for src/manga_animation/rendering.

Video encoding is deterministic, CPU-bound, and non-ML -- already established as safe to
exercise for real, locally, in ADR 0005 (`scripts/phase2_video_feasibility.py`). These tests
run the real `render()` function end-to-end against a synthetic frame sequence whenever an
ffmpeg binary is resolvable (system `ffmpeg`, or `imageio-ffmpeg`'s vendored binary as a
test-only convenience -- production code in `encode.py` never uses that fallback, only these
tests do), and skip cleanly with a clear reason otherwise.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_animation.pipeline.types import FrameSequence, PipelineStageError
from manga_animation.rendering import compute_loop_metrics, render
from manga_animation.rendering.encode import _ssim


def _resolve_test_ffmpeg() -> str | None:
    """System ffmpeg if present, else imageio-ffmpeg's vendored binary (test-only)."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return None


_FFMPEG_BIN = _resolve_test_ffmpeg()
requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG_BIN is None, reason="no ffmpeg binary resolvable (neither system nor imageio-ffmpeg)"
)


def _moving_circle_sequence(
    *, count: int = 24, size: tuple[int, int] = (64, 64), fps: int = 24, periodic: bool = True
) -> FrameSequence:
    """A cheap synthetic sequence: a colored circle orbiting the frame center.

    `periodic=True` makes frame 0 and the implicit "frame after the last" pixel-identical
    (a genuine loop) by sampling a full 2*pi cycle across exactly `count` steps, i.e. frame i
    samples angle `2*pi*i/count`, so frame `count` (not rendered) would equal frame 0 again.
    """
    w, h = size
    frames = []
    radius = min(w, h) * 0.3
    cx, cy = w / 2, h / 2
    for i in range(count):
        angle = 2 * np.pi * i / count if periodic else (i / count) * 1.3
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        img = Image.new("RGB", (w, h), (10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 40, 40))
        frames.append(np.array(img, dtype=np.uint8))
    return FrameSequence(frames=frames, fps=fps)


# --- compute_loop_metrics (pure function, no ffmpeg needed) ------------------------------


def test_compute_loop_metrics_returns_none_below_three_frames():
    frames = _moving_circle_sequence(count=2).frames
    assert compute_loop_metrics(frames) is None


def test_compute_loop_metrics_passes_both_checks_for_a_genuine_periodic_sequence():
    frames = _moving_circle_sequence(count=24, periodic=True).frames
    metrics = compute_loop_metrics(frames)
    assert metrics is not None
    assert metrics.wrap_step_within_2x_ordinary is True
    assert metrics.wrap_ssim_within_tolerance is True
    assert metrics.seamless is True
    # the wrap step is one more identical-magnitude motion step, same as any ordinary step --
    # structurally, it should score close to the ordinary step's own SSIM, not near zero.
    assert metrics.wrap_step_ssim == pytest.approx(metrics.ordinary_adjacent_step_ssim, abs=0.1)


def test_compute_loop_metrics_flags_both_checks_for_a_non_periodic_sequence():
    frames = _moving_circle_sequence(count=24, periodic=False).frames
    metrics = compute_loop_metrics(frames)
    assert metrics is not None
    assert metrics.wrap_step_within_2x_ordinary is False
    assert metrics.wrap_ssim_within_tolerance is False
    assert metrics.seamless is False


def test_ssim_is_one_for_identical_frames():
    frame = _moving_circle_sequence(count=1).frames[0]
    assert _ssim(frame, frame) == pytest.approx(1.0, abs=1e-6)


def test_ssim_is_low_for_structurally_unrelated_frames():
    rng = np.random.default_rng(0)
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[16:48, 16:48] = 255  # a solid square
    b = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)  # unstructured noise
    assert _ssim(a, b) < 0.3


# --- codec / precondition guards (no ffmpeg needed) --------------------------------------


def test_render_rejects_unsupported_codec(tmp_path: Path):
    frames = _moving_circle_sequence(count=4)
    with pytest.raises(PipelineStageError, match="codec"):
        render(frames, tmp_path / "out.mp4", codec="vp9")


def test_render_raises_cleanly_when_ffmpeg_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    frames = _moving_circle_sequence(count=4)
    with pytest.raises(PipelineStageError, match="ffmpeg"):
        render(frames, tmp_path / "out.mp4")


# --- real encode + validation --------------------------------------------------------------


@requires_ffmpeg
def test_render_produces_a_demuxable_video_with_matching_frame_count_and_fps(tmp_path: Path):
    frames = _moving_circle_sequence(count=24, fps=24)
    result = render(frames, tmp_path / "out.mp4")

    assert result.output_path.exists()
    assert result.frame_count == 24
    assert result.fps == pytest.approx(24, abs=0.1)
    assert result.codec == "h264"
    assert result.pixel_format == "yuv420p"
    assert result.duration_s == pytest.approx(1.0, abs=0.05)


@requires_ffmpeg
def test_render_reports_seamless_loop_for_a_periodic_sequence(tmp_path: Path):
    frames = _moving_circle_sequence(count=24, periodic=True)
    result = render(frames, tmp_path / "out.mp4")
    assert result.seamless_loop_verified is True
    assert result.loop_metrics is not None
    assert result.loop_metrics.seamless is True


@requires_ffmpeg
def test_render_reports_non_seamless_loop_for_a_non_periodic_sequence(tmp_path: Path):
    frames = _moving_circle_sequence(count=24, periodic=False)
    result = render(frames, tmp_path / "out.mp4")
    assert result.seamless_loop_verified is False
    assert result.loop_metrics is not None
    assert result.loop_metrics.seamless is False


@requires_ffmpeg
def test_render_handles_odd_dimension_frames_via_padding(tmp_path: Path):
    frames = _moving_circle_sequence(count=6, size=(65, 65))
    result = render(frames, tmp_path / "out.mp4")

    # padded up to the nearest even dimension, never cropped (no source content lost)
    assert result.resolution == (66, 66)


@requires_ffmpeg
def test_render_writes_and_keeps_frames_dir_when_requested(tmp_path: Path):
    frames = _moving_circle_sequence(count=5)
    frames_dir = tmp_path / "frames"
    render(frames, tmp_path / "out.mp4", frames_dir=frames_dir)

    written = sorted(frames_dir.glob("frame_*.png"))
    assert len(written) == 5


@requires_ffmpeg
def test_render_cleans_up_temp_frames_by_default(tmp_path: Path):
    frames = _moving_circle_sequence(count=5)
    out_path = tmp_path / "out.mp4"
    render(frames, out_path)

    # no frame dump left behind next to the output when frames_dir/keep_frames weren't asked for
    leftovers = list(tmp_path.glob("*_frames"))
    assert leftovers == []
