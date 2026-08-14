"""Frame sequence -> validated H.264 MP4, via a real system `ffmpeg` binary.

Ported from `scripts/phase2_video_feasibility.py`, which already executed this exact
recipe for real, locally, CPU-only (no GPU/ML involved — see ADR 0005's "video-rendering"
section): same proven ffmpeg flags, the same even-dimension `yuv420p` padding requirement
(manga pages are not guaranteed even width/height), and the same measurement-based
validation approach (`cv2.VideoCapture`, since `ffprobe` is not guaranteed present either).

Unlike the feasibility script, this module does **not** fall back to `imageio-ffmpeg` — a
real system `ffmpeg` is a hard requirement here (see `.claude/agents/video-agent.md`); that
fallback stays a test-only/validation-only convenience.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import (
    FrameSequence,
    LoopMetrics,
    PipelineStageError,
    RenderResult,
)

logger = get_logger(__name__)

# Only h264/libx264 has a verified-working encode recipe (Phase 3.1's stack). Other
# PipelineConfig.output_codec values are recognized but not yet implemented -- see the
# explicit rejection in render() below rather than silently emitting an unverified encode.
_CODEC_TO_LIBX = {"h264": "libx264"}

_MAX_STDERR_CHARS = 2000


def render(
    frames: FrameSequence,
    out_path: Path,
    *,
    codec: str = "h264",
    keep_frames: bool = False,
    frames_dir: Path | None = None,
) -> RenderResult:
    """Encode `frames` to `out_path`, validate the result, and return what was measured.

    `frames_dir`, if given, is where the intermediate `frame_%04d.png` sequence is written
    and left in place (matching the Phase 3.1 brief's "keep the frame sequence available as
    an ignored output artifact for debugging"). If not given but `keep_frames=True`, a
    sibling directory next to `out_path` is used instead. Otherwise a temp dir is used and
    cleaned up before returning.

    Raises `PipelineStageError(stage="rendering", ...)` on any technical encode/validation
    failure (missing ffmpeg, ffmpeg exiting non-zero, a demux failure, or frame
    count/fps/resolution not matching what was requested) -- never returns a `RenderResult`
    that claims success while one of those is actually wrong. Seamless-loop continuity is
    *measured and reported* (`RenderResult` doesn't carry it directly -- see the
    `seamless_loop_verified` field) rather than treated as a hard failure here: whether the
    input `FrameSequence` actually loops cleanly is a property of the upstream animation
    stage's output, not something this stage can fix -- it can only verify and report.
    """
    if codec not in _CODEC_TO_LIBX:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail=f"codec {codec!r} has no verified encode recipe (supported: "
            f"{sorted(_CODEC_TO_LIBX)})",
            architectural=False,
            proposed_fix="pass a supported codec, or verify+add ffmpeg flags for this one",
        )

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail="no system ffmpeg binary found on PATH",
            root_cause="ffmpeg is a system dependency, not installed by uv/pip",
            architectural=False,
            proposed_fix="install ffmpeg (e.g. `brew install ffmpeg`) on this machine/worker",
        )

    # Seamless-loop promise, checked at the FrameSequence level -- see
    # docs/animation-plan-schema.md: the animation stage samples `frame_count` frames at
    # t_frac = i / frame_count for i in 0..frame_count-1 (matching the proven approach in
    # scripts/phase2_cv_feasibility.py), so frame[-1] (at t=(N-1)/N) is one step *before* the
    # wrap back to frame[0] (at t=0) -- it is NOT expected to equal frame[0] pixel-for-pixel
    # for any real periodic motion (only a motionless object would satisfy that). What the
    # schema actually promises ("frame 0 and the frame after the last one are visually
    # identical") is *wrap-step continuity*: the transition from frame[-1] back to frame[0]
    # should be the same order of magnitude as an ordinary adjacent-frame step, exactly the
    # metric `scripts/phase2_video_feasibility.py` already validated post-encode. A prior
    # version of this check compared frame[0] to frame[-1] for exact equality, which is wrong
    # for the same reason a `mask < 8` threshold was wrong for static-region checks (see ADR
    # 0005) -- it flags healthy periodic motion as "not seamless".
    first_frame = frames.frames[0]
    source_loop_metrics = compute_loop_metrics(frames.frames)

    work_dir, cleanup = _resolve_work_dir(out_path, frames_dir, keep_frames)
    try:
        for i, frame in enumerate(frames.frames):
            Image.fromarray(frame).save(work_dir / f"frame_{i:04d}.png")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(frames.fps),
            "-i",
            str(work_dir / "frame_%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            _CODEC_TO_LIBX[codec],
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PipelineStageError(
                stage="rendering",
                input_ref=str(out_path),
                detail=f"ffmpeg exited {result.returncode}",
                root_cause=result.stderr[-_MAX_STDERR_CHARS:],
                architectural=False,
                proposed_fix="inspect ffmpeg stderr for the failing filter/codec/input",
            )
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

    src_h, src_w = first_frame.shape[:2]
    expected_resolution = (_round_up_even(src_w), _round_up_even(src_h))
    validation = _validate(
        out_path,
        expected_fps=frames.fps,
        expected_frame_count=frames.frame_count,
        expected_resolution=expected_resolution,
    )

    if not validation["demuxable"]:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail="encoded output is not demuxable by cv2.VideoCapture",
            architectural=False,
            proposed_fix="inspect the ffmpeg command/output file directly",
        )
    if not validation["frame_count_within_one"]:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail=(
                f"frame count mismatch: expected {frames.frame_count}, "
                f"got {validation['decoded_frame_count']} decoded frames"
            ),
            architectural=False,
            proposed_fix="check the input frame sequence for gaps/duplicates before encoding",
        )
    if not validation["fps_matches_expected"]:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail=f"fps mismatch: requested {frames.fps}, reported {validation['reported_fps']}",
            architectural=False,
            proposed_fix="check the -framerate argument and container timing",
        )
    if tuple(validation["resolution"]) != expected_resolution:
        raise PipelineStageError(
            stage="rendering",
            input_ref=str(out_path),
            detail=(
                f"resolution mismatch: expected {expected_resolution} (source padded to even "
                f"dimensions), got {tuple(validation['resolution'])}"
            ),
            architectural=False,
            proposed_fix="check the pad filter expression",
        )

    seamless_loop_verified = True
    if source_loop_metrics is not None and not source_loop_metrics.wrap_step_within_2x_ordinary:
        seamless_loop_verified = False
        logger.warning(
            "source FrameSequence wrap-step diff (%.3f) exceeds 2x the ordinary adjacent-frame "
            "step (%.3f) -- the loop is not seamless before encoding is even involved",
            source_loop_metrics.wrap_step_mean_abs_diff,
            source_loop_metrics.ordinary_adjacent_step_mean_abs_diff,
        )
    elif source_loop_metrics is not None and not source_loop_metrics.wrap_ssim_within_tolerance:
        seamless_loop_verified = False
        logger.warning(
            "source FrameSequence wrap-step SSIM (%.3f) falls short of the ordinary "
            "adjacent-frame step's SSIM (%.3f) by more than the tolerance -- structurally, the "
            "wrap transition does not resemble an ordinary motion step",
            source_loop_metrics.wrap_step_ssim,
            source_loop_metrics.ordinary_adjacent_step_ssim,
        )
    loop_metrics: LoopMetrics | None = validation.get("loop_metrics")
    if loop_metrics is not None and not loop_metrics.seamless:
        seamless_loop_verified = False
        logger.warning(
            "post-encode loop metrics failed: wrap-step diff %.3f (ordinary %.3f), wrap-step "
            "SSIM %.3f (ordinary %.3f) -- H.264 encoding may have introduced a visible seam",
            loop_metrics.wrap_step_mean_abs_diff,
            loop_metrics.ordinary_adjacent_step_mean_abs_diff,
            loop_metrics.wrap_step_ssim,
            loop_metrics.ordinary_adjacent_step_ssim,
        )

    # Prefer the post-encode, real-decoded measurement (what a viewer actually sees) as the
    # attached evidence -- falls back to the pre-encode source measurement only when the
    # decoded file somehow yielded too few frames to compute one (shouldn't happen once the
    # frame-count validation above has already passed, but avoids discarding real source
    # evidence over a defensive edge case).
    attached_loop_metrics = loop_metrics if loop_metrics is not None else source_loop_metrics

    return RenderResult(
        output_path=out_path,
        frame_count=validation["decoded_frame_count"],
        fps=validation["reported_fps"],
        resolution=tuple(validation["resolution"]),
        duration_s=validation["decoded_frame_count"] / frames.fps,
        codec=codec,
        # Not independently re-derivable from cv2.VideoCapture (it doesn't reliably expose
        # pixel format) -- this is the value we *requested* and the encode did not error, not
        # an independently re-measured fact. See the module docstring.
        pixel_format="yuv420p",
        seamless_loop_verified=seamless_loop_verified,
        loop_metrics=attached_loop_metrics,
    )


def _resolve_work_dir(
    out_path: Path, frames_dir: Path | None, keep_frames: bool
) -> tuple[Path, bool]:
    """Where to write the intermediate PNG frame sequence, and whether to delete it after."""
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir, False
    if keep_frames:
        sibling = out_path.parent / f"{out_path.stem}_frames"
        sibling.mkdir(parents=True, exist_ok=True)
        return sibling, False
    return Path(tempfile.mkdtemp(prefix="manga_animation_render_")), True


def _round_up_even(value: int) -> int:
    return value + (value % 2)


_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2
_SSIM_WRAP_TOLERANCE = 0.05
"""How far below the ordinary adjacent-step SSIM the wrap-step SSIM may fall before the loop is

flagged as structurally discontinuous. An absolute (not relative) tolerance on SSIM's own
[-1, 1] scale, deliberately conservative (small) rather than tuned for maximum acceptance --
same "documented, evidenced choice, not a statistically calibrated set" status as every other
threshold in this codebase (e.g. ADR 0008's transform-geometry bounds). Verified against both a
genuine periodic sequence and a deliberately non-periodic one in
`tests/test_rendering.py::test_compute_loop_metrics_*`."""


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity (Wang et al. 2004) between two same-shape uint8 images,

    windowed with an 11x11 Gaussian (the original paper's own window), averaged over color
    channels for an RGB image. 1.0 = structurally identical; lower values mean the local
    luminance/contrast/pattern correlation between `a` and `b` is weaker, independent of (and
    not derivable from) their raw pixel-magnitude difference -- see `LoopMetrics`'s docstring
    for why this project wants both signals, not just one. Pure `cv2`/`numpy`, no new
    dependency (`cv2.getGaussianKernel`/`cv2.filter2D` are core `imgproc`, not `contrib`).
    """
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    kernel_1d = cv2.getGaussianKernel(11, 1.5)
    window = kernel_1d @ kernel_1d.T

    def _windowed(x: np.ndarray) -> np.ndarray:
        return cv2.filter2D(x, -1, window, borderType=cv2.BORDER_REPLICATE)

    channel_scores = []
    for c in range(a64.shape[2]):
        x, y = a64[..., c], b64[..., c]
        mu_x, mu_y = _windowed(x), _windowed(y)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x_sq = _windowed(x * x) - mu_x_sq
        sigma_y_sq = _windowed(y * y) - mu_y_sq
        sigma_xy = _windowed(x * y) - mu_xy
        numerator = (2 * mu_xy + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)
        denominator = (mu_x_sq + mu_y_sq + _SSIM_C1) * (sigma_x_sq + sigma_y_sq + _SSIM_C2)
        channel_scores.append(float((numerator / denominator).mean()))
    return sum(channel_scores) / len(channel_scores)


def compute_loop_metrics(frames: list[np.ndarray]) -> LoopMetrics | None:
    """Is the transition from the last frame back to the first the same order of magnitude, and

    as structurally continuous, as an ordinary adjacent-frame step? That -- not
    `frames[0] == frames[-1]` -- is what "the loop is seamless" actually means for a sequence
    sampled at t_frac = i / N (see the note in `render()`): frame[-1] (at t=(N-1)/N) is one
    step *before* the wrap back to frame[0] (at t=0), so it is not expected to equal frame[0]
    pixel-for-pixel for any real periodic motion, only to *transition* the same way an ordinary
    step does. `None` when there aren't enough frames to compare (fewer than 3).
    """
    if len(frames) < 3:
        return None
    ordinary_step = float(cv2.absdiff(frames[1], frames[0]).mean())
    wrap_step = float(cv2.absdiff(frames[0], frames[-1]).mean())
    ordinary_ssim = _ssim(frames[1], frames[0])
    wrap_ssim = _ssim(frames[0], frames[-1])
    return LoopMetrics(
        ordinary_adjacent_step_mean_abs_diff=ordinary_step,
        wrap_step_mean_abs_diff=wrap_step,
        wrap_step_within_2x_ordinary=wrap_step <= 2.0 * max(ordinary_step, 1e-6),
        ordinary_adjacent_step_ssim=ordinary_ssim,
        wrap_step_ssim=wrap_ssim,
        wrap_ssim_within_tolerance=wrap_ssim >= ordinary_ssim - _SSIM_WRAP_TOLERANCE,
    )


def _validate(
    out_path: Path,
    *,
    expected_fps: int,
    expected_frame_count: int,
    expected_resolution: tuple[int, int],
) -> dict:
    """Decode the encoded file for real and measure what it actually is.

    Mirrors `scripts/phase2_video_feasibility.py`'s `validate()` -- `ffprobe` isn't
    guaranteed present, so `cv2.VideoCapture` (a real decode, not a metadata guess) is used
    instead, exactly as that already-executed feasibility check established.
    """
    cap = cv2.VideoCapture(str(out_path))
    if not cap.isOpened():
        return {"demuxable": False}

    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()

    loop_metrics = compute_loop_metrics(frames)

    return {
        "demuxable": True,
        "reported_fps": fps,
        "reported_frame_count": reported_frame_count,
        "decoded_frame_count": len(frames),
        "resolution": (width, height),
        "fps_matches_expected": abs(fps - expected_fps) < 0.1,
        "frame_count_within_one": abs(len(frames) - expected_frame_count) <= 1,
        "metadata_frame_count_matches": abs(reported_frame_count - len(frames)) <= 1,
        "resolution_matches_expected": (width, height) == expected_resolution,
        "loop_metrics": loop_metrics,
    }
