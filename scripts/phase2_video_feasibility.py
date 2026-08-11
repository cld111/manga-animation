"""Phase 2 feasibility check for the video-rendering stage.

Encodes the frame sequence produced by `phase2_cv_feasibility.py`
(`outputs/frames/phase2-demo/frame_%04d.png`) into H.264 using the settings documented in
the `video-rendering` skill, then validates the output numerically (not just "the file
exists") — frame count, fps, resolution, and first/last-frame loop continuity within codec
tolerance (see the `evaluation` skill's "Loop quality" section).

This is a **local, non-GPU, non-ML** check — encoding is deterministic and CPU-bound, so
running it locally does not conflict with the project's "remote GPU for model work" policy
(ADR 0003/0004; see CLAUDE.md). It uses `imageio-ffmpeg`'s vendored, sandboxed ffmpeg binary
as a fallback ONLY when no system `ffmpeg` is on PATH, purely so this feasibility check can
run without requiring a system-wide install first. Production `video-agent` code must still
check for and depend on a real system `ffmpeg` per its documented constraint (see
`.claude/agents/video-agent.md`) — this fallback is a validation convenience, not a proposed
runtime dependency.

Usage: uv run python scripts/phase2_video_feasibility.py
Requires the `cv` optional dependency group for output validation: `uv sync --extra cv`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2


def _resolve_ffmpeg() -> tuple[str, str]:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg, "system"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg (sandboxed fallback)"
    except ImportError:
        raise SystemExit(
            "no system ffmpeg found and imageio-ffmpeg not installed — "
            "install ffmpeg (e.g. `brew install ffmpeg`) or "
            "`uv run --with imageio-ffmpeg python scripts/phase2_video_feasibility.py`"
        ) from None


def encode(ffmpeg_bin: str, frames_dir: Path, fps: int, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        # yuv420p requires even width/height; a real manga page is not guaranteed to have
        # either (this run's sample is 800x2305 — odd height) — pad up by at most 1px/side
        # rather than crop, so no source content is lost. See the finding recorded in
        # docs/phase2-benchmark-results.md.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
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
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - start
    if result.returncode != 0:
        raise SystemExit(
            f"ffmpeg encode failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    return {"command": cmd, "encode_time_s": elapsed_s}


def validate(out_path: Path, expected_fps: int, expected_frame_count: int) -> dict:
    """Numeric validation via cv2 (not ffprobe, which imageio-ffmpeg doesn't vendor) —

    still real decode + measurement, not just an existence check, matching the intent of
    the video-rendering skill's "Output validation" section.
    """
    cap = cv2.VideoCapture(str(out_path))
    if not cap.isOpened():
        return {"demuxable": False}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()

    # The wrap-point step (last decoded frame -> first, i.e. what playback shows when the
    # loop repeats) should be the SAME MAGNITUDE as an ordinary adjacent-frame step, not
    # necessarily near-zero — frame[N-1] and frame[0] are themselves a full oscillation step
    # apart under an integer-cycle sinusoid (see the schema's seamless-loop constraint); a
    # *discontinuous* wrap would show up as a wrap-step diff much larger than the ordinary
    # step, not as a nonzero diff per se. This is what phase2_cv_feasibility.py's pre-encode
    # frame-0-vs-regenerated-frame-N check cannot see: what H.264 quantization/GOP structure
    # does to that step.
    loop_continuity = None
    if len(frames) >= 3:
        ordinary_step = cv2.absdiff(frames[1], frames[0]).mean()
        wrap_step = cv2.absdiff(frames[0], frames[-1]).mean()
        loop_continuity = {
            "ordinary_adjacent_step_mean_abs_diff": float(ordinary_step),
            "wrap_step_mean_abs_diff": float(wrap_step),
            "wrap_step_within_2x_ordinary": bool(wrap_step <= 2.0 * max(ordinary_step, 1e-6)),
        }

    return {
        "demuxable": True,
        "reported_fps": fps,
        "reported_frame_count": frame_count,
        "resolution": [width, height],
        "fps_matches_expected": abs(fps - expected_fps) < 0.1,
        "frame_count_within_one": abs(frame_count - expected_frame_count) <= 1,
        "loop_continuity": loop_continuity,
    }


def main() -> None:
    frames_dir = Path("outputs/frames/phase2-demo")
    if not frames_dir.exists() or not any(frames_dir.glob("frame_*.png")):
        raise SystemExit(f"{frames_dir} has no frames — run scripts/phase2_cv_feasibility.py first")
    frame_count = len(list(frames_dir.glob("frame_*.png")))
    fps = 24
    out_path = Path("outputs/videos/phase2_feasibility_demo.mp4")

    ffmpeg_bin, ffmpeg_source = _resolve_ffmpeg()
    encode_info = encode(ffmpeg_bin, frames_dir, fps, out_path)
    validation = validate(out_path, expected_fps=fps, expected_frame_count=frame_count)

    summary = {
        "ffmpeg_binary": ffmpeg_bin,
        "ffmpeg_source": ffmpeg_source,
        "frames_dir": str(frames_dir),
        "input_frame_count": frame_count,
        "fps": fps,
        "output_path": str(out_path),
        "output_size_bytes": out_path.stat().st_size,
        "encode": encode_info,
        "validation": validation,
    }
    out_json = Path("outputs/experiments/phase2_video_feasibility.json")
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print(f"wrote {out_path} ({summary['output_size_bytes'] / 1024:.0f} KiB)")
    print(f"wrote {out_json}")
    print(f"ffmpeg source: {ffmpeg_source}")
    print(f"encode time: {encode_info['encode_time_s']:.2f}s for {frame_count} frames")
    print(f"demuxable: {validation['demuxable']}")
    print(
        f"fps matches: {validation.get('fps_matches_expected')}  "
        f"frame count matches: {validation.get('frame_count_within_one')}  "
        f"resolution: {validation.get('resolution')}"
    )
    if validation.get("loop_continuity"):
        print(f"loop continuity: {validation['loop_continuity']}")


if __name__ == "__main__":
    main()
