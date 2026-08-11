---
name: video-rendering
description: FFmpeg/H.264 encoding guidance — frame-to-video assembly, FPS/duration handling, building a genuinely seamless loop, sensible encode settings, and validating the output file. Load when implementing or reviewing the frame-sequence-to-video rendering step.
---

# Video rendering

Practical FFmpeg/encoding guidance for the final rendering stage
(`src/manga_animation/rendering`). This is about mechanics — for what "seamless" and
"quality" mean architecturally, see `docs/architecture.md` and the `evaluation` skill.

## Frame sequence conventions

- Number frames zero-padded and sequentially (`frame_0000.png`, `frame_0001.png`, ...) so
  ffmpeg's `-i frame_%04d.png` pattern (or equivalent) works without a manual file list.
- Frame count must equal `LoopSpec.frame_count` (`round(duration_s * fps)`) exactly — verify
  this before invoking ffmpeg rather than discovering a length mismatch after encoding.
- Prefer lossless intermediate frames (PNG) even though the final output is a lossy codec —
  compounding compression artifacts across the compositing → encoding pipeline is avoidable
  and makes debugging quality issues much harder.

## Building an actually-seamless loop

`AnimationPlan.loop.seamless=True` is a promise the *frame sequence* must keep, not just
the `MotionSpec.speed` math (see the seamless-loop constraint in
`docs/animation-plan-schema.md` — that constraint is necessary but not sufficient):

- With every object's `speed` correctly integer-cyclic, frame 0 and the frame *after* the
  last frame should already be identical in principle — verify this numerically (pixel
  diff between frame 0 and a freshly-generated "frame N" at `t=duration`) before assuming
  it's true.
- If `LoopSpec.crossfade_frames > 0`, blend the last N frames toward frame 0 (linear alpha
  blend is usually sufficient) rather than encoding a hard cut — use this as a safety net
  for small numerical mismatches, not as a substitute for getting the motion math right.
- When concatenating for playback preview, use ffmpeg's `-stream_loop` or a looped output
  container rather than manually duplicating frames — keeps the source frame count honest.

## Baseline H.264 encode settings

A reasonable, broadly-compatible starting point (adjust deliberately, not by copying flags
without understanding them):

```bash
ffmpeg -y -framerate <fps> -i frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
  -movflags +faststart \
  output.mp4
```

- `-pix_fmt yuv420p` — required for broad player/browser compatibility; without it some
  players fail to display the video at all.
- `-crf 18` — visually near-lossless; raise toward 23 if file size matters more than
  fidelity to the (already lightly-modified) source frames.
- `-movflags +faststart` — moves the moov atom to the front so the file starts playing
  before it's fully downloaded; cheap to include, no downside for this use case.

## Output validation

Before reporting a render as successful, check (via `ffprobe`, not just "the file exists"):

- Duration matches `LoopSpec.duration_s` within one frame interval.
- FPS matches `LoopSpec.fps`.
- Resolution matches the source image (or the configured output resolution, if downscaled
  deliberately).
- The file is a valid, demuxable H.264 stream (`ffprobe -v error output.mp4` produces no
  errors).

## System dependency, not a Python package

`ffmpeg` must be present on the machine actually running this stage (local or remote) —
check with `ffmpeg -version` and fail with a clear, actionable message if it's missing,
rather than letting a subprocess call fail cryptically.
