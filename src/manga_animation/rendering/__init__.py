"""Frame sequencing, seamless looping and H.264 video encoding.

Turns a `FrameSequence` (`src/manga_animation/pipeline/types.py`) into a validated MP4 via a
real system `ffmpeg` binary -- see `encode.render`. Ported from the already
locally-executed feasibility check in `scripts/phase2_video_feasibility.py` (see ADR 0005's
"video-rendering" section): same proven ffmpeg settings, same even-dimension padding
requirement, same measurement-based validation approach, now as real stage code rather than
a throwaway script.
"""

from manga_animation.rendering.encode import compute_loop_metrics, render

__all__ = ["render", "compute_loop_metrics"]
