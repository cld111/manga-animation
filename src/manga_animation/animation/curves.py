"""Sampling a `MotionSpec`'s motion curve at a point in time.

Every `TransformKind` in `transforms.py` is driven by a single signed scalar in `[-1, 1]`
produced here, where `0.0` always means "object at rest" (its undeformed/original state) and
`+-1.0` means "at peak deflection" in that transform's own convention (see `transforms.py`).
Keeping that convention uniform across all three `loop_mode`s is what lets a single
`_apply_value` function in `transforms.py` not need to know which mode produced the value.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from manga_animation.schemas.animation_plan import Easing, MotionSpec

EASING_FUNCS: dict[Easing, Callable[[float], float]] = {
    Easing.LINEAR: lambda t: t,
    Easing.EASE_IN: lambda t: t * t,
    Easing.EASE_OUT: lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    Easing.EASE_IN_OUT: lambda t: 3 * t**2 - 2 * t**3,
    Easing.SINE: lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
}


def sample_motion_value(motion: MotionSpec, t_s: float, loop_duration_s: float) -> float:
    """Sample `motion`'s signed progress in `[-1, 1]` at absolute loop time `t_s` (seconds).

    Before `timing.delay_s`, the object is at rest (`0.0`). Within its active window
    (`[delay_s, delay_s + duration_s)`), the three `loop_mode`s each interpret local progress
    `u` in `[0, 1)` differently:

    - `cycle`: a periodic `sin` oscillation (`speed` cycles per window), matching the
      seamless-loop convention in docs/animation-plan-schema.md — a whole-number `speed`
      returns exactly to `0.0` at `u=1`.
    - `once_hold`: a single monotonic sweep `0.0 -> 1.0` via `easing`, then holds `1.0`.
    - `ping_pong`: a single out-and-back sweep `0.0 -> 1.0 -> 0.0` via a triangular envelope,
      then holds `0.0` — i.e. it naturally returns to rest once its window closes.

    Past the active window, the value freezes at whatever the curve reaches at `u=1`: for
    `cycle` that's `0.0` whenever `speed` is a whole number (the same condition the schema
    already requires for a seamless loop), for `once_hold` it's `1.0`, for `ping_pong` it's
    `0.0`. This is a deliberate, documented interpretation of an otherwise-unspecified corner
    of the schema (what happens after a bounded window closes), not a schema requirement.
    """
    timing = motion.timing
    duration = (
        timing.duration_s if timing.duration_s is not None else (loop_duration_s - timing.delay_s)
    )
    if t_s < timing.delay_s:
        return 0.0

    t_local = t_s - timing.delay_s
    u = 0.0 if duration <= 0 else min(t_local / duration, 1.0)

    if timing.loop_mode == "cycle":
        raw = math.sin(2 * math.pi * (motion.speed * u + motion.phase))
        progress = (raw + 1.0) / 2.0
        eased = EASING_FUNCS[motion.easing](progress)
        return eased * 2.0 - 1.0
    if timing.loop_mode == "once_hold":
        return EASING_FUNCS[motion.easing](u)
    if timing.loop_mode == "ping_pong":
        triangle = 1.0 - abs(1.0 - 2.0 * u)
        return EASING_FUNCS[motion.easing](triangle)
    raise ValueError(f"unknown loop_mode {timing.loop_mode!r}")
