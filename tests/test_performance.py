"""Phase 7.1.5: opt-in performance regression protecting Phase 6's local-rendering property.

Deliberately excluded from the default `uv run pytest` suite (see pyproject.toml's
`addopts = "-ra -m \"not slow\""` and the registered `slow` marker) -- wall-clock timing is
inherently noisier than this project's otherwise-fast, purely-deterministic suite, and this
project's own acceptance criteria explicitly warn against forcing a flaky timing test into the
default collection. Run it explicitly:

    uv run pytest -m slow tests/test_performance.py -v

This does NOT assert an exact benchmark number -- `docs/phase6-results.md` and
`scripts/phase6_local_rendering_performance.py` already carry that real, captured evidence
(25x-109x raw interpolation speedup, ~20x-60x end-to-end, both environment-dependent). What
this test protects is the qualitative property Phase 6 exists to guarantee (see
docs/decisions/0012-phase6-seamless-loop-and-local-rendering.md): `generate_transformed_layer`'s
cost for a FIXED small object must stay roughly flat as the PAGE grows, not scale with total
page pixel count the way the pre-Phase-6 full-page `cv2.warpAffine`/`cv2.remap` implementation
did. The bound below (4x) is deliberately generous against a ~7.6x page-pixel-count increase --
comfortably distinguishing "still local" from "regressed toward full-page scaling" without
flaking on ordinary CI/wall-clock noise.

Mirrors real production usage: `object_bbox_px` is precomputed once per object (exactly like
`pipeline/orchestrator.py` passing `SegmentationResult.bbox`, the post-Phase-6-follow-up fix
that closed the bbox-recomputation redundancy -- see ADR 0012's "Known limitations") and reused
across every per-frame call, not recomputed via a fresh `bbox_of_mask` scan every time.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from manga_animation.animation.transforms import bbox_of_mask, generate_transformed_layer
from manga_animation.pipeline.types import BBoxPx
from manga_animation.schemas.animation_plan import MotionSpec, TimingSpec, TransformKind

pytestmark = pytest.mark.slow


def _make_page(
    page_shape: tuple[int, int], object_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = page_shape
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = object_bbox
    mask[y0:y1, x0:x1] = 255
    return image, mask


def _mean_per_frame_ms(
    page_shape: tuple[int, int],
    object_bbox: tuple[int, int, int, int],
    motion: MotionSpec,
    n_frames: int = 24,
) -> float:
    image, mask = _make_page(page_shape, object_bbox)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    precomputed_bbox = bbox_of_mask(mask)  # computed once, like SegmentationResult.bbox

    start = time.perf_counter()
    for i in range(n_frames):
        generate_transformed_layer(
            image,
            mask,
            motion,
            panel_bbox,
            page_shape,
            i / n_frames,
            loop_duration_s=4.0,
            object_bbox_px=precomputed_bbox,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / n_frames


def test_generate_transformed_layer_cost_does_not_scale_with_page_size_for_a_fixed_small_object():
    """Phase 6's core claim, as regression protection: a fixed 40x40 object's per-frame cost on
    a small page vs. an extreme-aspect-ratio page ~7.6x its pixel count must not blow up
    proportionally with page size. Catastrophic loss of local-region rendering (e.g. a revert
    to full-page warpAffine/remap) would show up here as roughly a 7-8x cost increase; the 4x
    bound below still comfortably distinguishes "still local" from "regressed to full-page",
    without asserting a tight, environment-sensitive number.
    """
    small_object = (100, 100, 140, 140)  # fixed 40x40 object, same on both pages
    motion = MotionSpec(
        transform_kind=TransformKind.ROTATE, amplitude=25.0, timing=TimingSpec(loop_mode="cycle")
    )

    small_page_ms = _mean_per_frame_ms((600, 800), small_object, motion)
    extreme_page_ms = _mean_per_frame_ms((5062, 720), small_object, motion)  # ~7.6x the pixels

    assert extreme_page_ms < small_page_ms * 4.0, (
        f"generate_transformed_layer cost grew {extreme_page_ms / small_page_ms:.1f}x going "
        f"from a 600x800 ({small_page_ms:.4f}ms/frame) to a 5062x720 page "
        f"({extreme_page_ms:.4f}ms/frame) for the SAME small object -- Phase 6's local-region "
        "rendering property (docs/decisions/0012-phase6-seamless-loop-and-local-rendering.md) "
        "appears to have regressed toward full-page scaling"
    )


def test_generate_transformed_layer_cost_does_not_scale_with_frame_count_growth_alone():
    """A secondary, cheap sanity check on the same property from a different angle: per-frame
    cost on the SAME extreme page must stay stable whether the loop has 24 or 96 frames (the
    schema's real default), since each frame is an independent, local call -- no per-frame cost
    should grow with how many frames the loop has.
    """
    object_bbox = (100, 100, 140, 140)
    motion = MotionSpec(
        transform_kind=TransformKind.ROTATE, amplitude=25.0, timing=TimingSpec(loop_mode="cycle")
    )
    page_shape = (5062, 720)

    short_loop_ms = _mean_per_frame_ms(page_shape, object_bbox, motion, n_frames=24)
    long_loop_ms = _mean_per_frame_ms(page_shape, object_bbox, motion, n_frames=96)

    assert long_loop_ms < short_loop_ms * 4.0, (
        f"per-frame cost grew {long_loop_ms / short_loop_ms:.1f}x going from a 24-frame to a "
        f"96-frame loop on the same page/object -- per-frame cost should be loop-length-"
        "independent"
    )
