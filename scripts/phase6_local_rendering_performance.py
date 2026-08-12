"""Phase 6 performance evidence: local-region rendering cost vs. the old full-page approach.

Purpose (see docs/phase6-results.md for the full write-up): confirm, with real measurements
on synthetic data, that `animation.transforms.generate_transformed_layer`'s Phase 6 local-ROI
rewrite makes the expensive `cv2.warpAffine`/`cv2.remap` work scale with the animated OBJECT
region rather than the full PAGE — the architectural property Phase 6 targets, not merely "it
got faster." A verbatim copy of the pre-Phase-6 full-page implementation is kept below purely
as a timing baseline (the correctness equivalence between the two is already established by
`tests/test_animation.py`'s `test_localized_transform_matches_full_page_reference` — this
script is NOT a correctness check).

This is a deterministic, local, non-GPU, non-ML script — synthetic images/masks only, no
sample pages, no model inference — consistent with "Step 2 should first be validated entirely
with deterministic CPU/OpenCV/NumPy tests" (Phase 6 brief) and ADR 0003/0004's remote-GPU
policy. It is NOT a new evaluation framework or batch-processing infrastructure: one script,
one page/object synthesized per measurement, printed results only.

Timings are wall-clock and therefore environment-dependent — this script reports measurements
as evidence (see docs/phase6-results.md for one real, captured run), not a pass/fail gate; it
does not assert on the numbers itself.

Usage: uv run python scripts/phase6_local_rendering_performance.py
Requires the `cv` optional dependency group: `uv sync --extra cv`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from manga_animation.animation.curves import sample_motion_value
from manga_animation.animation.transforms import (
    _affine_dest_roi,
    _affine_matrix,
    bbox_of_mask,
    generate_transformed_layer,
    resolve_pivot_px,
)
from manga_animation.compositing import composite_frame_stack
from manga_animation.pipeline.types import BBoxPx, Layer
from manga_animation.schemas.animation_plan import MotionSpec, TimingSpec, TransformKind, Vector2


def _old_full_page_generate_transformed_layer(
    image: np.ndarray,
    mask: np.ndarray,
    motion: MotionSpec,
    panel_bbox_px: BBoxPx,
    page_shape: tuple[int, int],
    t_frac: float,
    *,
    loop_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-Phase-6 `generate_transformed_layer`, kept verbatim as a timing baseline only (see
    module docstring) -- always warps/remaps the WHOLE page, regardless of object size.
    """
    t_s = t_frac * loop_duration_s
    value = sample_motion_value(motion, t_s, loop_duration_s)
    object_bbox_px = bbox_of_mask(mask)
    kind = motion.transform_kind
    h, w = mask.shape

    if kind == TransformKind.OPACITY:
        alpha_scale = min(max(1.0 + value * motion.amplitude, 0.0), 1.0)
        scaled_mask = np.clip(mask.astype(np.float32) * alpha_scale, 0, 255).astype(np.uint8)
        return image, scaled_mask

    if kind == TransformKind.MESH_WARP:
        x0, y0, x1, y1 = object_bbox_px.as_xyxy()
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        direction = motion.direction
        dir_x = direction.x if direction is not None else 1.0
        dir_y = direction.y if direction is not None else 0.0
        if abs(dir_y) >= abs(dir_x):
            local = np.clip((map_y - y0) / max(y1 - y0, 1), 0.0, 1.0)
        else:
            local = np.clip((map_x - x0) / max(x1 - x0, 1), 0.0, 1.0)
        strength = value * motion.amplitude * max(x1 - x0, y1 - y0)
        warped_map_x = map_x + strength * dir_x * local
        warped_map_y = map_y + strength * dir_y * local
        warped_layer = cv2.remap(
            image,
            warped_map_x,
            warped_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.remap(
            mask, warped_map_x, warped_map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
        )
        return warped_layer, warped_mask

    pivot_px = resolve_pivot_px(motion.pivot, object_bbox_px, panel_bbox_px, page_shape)
    panel_diag_px = math.hypot(panel_bbox_px.width, panel_bbox_px.height)
    matrix = _affine_matrix(kind, value, motion, pivot_px, panel_diag_px)
    warped_layer = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return warped_layer, warped_mask


@dataclass
class TimingResult:
    label: str
    page_shape: tuple[int, int]
    object_px: int
    frames_measured: int
    old_mean_ms: float
    new_mean_ms: float
    speedup: float


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


def _time_generate_transformed_layer(
    label: str,
    page_shape: tuple[int, int],
    object_bbox: tuple[int, int, int, int],
    motion: MotionSpec,
    n_frames: int,
) -> TimingResult:
    image, mask = _make_page(page_shape, object_bbox)
    panel_bbox = BBoxPx(x0=0, y0=0, x1=page_shape[1], y1=page_shape[0])
    t_fracs = [i / n_frames for i in range(n_frames)]

    start = time.perf_counter()
    for t_frac in t_fracs:
        _old_full_page_generate_transformed_layer(
            image, mask, motion, panel_bbox, page_shape, t_frac, loop_duration_s=4.0
        )
    old_elapsed_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    for t_frac in t_fracs:
        generate_transformed_layer(
            image, mask, motion, panel_bbox, page_shape, t_frac, loop_duration_s=4.0
        )
    new_elapsed_ms = (time.perf_counter() - start) * 1000.0

    x0, y0, x1, y1 = object_bbox
    return TimingResult(
        label=label,
        page_shape=page_shape,
        object_px=(x1 - x0) * (y1 - y0),
        frames_measured=n_frames,
        old_mean_ms=old_elapsed_ms / n_frames,
        new_mean_ms=new_elapsed_ms / n_frames,
        speedup=(old_elapsed_ms / new_elapsed_ms) if new_elapsed_ms > 0 else float("inf"),
    )


@dataclass
class WarpOnlyTimingResult:
    label: str
    page_shape: tuple[int, int]
    roi_px: int
    n_calls: int
    old_full_page_warp_mean_ms: float
    new_roi_only_warp_mean_ms: float
    speedup: float


def _time_warp_only(
    label: str,
    page_shape: tuple[int, int],
    object_bbox: tuple[int, int, int, int],
    amplitude_deg: float,
    n_calls: int,
) -> WarpOnlyTimingResult:
    """Isolates the raw `cv2.warpAffine` cost -- full-page `dsize` vs. ROI-restricted `dsize`
    for the identical matrix -- from `generate_transformed_layer`'s surrounding full-page
    zero-allocation (a separate, architecturally-required cost; see the "place transformed
    result into page coordinates" step in docs/phase6-results.md). This isolates exactly the
    claim Phase 6 makes: the expensive interpolation work itself scales with the animated
    region, not the page.
    """
    image, mask = _make_page(page_shape, object_bbox)
    h, w = page_shape
    object_bbox_px = BBoxPx(*object_bbox)
    matrix = cv2.getRotationMatrix2D(
        ((object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2),
        amplitude_deg,
        1.0,
    )
    roi = _affine_dest_roi(object_bbox_px, matrix, page_shape)
    assert roi is not None
    rx0, ry0, rx1, ry1 = roi
    matrix_roi = matrix.copy()
    matrix_roi[:, 2] -= (rx0, ry0)

    start = time.perf_counter()
    for _ in range(n_calls):
        cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    old_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    for _ in range(n_calls):
        cv2.warpAffine(
            image,
            matrix_roi,
            (rx1 - rx0, ry1 - ry0),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
    new_ms = (time.perf_counter() - start) * 1000.0

    return WarpOnlyTimingResult(
        label=label,
        page_shape=page_shape,
        roi_px=(rx1 - rx0) * (ry1 - ry0),
        n_calls=n_calls,
        old_full_page_warp_mean_ms=old_ms / n_calls,
        new_roi_only_warp_mean_ms=new_ms / n_calls,
        speedup=(old_ms / new_ms) if new_ms > 0 else float("inf"),
    )


@dataclass
class MultiObjectTimingResult:
    label: str
    page_shape: tuple[int, int]
    n_objects: int
    frame_count: int
    total_composite_ms: float
    mean_composite_ms_per_frame: float


def _time_multi_object_compositing(
    label: str, page_shape: tuple[int, int], n_objects: int, frame_count: int
) -> MultiObjectTimingResult:
    h, w = page_shape
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    region_size = 60
    layers = []
    for i in range(n_objects):
        y0 = (i * 97) % max(h - region_size, 1)
        x0 = (i * 131) % max(w - region_size, 1)
        y1, x1 = y0 + region_size, x0 + region_size
        frames = []
        for _f in range(frame_count):
            layer_image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 255
            frames.append((layer_image, mask))
        layers.append(Layer(object_id=f"obj_{i}", frames=tuple(frames), z_order=i))

    start = time.perf_counter()
    for frame_index in range(frame_count):
        composite_frame_stack(image, layers, frame_index=frame_index)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return MultiObjectTimingResult(
        label=label,
        page_shape=page_shape,
        n_objects=n_objects,
        frame_count=frame_count,
        total_composite_ms=elapsed_ms,
        mean_composite_ms_per_frame=elapsed_ms / frame_count,
    )


def main() -> None:
    small_object = (100, 100, 140, 140)  # 40x40 object, fixed size across all page scales
    rotate_motion = MotionSpec(
        transform_kind=TransformKind.ROTATE,
        amplitude=25.0,
        timing=TimingSpec(loop_mode="cycle"),
    )

    print("=== Raw cv2.warpAffine cost only: full-page dsize vs. ROI-restricted dsize ===")
    warp_only_results = [
        _time_warp_only("warp_only_600x800", (600, 800), small_object, 25.0, 200),
        _time_warp_only("warp_only_720x5062", (5062, 720), small_object, 25.0, 200),
        _time_warp_only("warp_only_1100x6613", (6613, 1100), small_object, 25.0, 200),
    ]
    for r in warp_only_results:
        print(json.dumps(asdict(r)))

    print("\n=== Local-region cost vs. page size (fixed small object, ROTATE) ===")
    page_size_results = [
        _time_generate_transformed_layer(
            "baseline_600x800", (600, 800), small_object, rotate_motion, 48
        ),
        _time_generate_transformed_layer(
            "extreme_aspect_720x5062", (5062, 720), small_object, rotate_motion, 48
        ),
        _time_generate_transformed_layer(
            "extreme_aspect_1100x6613", (6613, 1100), small_object, rotate_motion, 48
        ),
    ]
    for r in page_size_results:
        print(json.dumps(asdict(r)))

    print("\n=== Local-region cost vs. transform kind (720x5062 page, small object) ===")
    kind_results = []
    mesh_motion = MotionSpec(
        transform_kind=TransformKind.MESH_WARP,
        amplitude=0.3,
        direction=Vector2(x=0.8, y=0.6),
        timing=TimingSpec(loop_mode="cycle"),
    )
    translate_motion = MotionSpec(
        transform_kind=TransformKind.TRANSLATE,
        amplitude=0.15,
        direction=Vector2(x=1.0, y=0.3),
        timing=TimingSpec(loop_mode="cycle"),
    )
    for label, motion in (
        ("rotate", rotate_motion),
        ("mesh_warp", mesh_motion),
        ("translate", translate_motion),
    ):
        r = _time_generate_transformed_layer(f"kind_{label}", (5062, 720), small_object, motion, 48)
        kind_results.append(r)
        print(json.dumps(asdict(r)))

    print("\n=== Multiple simultaneously-animated objects (1100x6613 page) ===")
    multi_object_results = [
        _time_multi_object_compositing("n_objects_1", (6613, 1100), 1, 24),
        _time_multi_object_compositing("n_objects_5", (6613, 1100), 5, 24),
    ]
    for r in multi_object_results:
        print(json.dumps(asdict(r)))

    print("\n=== Larger frame counts (720x5062 page, small object, ROTATE) ===")
    frame_count_results = [
        _time_generate_transformed_layer("frames_24", (5062, 720), small_object, rotate_motion, 24),
        _time_generate_transformed_layer("frames_96", (5062, 720), small_object, rotate_motion, 96),
        _time_generate_transformed_layer(
            "frames_240", (5062, 720), small_object, rotate_motion, 240
        ),
    ]
    for r in frame_count_results:
        print(json.dumps(asdict(r)))


if __name__ == "__main__":
    main()
