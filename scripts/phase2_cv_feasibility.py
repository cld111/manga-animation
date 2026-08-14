"""Phase 2 feasibility check for the deterministic/kinematic animation stage.

Per the Phase 2 brief, this stage is "primarily a CV implementation stage rather than a
generative-model benchmark". This script is a historical feasibility record; the production
implementation now lives in `src/manga_animation/animation` and `compositing`.

This script deliberately uses a synthetic mask and runs entirely on CPU. Its results validate
CV mechanics in isolation, while current end-to-end evidence is recorded in the phase result
documents and `docs/current-status.md`.

Usage: uv run python scripts/phase2_cv_feasibility.py [--page examples/sample_page_01.png]
OpenCV is part of the base development environment; `uv sync --extra cv` remains accepted as
an optional compatibility command.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

EASING_FUNCS = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "ease_in_out": lambda t: 3 * t**2 - 2 * t**3,
    "sine": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
}

TRANSFORM_KINDS = ["translate", "rotate", "scale", "shear", "mesh_warp", "opacity"]


@dataclass
class FeasibilityResult:
    transform_kind: str
    frames_rendered: int
    mean_transform_ms: float
    max_transform_ms: float
    static_region_max_diff: int
    static_region_exact: bool
    debug_frame_path: str


def _oscillation(t_frac: float, speed: float, phase: float, easing: str) -> float:
    """Sample position in [-1, 1] at t_frac in [0, 1) of the loop, per the schema's

    seamless-loop convention (docs/animation-plan-schema.md): a `sin` of
    `2*pi*(speed*t/duration + phase)` returns exactly to its start value at t=duration when
    `speed` is a whole number of cycles. `easing` reshapes the envelope, not the periodicity.
    """
    raw = math.sin(2 * math.pi * (speed * t_frac + phase))
    # Map raw in [-1, 1] to eased progress in [0, 1] then back, so easing affects the
    # in-cycle feel without breaking the exact periodic return to the start value.
    progress = (raw + 1.0) / 2.0
    eased = EASING_FUNCS[easing](progress)
    return eased * 2.0 - 1.0


def _synthetic_mask(
    shape: tuple[int, int], bbox_norm: tuple[float, float, float, float]
) -> np.ndarray:
    """A historical synthetic mask used by the Phase 2 feasibility experiment.

    bbox_norm is (x, y, w, h) normalized to [0, 1], matching the Animation Plan schema's
    BBox convention (docs/animation-plan-schema.md) so pivot/region resolution mirrors how
    a real ObjectPlan region would be resolved.
    """
    h, w = shape
    x, y, bw, bh = bbox_norm
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + bw) * w), int((y + bh) * h)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Elliptical, not rectangular, so mask edges aren't axis-aligned lines (a rectangle
    # would make every transform kind look artificially clean at the boundary).
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    axes = ((x1 - x0) // 2, (y1 - y0) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _resolve_pivot_px(
    bbox_px: tuple[int, int, int, int], pivot_norm: tuple[float, float]
) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox_px
    px, py = pivot_norm
    return (x0 + px * (x1 - x0), y0 + py * (y1 - y0))


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _affine_frame(
    kind: str, image: np.ndarray, mask: np.ndarray, pivot_px: tuple[float, float], value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (transformed_layer, transformed_mask) for translate/rotate/scale/shear."""
    h, w = mask.shape
    if kind == "translate":
        # amplitude convention: fraction of panel diagonal, direction is a unit vector.
        diag = math.hypot(w, h)
        dx, dy = value * diag * 0.02, 0.0  # small fixed direction (1, 0) for this check
        matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    elif kind == "rotate":
        matrix = cv2.getRotationMatrix2D(pivot_px, value * 8.0, 1.0)  # +-8 degrees at peak
    elif kind == "scale":
        s = 1.0 + value * 0.06  # +-6% at peak
        matrix = cv2.getRotationMatrix2D(pivot_px, 0.0, s)
    elif kind == "shear":
        shear_factor = value * 0.08
        matrix = np.array(
            [[1, shear_factor, -shear_factor * pivot_px[1]], [0, 1, 0]], dtype=np.float32
        )
    else:
        raise ValueError(kind)

    warped_layer = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return warped_layer, warped_mask


def _mesh_warp_frame(
    image: np.ndarray, mask: np.ndarray, value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cloth/hair-style ripple: smooth horizontal displacement, stronger near the mask edge

    farthest from its bbox top (mimics hair swaying from a fixed scalp attachment).
    """
    h, w = mask.shape
    x0, y0, x1, y1 = _bbox_of_mask(mask)
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    local_y = np.clip((map_y - y0) / max(y1 - y0, 1), 0.0, 1.0)
    displacement = value * 0.05 * (x1 - x0) * local_y  # normalized warp strength * region width
    warped_map_x = map_x + displacement
    warped_layer = cv2.remap(
        image, warped_map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_mask = cv2.remap(
        mask, warped_map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0
    )
    return warped_layer, warped_mask


def _composite(
    original: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha_scale: float = 1.0
) -> np.ndarray:
    """Alpha-composite `layer` over a FRESH copy of `original` using `mask` as alpha.

    Building each frame from a fresh copy (never patching the previous frame) is the
    mechanism that makes "pixels outside the mask are untouched" a structural guarantee
    rather than a best-effort one — see docs/architecture.md ("Original Image Is the Source
    of Truth") and the cv-animation skill.
    """
    frame = original.copy()
    alpha = (mask.astype(np.float32) / 255.0 * alpha_scale)[..., None]
    frame = (layer.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    return frame


def run_transform_check(
    kind: str,
    image: np.ndarray,
    mask: np.ndarray,
    pivot_px: tuple[float, float],
    *,
    fps: int,
    duration_s: float,
    speed: float,
    easing: str,
    debug_dir: Path,
    full_sequence_dir: Path | None = None,
) -> FeasibilityResult:
    frame_count = round(duration_s * fps)
    static_outside = mask == 0
    max_diff = 0
    transform_times_ms: list[float] = []
    debug_frame_path = ""

    for i in range(frame_count):
        t_frac = i / frame_count
        value = _oscillation(t_frac, speed=speed, phase=0.0, easing=easing)

        start = time.perf_counter()
        if kind == "opacity":
            layer, layer_mask = image, mask
            alpha_scale = 0.6 + 0.4 * ((value + 1.0) / 2.0)  # 0.6-1.0 fractional opacity swing
            frame = _composite(image, layer, layer_mask, alpha_scale=alpha_scale)
        elif kind == "mesh_warp":
            layer, layer_mask = _mesh_warp_frame(image, mask, value)
            frame = _composite(image, layer, layer_mask)
        else:
            layer, layer_mask = _affine_frame(kind, image, mask, pivot_px, value)
            frame = _composite(image, layer, layer_mask)
        transform_times_ms.append((time.perf_counter() - start) * 1000.0)

        # Static-region preservation: every pixel with exactly zero alpha contribution from
        # the transformed layer must equal the source exactly (see "Original Image Is the
        # Source of Truth"). "Outside" is alpha==0 (layer_mask==0), not an arbitrary
        # threshold — pixels in the interpolated mask's edge ramp (0 < mask < 255) are
        # *intentionally* partially blended, not a preservation violation.
        outside = static_outside if kind == "opacity" else (layer_mask == 0)
        diff = cv2.absdiff(frame, image)
        diff[~outside] = 0
        max_diff = max(max_diff, int(diff.max()))

        if i == frame_count // 4:
            debug_frame_path = str(debug_dir / f"{kind}_frame_{i:04d}.png")
            cv2.imwrite(debug_frame_path, frame)

        if full_sequence_dir is not None:
            full_sequence_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full_sequence_dir / f"frame_{i:04d}.png"), frame)

    return FeasibilityResult(
        transform_kind=kind,
        frames_rendered=frame_count,
        mean_transform_ms=sum(transform_times_ms) / len(transform_times_ms),
        max_transform_ms=max(transform_times_ms),
        static_region_max_diff=max_diff,
        static_region_exact=(max_diff == 0),
        debug_frame_path=debug_frame_path,
    )


def run_seamless_loop_check(
    image: np.ndarray,
    mask: np.ndarray,
    pivot_px: tuple[float, float],
    *,
    fps: int,
    duration_s: float,
) -> dict:
    """Frame 0 vs. a freshly-regenerated frame at t=duration_s, for an integer-speed cycle.

    This is the check the video-rendering/evaluation skills describe as necessary but not
    covered by the schema's speed==integer validation alone (docs/animation-plan-schema.md,
    "The seamless-loop constraint on speed").
    """
    speed = 2.0  # whole-number cycles, as the schema requires under loop.seamless=True

    def render_at(t_frac: float) -> np.ndarray:
        value = _oscillation(t_frac, speed=speed, phase=0.0, easing="sine")
        layer, layer_mask = _affine_frame("rotate", image, mask, pivot_px, value)
        return _composite(image, layer, layer_mask)

    frame_0 = render_at(0.0)
    frame_n = render_at(1.0)  # t = duration_s, i.e. one full loop later
    diff = cv2.absdiff(frame_0, frame_n)
    return {
        "speed_cycles": speed,
        "max_pixel_diff": int(diff.max()),
        "exact_loop": bool(diff.max() == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=Path("examples/sample_page_01.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument(
        "--debug-dir", type=Path, default=Path("outputs/debug/phase2-cv-feasibility")
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument(
        "--full-sequence-kind",
        default="rotate",
        choices=TRANSFORM_KINDS,
        help="Also dump every frame for this one transform kind, as fixture input for the "
        "video-rendering feasibility check (frame_%%04d.png, ready for ffmpeg).",
    )
    parser.add_argument("--frames-dir", type=Path, default=Path("outputs/frames/phase2-demo"))
    args = parser.parse_args()

    if not args.page.exists():
        raise SystemExit(
            f"{args.page} not found — fetch a sample page first: "
            "uv run python scripts/fetch_sample_pages.py"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.page))
    if image is None:
        raise SystemExit(f"cv2 could not decode {args.page}")
    h, w = image.shape[:2]

    # Synthetic placeholder region (no real segmentation exists yet) — see module docstring.
    bbox_norm = (0.30, 0.04, 0.40, 0.16)
    mask = _synthetic_mask((h, w), bbox_norm)
    bbox_px = _bbox_of_mask(mask)
    pivot_px = _resolve_pivot_px(bbox_px, pivot_norm=(0.5, 0.0))  # top-center, e.g. hair-from-scalp

    results = [
        run_transform_check(
            kind,
            image,
            mask,
            pivot_px,
            fps=args.fps,
            duration_s=args.duration_s,
            speed=1.0 if kind != "mesh_warp" else 2.0,
            easing="sine",
            debug_dir=args.debug_dir,
            full_sequence_dir=(args.frames_dir if kind == args.full_sequence_kind else None),
        )
        for kind in TRANSFORM_KINDS
    ]
    loop_check = run_seamless_loop_check(
        image, mask, pivot_px, fps=args.fps, duration_s=args.duration_s
    )

    summary = {
        "source_page": str(args.page),
        "image_size": [w, h],
        "fps": args.fps,
        "duration_s": args.duration_s,
        "device": "cpu",
        "opencv_version": cv2.__version__,
        "transform_checks": [asdict(r) for r in results],
        "seamless_loop_check": loop_check,
        "full_frame_sequence": {
            "transform_kind": args.full_sequence_kind,
            "dir": str(args.frames_dir),
            "frame_count": round(args.duration_s * args.fps),
        },
    }

    out_path = args.out_dir / "phase2_cv_feasibility.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {out_path}")
    for r in results:
        status = (
            "OK (bit-exact)"
            if r.static_region_exact
            else f"FAIL (max_diff={r.static_region_max_diff})"
        )
        print(
            f"  {r.transform_kind:10s} static-region preservation: {status:22s} "
            f"mean {r.mean_transform_ms:6.2f} ms/frame  max {r.max_transform_ms:6.2f} ms/frame"
        )
    loop_status = (
        "OK (bit-exact)"
        if loop_check["exact_loop"]
        else f"FAIL max_diff={loop_check['max_pixel_diff']}"
    )
    print(f"  seamless loop (rotate, speed={loop_check['speed_cycles']}): {loop_status}")


if __name__ == "__main__":
    main()
