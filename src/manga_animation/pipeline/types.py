"""Typed contracts passed between pipeline stages.

`docs/pipeline.md` names the stage sequence and each stage's `src/manga_animation/<package>`
home; this module is the thing that actually lets them be wired together without leaking
model-specific tensors/APIs across stage boundaries (see "Interfaces" in the Phase 3.1 brief
and "Model Abstraction" in `docs/architecture.md`). Every adapter (Grounding DINO, SAM 2.1,
LaMa, ...) stays localized to its own stage module and converts to/from these types at the
boundary.

## Image convention

All pixel data in this module (and everywhere downstream of grounding) is **RGB**, `uint8`,
shape `(H, W, 3)` for images/frames and `(H, W)` for masks (`0-255`, not boolean — see the
`evaluation`/`cv-animation` skills' note that a mask's *alpha==0* boundary, not an arbitrary
threshold, is what "outside the mask" means). This matches `PIL.Image` (used by the
VLM/grounding/segmentation/inpainting adapters, all HF-ecosystem, all RGB-native) rather than
OpenCV's default `BGR` — stage code that calls `cv2.warpAffine`/`cv2.remap`/etc. is safe either
way (those ops are channel-order agnostic), but any code that touches `cv2.imread`/`cv2.imwrite`
directly must convert at that boundary, since those two (and only those two) assume BGR. Use
`PIL.Image.open(...).convert("RGB")` / `Image.fromarray(...)` for file I/O instead, wherever
possible, to avoid the conversion needing to happen at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from manga_animation.schemas.animation_plan import BBox

ImageArray = np.ndarray
"""RGB uint8 array, shape (H, W, 3). See module docstring for the channel-order convention."""

MaskArray = np.ndarray
"""uint8 array, shape (H, W), values 0-255 (an alpha channel, not a boolean mask)."""

# Coverage-fraction bounds a grounded region must fall within to be plausibly "a specific
# object" rather than noise or a false-positive "select everything" region, as a fraction of
# the full source image's pixel count. Originally `segmentation/segment.py`'s mask-coverage
# check (see its own comment: deliberately permissive rather than tuned per-object-class,
# since no real mask data exists yet to tune against — ADR 0005's "no visual QA done yet").
# Shared here so `validation/validate.py`'s pre-segmentation bbox-plausibility check reuses the
# exact same reasoning/values on a grounding bbox instead of inventing a second, uncalibrated
# number — a bbox is always >= its eventual tight mask, so the same bounds are, if anything,
# more permissive than the segmentation check that already uses them (see the Phase 3.2 brief's
# "do not make arbitrary confidence thresholds without calibration/evidence").
MIN_OBJECT_COVERAGE_FRACTION = 0.0001
MAX_OBJECT_COVERAGE_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class BBoxPx:
    """A pixel-space axis-aligned box, in the source image's actual resolution.

    Distinct from the schema's `BBox` (normalized [0, 1], resolution-independent, lives on
    the `AnimationPlan`) — this is what grounding produces *from* a plan's `ObjectPlan`, per
    "It is deliberately pixel-free" in docs/animation-plan-schema.md.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    score: float | None = None

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"degenerate bbox: ({self.x0}, {self.y0}, {self.x1}, {self.y1})")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


def normalized_bbox_to_px(bbox: BBox, page_width: int, page_height: int) -> BBoxPx:
    """Convert a schema-space normalized (`[0, 1]`) `BBox` to pixel space for a page of the

    given size — the canonical "panel-local coordinates -> page coordinates" direction Phase
    3.3 needs to be deterministic and tested (see `docs/decisions/0007-panel-aware-analysis.md`).
    The origin is floored and the far edge is ceiled (not both floored, as a naive `int(...)`
    cast would) so a normalized box that is meant to reach a page edge doesn't lose a row/column
    of pixels to truncation; both are then clamped to the page's actual bounds.
    """
    x0 = max(0, math.floor(bbox.x * page_width))
    y0 = max(0, math.floor(bbox.y * page_height))
    x1 = min(page_width, math.ceil((bbox.x + bbox.width) * page_width))
    y1 = min(page_height, math.ceil((bbox.y + bbox.height) * page_height))
    return BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1)


def bbox_px_to_normalized(bbox: BBoxPx, page_width: int, page_height: int) -> BBox:
    """Convert a pixel-space `BBoxPx` (e.g. a detected `PanelCandidate.bbox`) to a schema-space

    normalized `BBox` relative to the full page — the inverse direction of
    `normalized_bbox_to_px`. Not a bit-exact round trip (both directions round to integer
    pixels), but stable to within one pixel, which is what `PanelPlan.bbox` needs: something
    deterministic and reproducible, not sub-pixel-precise.
    """
    x = bbox.x0 / page_width
    y = bbox.y0 / page_height
    width = (bbox.x1 - bbox.x0) / page_width
    height = (bbox.y1 - bbox.y0) / page_height
    # Clamp for float rounding at the page edge (e.g. x + width landing at 1.0000000002,
    # which BBox's own bounds validator would otherwise reject).
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    return BBox(x=x, y=y, width=width, height=height)


PanelSource = Literal["gutter_xy_cut", "fallback_full_page"]
"""Where a `PanelCandidate` came from: `"gutter_xy_cut"` for a real detected panel (see

`analysis/panels.py`), `"fallback_full_page"` for the degenerate "no internal gutters found,
whole page is one panel" case, which is a valid splash-page read (see the `manga-analysis`
skill), not a detector failure — distinguished from a genuine zero-panel result (which returns
an empty list rather than a candidate at all; see `analysis/panels.py::detect_panels`).
"""


@dataclass(frozen=True, slots=True)
class PanelCandidate:
    """One detected panel/region on a page, in page-space pixel coordinates.

    `crop` always corresponds exactly to `bbox` (checked below) — any context margin around a
    detected gutter boundary is baked into `bbox` itself before this type is constructed, not
    tracked as a second, separately-offset region (see ADR 0007's "Structured output").
    """

    id: str
    bbox: BBoxPx
    crop: ImageArray
    confidence: float
    source: PanelSource
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"PanelCandidate confidence must be within [0, 1], got {self.confidence}"
            )
        expected = (self.bbox.height, self.bbox.width)
        if self.crop.shape[:2] != expected:
            raise ValueError(
                f"PanelCandidate crop shape {self.crop.shape[:2]} does not match its bbox "
                f"extent {expected} -- crop must be exactly image[bbox]"
            )


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Where one `ObjectPlan.object_id` actually is in the source image."""

    object_id: str
    bbox: BBoxPx
    model_id: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Explicit ACCEPT/REJECT diagnostics for one grounding candidate, produced by
    `src/manga_animation/validation` between grounding and segmentation (Phase 3.2).

    A technically valid detection (clears the grounding model's own score threshold, lands
    inside the image) is not the same thing as a semantically correct one — this is the
    structured record of *why* a candidate was accepted or rejected, so a rejection is always
    explainable and never a silent drop. See docs/decisions/0006-grounding-target-validation.md.
    """

    object_id: str
    candidate_rank: int
    """0-based position of this candidate within its object's ranked grounding candidates."""
    accepted: bool
    grounding_score: float | None
    bbox_area_fraction: float
    bbox_plausible: bool
    semantic_match: bool | None
    """VLM's yes/no read on whether the cropped region depicts the target. `None` when the
    bbox-plausibility pre-filter rejected the candidate before the (more expensive) VLM crop
    check ran — see `bbox_plausible`/`reason` for why."""
    semantic_confidence: float | None
    reason: str
    model_id: str
    transform_compatible: bool | None = None
    """Phase 3.3.1: whether the candidate's bbox geometry is safe for the plan's specific
    `transform_kind` (see `validation/transform_geometry.py`) — a semantically-correct region
    can still be geometrically unsafe to animate (e.g. a bbox too large to `rotate` without
    visibly swinging the whole panel, not just the intended object; see
    docs/decisions/0008-transform-aware-target-validation.md). `None` when this check was never
    reached (bbox-implausible or semantic-mismatch already rejected the candidate first, or the
    object has no `motion` to check against)."""


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """A pixel-accurate mask for one grounded object.

    `mask` is full-source-image-shape (not cropped to `bbox`) so downstream compositing never
    has to re-derive an offset — see "Local Modification" in docs/architecture.md for why the
    mask's *content* should still be tight around the object even though the array isn't
    cropped.
    """

    object_id: str
    mask: MaskArray
    bbox: BBoxPx
    model_id: str
    iou_score: float | None = None


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """Replacement pixels for the hole one object's motion reveals.

    `filled_pixels` and `hole_mask` are both normalized to the source image's exact geometry
    (`(H, W[, 3])`) before this type is constructed — never the raw, possibly
    differently-sized output of an inpainting model. See the "Reconstruction" section of the
    Phase 3.1 brief and ADR 0005's LaMa pixel-alignment finding: this normalization is a hard
    requirement, enforced at the boundary of `src/manga_animation/reconstruction`, not left to
    the compositing stage to work around.
    """

    object_id: str
    hole_mask: MaskArray
    filled_pixels: ImageArray
    model_id: str

    def __post_init__(self) -> None:
        if self.hole_mask.shape != self.filled_pixels.shape[:2]:
            raise ValueError(
                f"hole_mask shape {self.hole_mask.shape} does not match filled_pixels "
                f"shape {self.filled_pixels.shape[:2]} — reconstruction output must be "
                "normalized to source geometry before this type is constructed"
            )


@dataclass(frozen=True, slots=True)
class Layer:
    """One independently-transformable animated object's full per-frame footprint across the

    loop -- the formalized "layer" `docs/pipeline.md`'s "Layer decomposition" stage names
    (Phase 4, see docs/decisions/0010-multi-object-layer-decomposition.md). Before this type
    existed, `animation.generate_transformed_layer` returned a raw `(ImageArray, MaskArray)`
    tuple per frame and `compositing.composite_frame` only ever consumed exactly one such pair
    per frame -- `run_pipeline` never animated more than one `ObjectPlan` at a time (Phase
    3.1-3.3.x's documented, deliberate scope limit; see docs/phase3.2-results.md's "kept, by
    design" note). This type and `compositing.composite_frame_stack` are the minimal
    formalization needed to composite more than one simultaneously-animated object correctly.
    """

    object_id: str
    frames: tuple[tuple[ImageArray, MaskArray], ...]
    """One `(transformed_image, transformed_mask)` pair per frame index, same order/count as

    the plan's `LoopSpec.frame_count` -- `generate_transformed_layer`'s per-frame output,
    collected across the whole loop rather than consumed one frame at a time."""
    z_order: int
    """Compositing order: lower is drawn first (further back), higher drawn last (on top).

    Ties are broken by `object_id` (lexicographic) for full determinism. Populated from each
    object's `MotionType` -- PRIMARY on top (the reader's intended focus, per the analysis
    prompt's own definition of "primary"), then SECONDARY, then MICRO -- a simple, documented
    rule, not a depth/occlusion inference this project has no real evidence to support (see
    ADR 0010's "Open questions")."""

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError(f"Layer {self.object_id!r} has zero frames")


@dataclass(slots=True)
class FrameSequence:
    """A rendered, loop-ready sequence of composited frames, not yet encoded."""

    frames: list[ImageArray]
    fps: int

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("FrameSequence must contain at least one frame")

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class LoopMetrics:
    """Numeric evidence for the seamless-loop guarantee (see "Deterministic First" in
    docs/architecture.md), not just a pass/fail bit.

    Computed by `rendering.encode.compute_loop_metrics` from a real decoded/sampled frame
    sequence and attached to every `RenderResult` this stage produces (Phase 8 -- previously
    this same computation existed but was private, discarded after being logged, and had been
    re-implemented ad hoc, uncommitted, by every session that needed the actual numbers post
    hoc; see docs/phase7-results.md section 6.3's `ordinary_adjacent_step`/`wrap_step` figures
    for exactly that gap).

    Two algorithmically independent signals, per the Phase 8 brief's explicit instruction not
    to rely solely on raw pixel equality when judging loop quality:

    - Pixel-level (`*_mean_abs_diff`): does the last-frame-to-first-frame ("wrap") transition
      move roughly as many total pixel values as an ordinary adjacent-frame step does? (Not
      `frame[0] == frame[-1]` -- see `compute_loop_metrics`'s docstring for why exact equality
      is the wrong test for a periodically-sampled sequence.)
    - Structural (`*_ssim`): does the wrap transition preserve as much local structure
      (luminance/contrast/pattern correlation, not just raw magnitude) as an ordinary step does?
      A magnitude-only check cannot distinguish "a small further step along the same motion"
      from "a similarly-sized but structurally unrelated jump" -- this is the second,
      independent check that distinction needs.
    """

    ordinary_adjacent_step_mean_abs_diff: float
    wrap_step_mean_abs_diff: float
    wrap_step_within_2x_ordinary: bool
    ordinary_adjacent_step_ssim: float
    wrap_step_ssim: float
    wrap_ssim_within_tolerance: bool

    @property
    def seamless(self) -> bool:
        """Both independent checks must agree the wrap transition is unremarkable -- either one
        alone flagging a problem is enough to withhold the seamless claim."""
        return self.wrap_step_within_2x_ordinary and self.wrap_ssim_within_tolerance


@dataclass(frozen=True, slots=True)
class RenderResult:
    """The final encoded video, plus what was actually measured about it (not just the intent)."""

    output_path: Path
    frame_count: int
    fps: float
    resolution: tuple[int, int]
    duration_s: float
    codec: str
    pixel_format: str
    seamless_loop_verified: bool
    loop_metrics: LoopMetrics | None = None
    """`None` only when there weren't enough frames (<3) to compute a meaningful comparison --
    see `compute_loop_metrics`. Whenever this is not `None`, `seamless_loop_verified` is exactly
    `loop_metrics.seamless`."""


Stage = Literal[
    "analysis",
    "grounding",
    "validation",
    "segmentation",
    "reconstruction",
    "animation",
    "compositing",
    "rendering",
]


@dataclass
class PipelineStageError(Exception):
    """Raised (not swallowed) when a stage fails, carrying what the Phase 3.1 failure policy

    requires be reported: failing stage, input, error, and a best-effort root-cause/fix
    classification. `orchestrator.run_pipeline` lets this propagate rather than converting a
    failed run into a false PASS — see the "Failure policy" section of the Phase 3.1 brief.
    """

    stage: Stage
    input_ref: str
    detail: str
    root_cause: str | None = None
    architectural: bool | None = None
    proposed_fix: str | None = None

    def __str__(self) -> str:
        return f"[{self.stage}] {self.detail} (input={self.input_ref})"
