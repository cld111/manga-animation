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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

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
