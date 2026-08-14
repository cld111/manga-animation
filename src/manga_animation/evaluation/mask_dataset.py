"""Loader for the Phase 12 semantic-mask validation benchmark

(`configs/phase12_semantic_mask_benchmark.yaml`) -- REAL `SegmentationResult.mask` arrays from
real, previously-completed live GPU runs (Phase 8.3's/Phase 11's own diagnostic captures), with
human-assigned GOOD/BAD ground truth cited to the phase report that established it. See
`docs/decisions/0018-semantic-mask-validation.md` and `docs/phase12-results.md`'s benchmarking
section for the full design and disclosed limitations (small sample size, development-data-only
status -- Workstream 53/54).

Deliberately a new, small schema (`MaskSemanticSample`) rather than squeezed into
`evaluation.dataset.EvalSample`: that type describes a whole-page pipeline run (VLM plan,
render, loop metrics); this describes one already-segmented mask in isolation, with no
pipeline run involved at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from manga_animation.pipeline.types import BBoxPx, ImageArray, MaskArray
from manga_animation.schemas.animation_plan import TransformKind

DEFAULT_MASK_BENCHMARK_PATH = Path("configs/phase12_semantic_mask_benchmark.yaml")

MaskGroundTruth = Literal["good", "bad"]
MaskDifficulty = Literal["typical", "difficult"]


class MaskSemanticSample(BaseModel):
    """One real, human-labeled `(image, mask, bbox, semantic_label)` benchmark entry.

    `mask_path`/`source_page` point at git-ignored, locally-generated files (ADR 0002) that may
    not exist on every checkout -- see `artifacts_available`.
    """

    sample_id: str
    source_page: Path
    mask_path: Path
    semantic_label: str
    bbox_xyxy: tuple[int, int, int, int]
    transform_kind: TransformKind
    ground_truth: MaskGroundTruth
    difficulty: MaskDifficulty = "typical"
    evidence: str = Field(min_length=1)
    provenance: str = Field(min_length=1)

    @property
    def bbox(self) -> BBoxPx:
        x0, y0, x1, y1 = self.bbox_xyxy
        return BBoxPx(x0=x0, y0=y0, x1=x1, y1=y1)

    def artifacts_available(self) -> bool:
        return self.source_page.exists() and self.mask_path.exists()

    def load_image(self) -> ImageArray:
        from PIL import Image

        return np.asarray(Image.open(self.source_page).convert("RGB"))

    def load_mask(self) -> MaskArray:
        image = self.load_image()
        mask = np.load(self.mask_path)
        self.validate_artifacts(image, mask)
        return mask

    def validate_artifacts(self, image: ImageArray, mask: MaskArray) -> None:
        """Validate the on-disk benchmark contract before any method consumes the sample."""
        if mask.ndim != 2:
            raise ValueError(f"{self.sample_id}: mask must be 2D, got {mask.shape}")
        if mask.dtype != np.uint8:
            raise ValueError(f"{self.sample_id}: mask must be uint8, got {mask.dtype}")
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"{self.sample_id}: mask shape {mask.shape} does not match image shape "
                f"{image.shape[:2]}"
            )
        if not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"{self.sample_id}: mask values must be binary uint8 values 0/255")

        x0, y0, x1, y1 = self.bbox_xyxy
        height, width = image.shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"{self.sample_id}: bbox {self.bbox_xyxy} is outside image bounds "
                f"(width={width}, height={height})"
            )
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            raise ValueError(f"{self.sample_id}: mask is empty")
        tight_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if tight_bbox != self.bbox_xyxy:
            raise ValueError(
                f"{self.sample_id}: bbox {self.bbox_xyxy} does not match mask tight bbox "
                f"{tight_bbox}"
            )


def load_mask_semantic_benchmark(
    path: Path = DEFAULT_MASK_BENCHMARK_PATH,
) -> list[MaskSemanticSample]:
    data = yaml.safe_load(path.read_text())
    return [MaskSemanticSample.model_validate(entry) for entry in data["samples"]]
