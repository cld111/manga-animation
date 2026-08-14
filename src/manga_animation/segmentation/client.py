"""SAM 2.1 client: box-prompted mask candidate generation.

Lazy-imports `torch`/`transformers` inside methods — see the same note in
`manga_animation.grounding.client`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from manga_animation.pipeline.types import BBoxPx, ImageArray, MaskArray


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    """One candidate mask SAM proposed for a box prompt, full-source-image-shape."""

    mask: MaskArray  # uint8, (H, W), values 0/255
    iou_score: float


class SegmentationClient(Protocol):
    model_id: str

    def load(self) -> None: ...
    def segment(self, image: ImageArray, box: BBoxPx) -> list[MaskCandidate]: ...
    def unload(self) -> None: ...


class Sam21Client:
    """Wraps `sam2.1-hiera-base` via `transformers`.

    API confirmed on a real Kaggle T4 run (ADR 0005): `post_process_masks(masks,
    original_sizes)` takes no `reshaped_input_sizes` argument on this `transformers` version.

    # VERIFY: the exact rank/order of `outputs.pred_masks` / `outputs.iou_scores` (candidates
    # per box, per image) against the installed `transformers` version at real-run time — this
    # follows the documented Sam2Model output convention (batch, num_boxes, num_candidates,
    # H, W) / (batch, num_boxes, num_candidates), consistent with the project's existing
    # practice of flagging unverified exact tensor shapes (see scripts/phase2_kaggle_benchmark.py's
    # own `# VERIFY:` comments) rather than asserting confidence this assistant can't back up
    # without a live run.
    """

    model_id = "sam2.1-hiera-base"

    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        if self.model is not None:
            return  # idempotent -- ModelStage may re-enter a stage that is already loaded
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.processor = Sam2Processor.from_pretrained(self.source)
        self.model = Sam2Model.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def segment(self, image: ImageArray, box: BBoxPx) -> list[MaskCandidate]:
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        input_boxes = [[[box.x0, box.y0, box.x1, box.y1]]]  # (image, box, coords)
        inputs = self.processor(pil_image, input_boxes=input_boxes, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )
        # masks[0]: (num_boxes, num_candidates, H, W) for image 0 -> our one box is masks[0][0]
        candidate_masks = masks[0][0]
        scores = outputs.iou_scores.cpu()[0][0]

        candidates = []
        for i in range(candidate_masks.shape[0]):
            binary = (candidate_masks[i].numpy() > 0).astype(np.uint8) * 255
            candidates.append(MaskCandidate(mask=binary, iou_score=float(scores[i])))
        return candidates

    def unload(self) -> None:
        import torch

        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
