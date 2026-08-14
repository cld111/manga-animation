"""Grounding DINO client: text-prompted object detection.

Third-party model imports (`torch`, `transformers`) are localized here and lazy-imported
inside methods — this module must stay importable on a machine with no GPU stack installed
(the local dev machine, per ADR 0003/CLAUDE.md; only the remote Kaggle/Jupyter GPU worker has
`torch`/`transformers`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from manga_animation.pipeline.types import ImageArray


@dataclass(frozen=True, slots=True)
class Detection:
    """One candidate box a `GroundingClient` found for a text prompt."""

    label: str
    score: float
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1), pixel coords in the input image


def _detections_from_scores_boxes_labels(
    scores: Sequence[float],
    boxes: Sequence[Sequence[float]],
    text_labels: Sequence[str],
    fallback_label: str,
) -> list[Detection]:
    """Pure assembly logic, deliberately separated from `GroundingDinoClient.detect`'s

    tensor/model handling so it's unit-testable without `torch` installed (see this module's
    docstring). `scores`/`boxes` are the two fields confirmed, by direct reproduction on a real
    Kaggle run (Phase 3.2's first real end-to-end run), to always be the same length as each
    other. `text_labels` is NOT reliably the same length -- a real, reproduced zero-detection
    case returned `scores`/`boxes` of length 0 but `text_labels=['']` (a length-1 placeholder).
    Pull the label opportunistically by index rather than zipping a third, unreliably-aligned
    sequence against the two that are actually guaranteed to match.
    """
    detections = []
    for i, (score, box) in enumerate(zip(scores, boxes, strict=True)):
        label = text_labels[i] if i < len(text_labels) else fallback_label
        x0, y0, x1, y1 = (int(v) for v in box)
        detections.append(Detection(label=str(label), score=float(score), box=(x0, y0, x1, y1)))
    return detections


class GroundingClient(Protocol):
    model_id: str

    def load(self) -> None: ...
    def detect(self, image: ImageArray, text_prompt: str) -> list[Detection]: ...
    def unload(self) -> None: ...


class GroundingDinoClient:
    """Wraps `grounding-dino-swin-l` via `transformers`.

    API confirmed on a real Kaggle T4x2 run (ADR 0005): `post_process_grounded_object_detection`
    takes `threshold=` (not `box_threshold=`, which `transformers` 5.0.0 renamed).
    """

    model_id = "grounding-dino-swin-l"

    def __init__(self, source: str, device: str, dtype: str):
        self.source, self.device, self.dtype = source, device, dtype
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        if self.model is not None:
            return  # idempotent -- ModelStage may re-enter a stage that is already loaded
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.source)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device)
        self.model.eval()

    def detect(self, image: ImageArray, text_prompt: str) -> list[Detection]:
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.25,
            text_threshold=0.2,
            target_sizes=[(image.shape[0], image.shape[1])],
        )
        result = results[0]
        text_labels = result.get("text_labels", result.get("labels", []))
        return _detections_from_scores_boxes_labels(
            scores=result["scores"].tolist(),
            boxes=result["boxes"].tolist(),
            text_labels=[str(t) for t in text_labels],
            fallback_label=text_prompt,
        )

    def unload(self) -> None:
        import torch

        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
