"""The serialized contract between `Wan2Client` and the isolated `wan2_worker.py`.

Wan2.2-TI2V-5B generates video from an input image + text prompt (I2V mode). Unlike
AnimateAnything, there is no motion mask input -- the model generates the entire video
from the image and prompt. The spec carries the same filesystem-addressed shape so the
pipeline can persist work directories and provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Wan2.2-TI2V-5B native output: 121 frames @ 24 fps = 5.04s at 720P.
DEFAULT_NUM_FRAMES = 121
DEFAULT_FPS = 24
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_SEED = 42


class Wan2Spec(BaseModel):
    """One inference request for the Wan2.2 worker, fully filesystem-addressed.

    The worker needs no knowledge of this project's pipeline: it receives an image,
    a prompt and hyper-parameters, and produces frame PNGs in `output_dir`.
    """

    image_path: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    device: str = Field(default="cuda", description="CUDA device the worker loads onto")
    num_frames: int = Field(gt=0, default=DEFAULT_NUM_FRAMES)
    fps: int = Field(gt=0, default=DEFAULT_FPS)
    num_inference_steps: int = Field(gt=0, default=DEFAULT_NUM_INFERENCE_STEPS)
    guidance_scale: float = Field(gt=0.0, default=DEFAULT_GUIDANCE_SCALE)
    seed: int = Field(default=DEFAULT_SEED)
    negative_prompt: str = Field(
        default="static, blurry, low quality, worst quality",
        description="Negative prompt for Wan2.2",
    )
    height: int = Field(gt=0, default=704, description="Output height (must be divisible by 16)")
    width: int = Field(gt=0, default=1280, description="Output width (must be divisible by 16)")

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> Wan2Spec:
        return cls.model_validate_json(Path(path).read_text())

    def as_manifest(self) -> dict[str, Any]:
        """A JSON-safe record for pipeline manifests / experiment reports (no secrets)."""
        data = self.model_dump()
        data["image_path"] = Path(self.image_path).name
        data["checkpoint_path"] = Path(self.checkpoint_path).name
        return data
