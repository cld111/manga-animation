"""The serialized contract between `CogVideoXClient` and the isolated `cogvideox_worker.py`.

CogVideoX-5B-I2V generates video from an input image + text prompt (I2V mode). Unlike
AnimateAnything, there is no motion mask input -- the model generates the entire video
from the image and prompt. The spec carries the same filesystem-addressed shape so the
pipeline can persist work directories and provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# CogVideoX-5B-I2V native output: 49 frames @ 8 fps = 6.12s at 720x480.
DEFAULT_NUM_FRAMES = 49
DEFAULT_FPS = 8
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_SEED = 42


class CogVideoXSpec(BaseModel):
    """One inference request for the CogVideoX worker, fully filesystem-addressed.

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
        description="Negative prompt for CogVideoX",
    )

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CogVideoXSpec:
        return cls.model_validate_json(Path(path).read_text())

    def as_manifest(self) -> dict[str, Any]:
        """A JSON-safe record for pipeline manifests / experiment reports (no secrets)."""
        data = self.model_dump()
        data["image_path"] = Path(self.image_path).name
        data["checkpoint_path"] = Path(self.checkpoint_path).name
        return data
