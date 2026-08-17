"""The serialized contract between `CogVideoXClient` and the isolated worker.

AnimeGen-I2V generates video from an input image + text prompt (I2V mode).
Default: 81 frames @ 16fps = 5s, 480×832, 4 inference steps (Lightning LoRA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# AnimeGen-I2V native output: 81 frames @ 16 fps = 5s at 480×832.
DEFAULT_NUM_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_NUM_INFERENCE_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 42


class CogVideoXSpec(BaseModel):
    """One inference request for the worker, fully filesystem-addressed."""

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
        default="3d, cg, photo, stop, wait",
        description="Negative prompt for AnimeGen-I2V",
    )
    height: int = Field(gt=0, default=480, description="Output height (divisible by 16)")
    width: int = Field(gt=0, default=832, description="Output width (divisible by 16)")

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> CogVideoXSpec:
        return cls.model_validate_json(Path(path).read_text())

    def as_manifest(self) -> dict[str, Any]:
        """A JSON-safe record for pipeline manifests / experiment reports."""
        data = self.model_dump()
        data["image_path"] = Path(self.image_path).name
        data["checkpoint_path"] = Path(self.checkpoint_path).name
        return data
