"""The serialized contract between `AnimateAnythingClient` and the isolated `worker.py`.

The client and the worker run in DIFFERENT environments (the worker needs AnimateAnything's
pinned diffusers==0.24.0/transformers==4.36.2/torch==2.0.0 stack; the pipeline needs
transformers>=5.0). They therefore communicate through the filesystem: the client writes one
`AnimateAnythingSpec` JSON plus the input image, invokes the worker on the isolated
interpreter, and reads the generated frame PNGs back. This module is the shared,
pydantic-validated shape of that hand-off -- nothing model-specific, so it is unit-testable
without torch/diffusers.

Since the 2026 architecture change the engine consumes a per-object DINO bbox crop and the
Qwen description prompt directly -- no SAM segmentation and no motion mask. `mask_path` is
therefore optional and the worker defaults to full-image motion when it is absent (mirroring
the upstream `app.py` all-black-mask fallback: a mask that does not restrict the motion area
means the whole crop is allowed to move).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Native output of AnimateAnything 512 v1.02 (see its config.yaml): 16 frames @ 8 fps = 2s.
DEFAULT_NUM_FRAMES = 16
DEFAULT_FPS = 8
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 9.0
# Motion strength is a meaningful value (upstream's README uses 5) because without a motion
# mask the WHOLE crop is the motion area and a small scalar leaves the object near-static --
# the real 2026 run measured visibly static objects at 1.0. The project's "minimum visually
# justified motion" principle still applies; 5.0 is the upstream default, not an extreme.
DEFAULT_MOTION_STRENGTH = 5.0
DEFAULT_SEED = 42


class AnimateAnythingSpec(BaseModel):
    """One inference request for the worker, fully filesystem-addressed.

    The worker needs no knowledge of this project's pipeline: it receives an image, a prompt
    and hyper-parameters, and produces frame PNGs in `output_dir`.
    """

    image_path: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    checkpoint_path: str = Field(min_length=1)
    device: str = Field(default="cuda", description="CUDA device the worker loads onto")
    mask_path: str | None = Field(
        default=None,
        description="Optional uint8 0/255 motion mask; None means the whole crop may move.",
    )
    num_frames: int = Field(gt=0, default=DEFAULT_NUM_FRAMES)
    fps: int = Field(gt=0, default=DEFAULT_FPS)
    num_inference_steps: int = Field(gt=0, default=DEFAULT_NUM_INFERENCE_STEPS)
    guidance_scale: float = Field(gt=0.0, default=DEFAULT_GUIDANCE_SCALE)
    motion_strength: float = Field(gt=0.0, default=DEFAULT_MOTION_STRENGTH)
    seed: int = Field(default=DEFAULT_SEED)

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> AnimateAnythingSpec:
        return cls.model_validate_json(Path(path).read_text())

    def as_manifest(self) -> dict[str, Any]:
        """A JSON-safe record for pipeline manifests / experiment reports (no secrets)."""
        data = self.model_dump()
        data["image_path"] = Path(self.image_path).name
        if data.get("mask_path"):
            data["mask_path"] = Path(data["mask_path"]).name
        data["checkpoint_path"] = Path(self.checkpoint_path).name
        return data
