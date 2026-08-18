"""Isolated AnimateAnything inference entrypoint (runs OUTSIDE the main pipeline process).

This script is launched by `AnimateAnythingClient` with the worker's DEDICATED interpreter
(the environment with AnimateAnything's pinned diffusers==0.24.0 / transformers==4.36.2 /
torch==2.0.0 stack). It reads one `AnimateAnythingSpec` JSON, loads the checkpoint, generates
`num_frames` frames from (image, motion mask, prompt), and writes `frame_%04d.png` files.

It deliberately does NOT import `manga_animation` (the isolated env does not install the
project's `ml` extra): the vendored model code under `vendored/` is imported via sys.path, and
the inference recipe is ported 1:1 from the upstream `train.py::eval`/`batch_eval` so the
generated frames match the reference implementation. `--selfcheck` verifies the environment
and the checkpoint without generating anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VENDORED_DIR = Path(__file__).resolve().parent / "vendored"
sys.path.insert(0, str(_VENDORED_DIR))


def _load_primary_models(pretrained_model_path: str, motion_strength: bool = False):
    """Mirror upstream `train.py::load_primary_models` (scheduler, tokenizer, text_encoder,
    vae, unet) using the vendored custom UNet3DConditionModel."""
    from diffusers import DDPMScheduler
    from diffusers.models import AutoencoderKL
    from models.unet_3d_condition_mask import UNet3DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    del noise_scheduler  # replaced by the DPMSolverMultistepScheduler in _build_pipeline
    return tokenizer, text_encoder, vae, unet


def _build_pipeline(pretrained_model_path: str, unet, text_encoder, vae):
    """Mirror upstream `batch_eval`: LatentToVideoPipeline with a DPMSolverMultistepScheduler."""
    import torch
    from diffusers import DPMSolverMultistepScheduler
    from models.pipeline import LatentToVideoPipeline

    pipeline = LatentToVideoPipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = diffusion_scheduler
    return pipeline


def _set_scheduler_timesteps(pipeline, num_inference_steps: int, device) -> None:
    """Initialize the scheduler's `timesteps`/`sigmas` before any add_noise call (mirrors
    upstream `batch_eval`: `diffusion_scheduler.set_timesteps(num_inference_steps,
    device=device)` runs BEFORE `eval`, and `DDPM_forward_timesteps` reads
    `scheduler.timesteps` / `add_noise` needs `sigmas`)."""
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)


def _generate(
    pipeline,
    *,
    image_path: str,
    mask_path: str,
    prompt: str,
    output_dir: str,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float,
    motion_strength: float,
    seed: int,
) -> list:
    """Port of upstream `train.py::eval` for one spec. Returns the frame list."""
    import math as _math

    import numpy as np
    import torch
    import torchvision.transforms as T
    from diffusers.image_processor import VaeImageProcessor
    from einops import rearrange
    from PIL import Image
    from utils.common import DDPM_forward_timesteps, tensor_to_vae_latent

    vae = pipeline.vae
    device = vae.device
    dtype = vae.dtype

    pimg = Image.open(str(Path(image_path).resolve())).convert("RGB")
    width, height = pimg.size
    # Aspect-preserving scale to roughly the validation area (512x512), rounded to /8 -- the
    # upstream eval computes exactly this.
    scale = _math.sqrt(width * height / (512.0 * 512.0))
    out_height = round(height / scale / 8) * 8
    out_width = round(width / scale / 8) * 8
    out_height = max(out_height, 8)
    out_width = max(out_width, 8)

    vae_processor = VaeImageProcessor()
    input_image = vae_processor.preprocess(pimg, out_height, out_width)
    input_image = input_image.unsqueeze(0).to(dtype).to(device)
    input_image_latents = tensor_to_vae_latent(input_image, vae)

    np_mask: np.ndarray | None
    if mask_path:
        mask_file = Path(mask_path).resolve()
        if mask_file.exists():
            np_mask = np.array(Image.open(str(mask_file)).resize((out_width, out_height)))
            np_mask[np_mask != 0] = 255
        else:
            np_mask = None
    else:
        np_mask = None
    if np_mask is None or np_mask.sum() == 0:
        # No mask (or an all-black mask): the WHOLE crop is the motion area -- upstream
        # app.py's full-motion fallback. Since the 2026 architecture change the engine
        # consumes per-object DINO bbox crops with no SAM mask, this is the default path.
        np_mask = np.full((out_height, out_width), 255, dtype=np.uint8)

    torch.manual_seed(seed)
    initial_latents, timesteps = DDPM_forward_timesteps(
        input_image_latents, num_inference_steps, num_frames, pipeline.scheduler
    )
    mask = T.ToTensor()(np_mask).to(dtype).to(device)
    b, c, f, h, w = initial_latents.shape
    mask = T.Resize([h, w], antialias=False)(mask)
    mask = rearrange(mask, "b h w -> b 1 1 h w")

    with torch.no_grad():
        video_frames, _video_latents = pipeline(
            prompt=prompt,
            latents=initial_latents,
            width=out_width,
            height=out_height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            condition_latent=input_image_latents,
            mask=mask,
            motion=[motion_strength],
            return_dict=False,
            timesteps=timesteps,
        )

    # Normalize each frame to an RGB uint8 array (tensor2vid returns a list of arrays here).
    frames = []
    for frame in video_frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = (frame * 255).round().astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frames.append(frame)
        elif frame.ndim == 2:
            frames.append(np.repeat(frame[:, :, None], 3, axis=2))
        else:
            raise ValueError(f"unexpected frame shape {frame.shape}")
    return frames


def _write_frames(frames: list, output_dir: str) -> None:
    from PIL import Image

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"frame_{index:04d}.png")


def _selfcheck(checkpoint_path: str) -> None:
    """Verify the isolated env can import the vendored model code and the checkpoint exists."""
    import torch
    from diffusers import __version__ as diffusers_version
    from models.pipeline import LatentToVideoPipeline  # noqa: F401
    from models.unet_3d_condition_mask import UNet3DConditionModel  # noqa: F401
    from PIL import Image  # noqa: F401
    from utils.common import DDPM_forward_timesteps, tensor_to_vae_latent  # noqa: F401

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint_dir = Path(checkpoint_path)
    required = [
        checkpoint_dir / d
        for d in ("scheduler", "tokenizer", "text_encoder", "vae", "unet")
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("checkpoint missing subfolders: " + ", ".join(missing))
    print(
        json.dumps(
            {
                "ok": True,
                "torch": torch.__version__,
                "diffusers": diffusers_version,
                "checkpoint": str(checkpoint_path),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="path to an AnimateAnythingSpec JSON")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())

    if args.selfcheck:
        _selfcheck(spec.get("checkpoint_path", ""))
        return

    import torch

    torch.set_grad_enabled(False)
    device = torch.device(spec.get("device", "cuda"))

    tokenizer, text_encoder, vae, unet = _load_primary_models(spec["checkpoint_path"])
    for model in (text_encoder, unet, vae):
        model.to(device).eval().to(torch.float16)
    vae.enable_slicing()

    pipeline = _build_pipeline(spec["checkpoint_path"], unet, text_encoder, vae)
    pipeline = pipeline.to(device, dtype=torch.float16)
    _set_scheduler_timesteps(pipeline, spec["num_inference_steps"], device)

    frames = _generate(
        pipeline,
        image_path=spec["image_path"],
        mask_path=spec.get("mask_path"),
        prompt=spec["prompt"],
        output_dir=spec["output_dir"],
        num_frames=spec["num_frames"],
        fps=spec["fps"],
        num_inference_steps=spec["num_inference_steps"],
        guidance_scale=spec["guidance_scale"],
        motion_strength=spec["motion_strength"],
        seed=spec["seed"],
    )
    _write_frames(frames, spec["output_dir"])
    print(
        json.dumps(
            {
                "ok": True,
                "frames": len(frames),
                "output_dir": spec["output_dir"],
                "fps": spec["fps"],
            }
        )
    )


if __name__ == "__main__":
    main()
