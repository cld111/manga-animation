"""Isolated AnimeGen-I2V inference entrypoint (runs OUTSIDE the main pipeline process).

This script is launched by `CogVideoXClient` (renamed from AnimateAnythingClient) with the
worker's DEDICATED interpreter. It reads one `CogVideoXSpec` JSON, loads the AnimeGen-I2V
checkpoint, generates `num_frames` frames from (image, prompt), and writes `frame_%04d.png`
files.

AnimeGen-I2V is a Wan 2.2 I2V A14B fine-tuned for anime-style video generation.
It uses two transformers (high_noise + low_noise) with FP8 layerwise casting and CPU offload
to fit on consumer GPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _selfcheck(checkpoint_path: str) -> None:
    """Verify the isolated env can import diffusers and the checkpoint exists."""
    import torch
    from diffusers import __version__ as diffusers_version

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    ckpt_dir = Path(checkpoint_path)
    required = ["transformer", "transformer_2", "model_index.json"]
    missing = [d for d in required if not (ckpt_dir / d).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint missing subfolders: {', '.join(missing)} in {checkpoint_path}"
        )
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


def _generate(
    *,
    image_path: str,
    prompt: str,
    output_dir: str,
    checkpoint_path: str,
    device: str,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str,
    height: int,
    width: int,
) -> list:
    """Generate video frames using AnimeGen-I2V (Wan 2.2 I2V A14B anime-tuned).

    Returns the list of RGB uint8 numpy arrays.
    """
    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        WanImageToVideoPipeline,
        WanTransformer3DModel,
    )
    from diffusers.utils import load_image
    from PIL import Image

    torch.set_grad_enabled(False)

    # Load the input image
    input_image = load_image(image_path)
    orig_w, orig_h = input_image.size

    # Resize to target dimensions (must be divisible by VAE spatial compression * patch_size)
    # For Wan2.2: vae_scale_factor_spatial=8, patch_size=2 → mod=16
    mod_value = 16
    target_h = max(mod_value, (height // mod_value) * mod_value)
    target_w = max(mod_value, (width // mod_value) * mod_value)
    input_image_resized = input_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    # Load transformers (high_noise + low_noise for MoE)
    print("loading high_noise transformer...", flush=True)
    transformer_high = WanTransformer3DModel.from_pretrained(
        checkpoint_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    print("loading low_noise transformer...", flush=True)
    transformer_low = WanTransformer3DModel.from_pretrained(
        checkpoint_path,
        subfolder="transformer_2",
        torch_dtype=dtype,
    )

    # Load VAE from base Wan2.2 model
    # The AnimeGen-I2V model card says to use Wan-AI/Wan2.2-I2V-A14B-Diffusers for VAE
    # But we might have it locally or need to download it
    import os
    local_vae_path = os.path.join(os.path.dirname(checkpoint_path), "Wan2.2-I2V-A14B-Diffusers")
    if os.path.exists(local_vae_path):
        vae_source = local_vae_path
    else:
        vae_source = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    print(f"loading VAE from {vae_source}...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        vae_source,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    # Build pipeline
    print("building pipeline...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        vae_source,
        transformer=transformer_high,
        transformer_2=transformer_low,
        scheduler=scheduler,
        vae=vae,
        torch_dtype=dtype,
    )

    # Load Lightning LoRA (4-step inference)
    lora_high = os.path.join(checkpoint_path, "high_noise.safetensors")
    lora_low = os.path.join(checkpoint_path, "low_noise.safetensors")
    if os.path.exists(lora_high) and os.path.exists(lora_low):
        print("loading Lightning LoRA...", flush=True)
        pipe.load_lora_weights(
            checkpoint_path,
            weight_name="high_noise.safetensors",
            adapter_name="high",
        )
        pipe.load_lora_weights(
            checkpoint_path,
            weight_name="low_noise.safetensors",
            adapter_name="low",
            load_into_transformer_2=True,
        )
        pipe.set_adapters(["high", "low"], adapter_weights=[1.0, 1.0])
        use_lora = True
    else:
        print("no LoRA found, using standard inference", flush=True)
        use_lora = False

    # Memory optimizations
    try:
        transformer_high.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype
        )
        transformer_low.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype
        )
        print("FP8 layerwise casting enabled", flush=True)
    except Exception as e:
        print(f"FP8 casting failed: {e}", flush=True)

    pipe.enable_model_cpu_offload()

    # Set up generator for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Build prompt (prepend anime style)
    full_prompt = "Japanese anime style, " + prompt
    neg = negative_prompt or "3d, cg, photo, stop, wait"

    # Determine inference steps (4 for Lightning LoRA, 30 otherwise)
    steps = 4 if use_lora else min(num_inference_steps, 30)

    print(
        f"generating {num_frames} frames at {target_w}x{target_h}, {steps} steps...",
        flush=True,
    )

    # Run inference
    with torch.no_grad():
        output = pipe(
            image=input_image_resized,
            prompt=full_prompt,
            negative_prompt=neg,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        )

    # Extract frames as numpy arrays
    video_frames = output.frames[0] if hasattr(output, "frames") else output
    frames = []
    for frame in video_frames:
        if isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))
        elif hasattr(frame, "dtype") and frame.dtype != np.uint8:
            frame = (frame * 255).round().astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frames.append(frame)
        elif frame.ndim == 2:
            frames.append(np.repeat(frame[:, :, None], 3, axis=2))
        else:
            raise ValueError(f"unexpected frame shape {str(frame.shape)}")

    # Clean up GPU memory
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return frames


def _write_frames(frames: list, output_dir: str) -> None:
    from PIL import Image

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(out / f"frame_{index:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="path to a CogVideoXSpec JSON")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())

    if args.selfcheck:
        _selfcheck(spec.get("checkpoint_path", ""))
        return

    frames = _generate(
        image_path=spec["image_path"],
        prompt=spec["prompt"],
        output_dir=spec["output_dir"],
        checkpoint_path=spec["checkpoint_path"],
        device=spec.get("device", "cuda"),
        num_frames=spec["num_frames"],
        fps=spec["fps"],
        num_inference_steps=spec["num_inference_steps"],
        guidance_scale=spec["guidance_scale"],
        seed=spec["seed"],
        negative_prompt=spec.get("negative_prompt", "3d, cg, photo, stop, wait"),
        height=spec.get("height", 480),
        width=spec.get("width", 832),
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
