"""Isolated Wan2.2-TI2V-5B inference entrypoint (runs OUTSIDE the main pipeline process).

This script is launched by `Wan2Client` with the worker's DEDICATED interpreter (the
environment with diffusers main branch, torch>=2.4.0, etc.). It reads one `Wan2Spec` JSON,
loads the Wan2.2-TI2V-5B checkpoint, generates `num_frames` frames from (image, prompt),
and writes `frame_%04d.png` files.

It deliberately does NOT import `manga_animation` (the isolated env may not install the
project's `ml` extra): the inference code uses only diffusers and standard libraries.

On 2x T4 GPUs, the model can be sharded with FSDP + DeepSpeed Ulysses via
`--nproc_per_node=2 --dit_fsdp --t5_fsdp --ulysses_size 2`.
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
    # Verify key subfolders exist
    ckpt_dir = Path(checkpoint_path)
    required = ["transformer", "vae", "tokenizer", "text_encoder"]
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
    """Generate video frames using Wan2.2-TI2V-5B WanPipeline (I2V mode).

    Returns the list of RGB uint8 numpy arrays.
    """
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
    from diffusers.utils import load_image
    from PIL import Image

    torch.set_grad_enabled(False)

    # Load the input image
    input_image = load_image(image_path)
    orig_w, orig_h = input_image.size

    # Resize to target dimensions (must be divisible by 16)
    input_image_resized = input_image.resize((width, height), Image.Resampling.LANCZOS)

    # Load VAE (float32 for quality), transformer and text encoder (dtype for memory)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_path, subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        checkpoint_path, subfolder="transformer", torch_dtype=dtype
    )

    pipe = WanPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=None,  # loaded from checkpoint config
        tokenizer=None,  # loaded from checkpoint
        text_encoder=None,  # loaded from checkpoint
    )
    pipe = WanPipeline.from_pretrained(
        checkpoint_path,
        vae=vae,
        torch_dtype=dtype,
    )

    # Move to device with memory optimizations
    pipe.to(device)

    # Enable memory-efficient attention if available
    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:  # noqa: BLE001 -- best-effort optimization
            pass

    # Set up generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)

    # Run inference in I2V mode
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image_resized,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # Extract frames as numpy arrays
    video_frames = output.frames[0] if hasattr(output, "frames") else output
    frames = []
    for frame in video_frames:
        if isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))
        elif frame.dtype != np.uint8:
            frame = (frame * 255).round().astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frames.append(frame)
        elif frame.ndim == 2:
            frames.append(np.repeat(frame[:, :, None], 3, axis=2))
        else:
            raise ValueError(f"unexpected frame shape {frame.shape}")

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
    parser.add_argument("--spec", required=True, help="path to a Wan2Spec JSON")
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
        negative_prompt=spec.get("negative_prompt", "static, blurry, low quality"),
        height=spec.get("height", 704),
        width=spec.get("width", 1280),
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
