"""Re-animate a SINGLE accepted object with AnimateAnything, resuming Qwen checkpoints.

Phase-2-only, single-object convenience runner: reads the persisted
`grounding.json`/`descriptions.json` checkpoints (so DINO and Qwen are never loaded),
picks one `panel_id` + `object_id`, crops the panel at that object's DINO bbox, builds the
AnimateAnything prompt from the accepted Qwen description, runs one worker, and re-renders
that object's MP4 (overwriting the previous one).

Usage:

    python scripts/run_aa_single_object.py \
        --page examples/realworld/wind_breaker_sprint.png \
        --panel panel_002 --object obj_character_0 \
        --aa-checkpoint /tmp/models/animate_anything_512_v1.02 \
        --aa-python /kaggle/working/aa-venv/bin/python \
        --out-dir outputs/experiments/videos

Requires the Qwen phase checkpoints at `--out-dir/<page_stem>/grounding.json` and
`descriptions.json` (written by run_animate_anything_e2e.py or a stop_after_description run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.animation_anything.client import AnimateAnythingClient
from manga_animation.animation_anything.prompt import build_animation_prompt
from manga_animation.core.config import load_config
from manga_animation.pipeline.persistence import (
    load_descriptions,
    load_grounding,
)
from manga_animation.rendering import render

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "manga_animation" / \
    "animation_anything" / "worker.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", required=True)
    parser.add_argument("--panel", required=True, help="e.g. panel_002")
    parser.add_argument("--object", required=True, help="e.g. obj_character_0")
    parser.add_argument("--aa-checkpoint", required=True)
    parser.add_argument("--aa-python", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--env", default="kaggle")
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": 1536})
    page = Path(args.page)
    page_dir = Path(args.out_dir) / page.stem

    candidates_by_panel, plans_by_panel, _ = load_grounding(page_dir)
    descriptions_by_panel = load_descriptions(page_dir)

    panel_id = args.panel
    object_id = args.object

    if panel_id not in candidates_by_panel or object_id not in candidates_by_panel[panel_id]:
        raise SystemExit(f"object {object_id} not found in grounding for {panel_id}")
    grounding = candidates_by_panel[panel_id][object_id][0]
    obj_plan = plans_by_panel[panel_id][object_id]
    desc = descriptions_by_panel[panel_id].get((object_id, 0))
    if desc is None or not desc.accepted:
        raise SystemExit(f"object {object_id} has no accepted description for {panel_id}")
    if desc.object_identity is None and obj_plan.semantic_label is None:
        raise SystemExit("no identity/label to build a prompt from")

    # Panel scene crop (grounding bboxes are crop-local).
    crop = np.asarray(Image.open(page_dir / "crops" / f"{panel_id}.png").convert("RGB"))
    bbox = grounding.bbox
    x0, y0 = max(0, bbox.x0), max(0, bbox.y0)
    x1, y1 = min(crop.shape[1], bbox.x1), min(crop.shape[0], bbox.y1)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"degenerate bbox for {object_id}")
    object_crop = crop[y0:y1, x0:x1]

    prompt = build_animation_prompt(obj_plan, desc)
    print(f"prompt: {prompt!r}")
    print(f"crop: {object_crop.shape[1]}x{object_crop.shape[0]}")

    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    client = AnimateAnythingClient(
        source=args.aa_checkpoint,
        python_bin=args.aa_python,
        worker_script=_WORKER_SCRIPT,
        device=device,
        num_frames=config.animation_num_frames,
        fps=config.animation_fps,
        num_inference_steps=config.animation_num_inference_steps,
        guidance_scale=config.animation_guidance_scale,
        motion_strength=config.animation_motion_strength,
        seed=config.seed,
    )
    try:
        client.load()
        frames = client.animate(
            object_crop,
            prompt,
            page_dir / "animate_anything" / panel_id / object_id,
        )
        out_path = page_dir / f"{panel_id}_{object_id}.mp4"
        render(
            frames,
            out_path,
            codec=config.output_codec,
            keep_frames=True,
            frames_dir=page_dir / "frames" / panel_id / object_id,
        )
        print(f"saved {out_path} ({frames.frame_count} frames @ {frames.fps} fps)")
    finally:
        client.unload()


if __name__ == "__main__":
    main()
