"""Phase 14 forensic GPU memory profiling of the model lifecycle (run on a GPU worker only).

This script is the evidence-gathering step for the stage-level model lifecycle work: it drives
the exact per-panel model lifecycle the pipeline performs (Qwen analysis -> DINO grounding ->
Qwen validation -> SAM segmentation -> Qwen mask_semantics -> LaMa reconstruction) using the
REAL clients, repeated for N panels, and records GPU memory at every meaningful boundary.

It deliberately drives the clients directly instead of running `run_page_panels` end to end:
the model stages are the thing whose memory lifecycle is under investigation, and isolating
them from the CPU-only CV/render tail keeps each run fast enough to iterate on.

Measurements per boundary (per GPU): `memory_allocated`, `memory_reserved`,
`max_memory_allocated` (peak since process start), a full-process `gc.get_objects()` scan for
live CUDA tensors (count + bytes), and wall-clock seconds since the run started. The live-
tensor scan is the forensic complement to the allocator stats: it distinguishes "memory held by
the CUDA caching allocator" from "memory held by tensors still referenced from Python", which
is what determines whether `torch.cuda.empty_cache()` alone can ever release it.

Usage (on the Kaggle/Jupyter worker, with the repo checked out and models downloaded):

    python scripts/run_phase14_gpu_mem_profiling.py \
        --page examples/realworld/villainess_ending_scuffle.png \
        --panels 4 \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase14_profiling_<ts>.json

Writes the boundary snapshot list as git-ignored experiment JSON under `outputs/experiments/`.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis import Qwen25VLClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.types import BBoxPx
from manga_animation.reconstruction import LamaClient
from manga_animation.segmentation import Sam21Client

# Small, real crop the profiling runs use as model input (the source page is resized to this
# long edge so a multi-panel profiling run stays fast; the memory lifecycle is what is being
# measured, not analysis quality).
_VLM_LONG_EDGE = 768


def _snapshot(label: str, started_at: float) -> dict[str, object]:
    import torch

    torch.cuda.synchronize()
    rec: dict[str, object] = {
        "label": label,
        "elapsed_s": round(time.perf_counter() - started_at, 3),
        "gpus": [],
        "live_cuda_tensors": 0,
        "live_cuda_mb": 0.0,
    }
    gpus: list[dict[str, float | int]] = []
    for i in range(torch.cuda.device_count()):
        gpus.append(
            {
                "device": i,
                "allocated_mb": round(torch.cuda.memory_allocated(i) / 2**20, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(i) / 2**20, 1),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated(i) / 2**20, 1),
            }
        )
    rec["gpus"] = gpus

    count = 0
    bytes_live = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                bytes_live += obj.numel() * obj.element_size()
        except Exception:  # noqa: BLE001 -- a single weird object must not abort the scan
            continue
    rec["live_cuda_tensors"] = count
    rec["live_cuda_mb"] = round(bytes_live / 2**20, 1)

    print(
        f"[mem] {label:<46} "
        + "  ".join(
            f"g{i}:alloc={g['allocated_mb']:>8}MB res={g['reserved_mb']:>8}MB "
            f"peak={g['peak_allocated_mb']:>8}MB"
            for i, g in enumerate(gpus)
        )
        + f"  live_tensors={count:<5} live_mb={bytes_live / 2**20:.1f}",
        flush=True,
    )
    return rec


def _input_image(page_path: Path) -> Image.Image:
    image = np.asarray(Image.open(page_path).convert("RGB"))
    h, w = image.shape[:2]
    scale = _VLM_LONG_EDGE / max(h, w)
    if scale < 1.0:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        image = np.asarray(
            Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
        )
    return Image.fromarray(image)


def _np_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--qwen", type=str, required=True, help="local Qwen2.5-VL checkpoint")
    parser.add_argument(
        "--dino", type=str, required=True, help="local Grounding DINO checkpoint"
    )
    parser.add_argument("--sam", type=str, required=True, help="local SAM 2.1 checkpoint")
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--current-unload", action="store_true",
        help="use the current (unfixed) unload path: set-None + empty_cache, no gc.collect()",
    )
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()

    image = _input_image(args.page)
    np_image = _np_image(image)
    bbox = BBoxPx(
        x0=int(np_image.shape[1] * 0.2),
        y0=int(np_image.shape[0] * 0.2),
        x1=int(np_image.shape[1] * 0.8),
        y1=int(np_image.shape[0] * 0.8),
    )

    vlm = Qwen25VLClient(source=args.qwen, dtype=config.dtype)
    dino = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=device, dtype="float32")
    lama = LamaClient(device=device)

    def unload(client: object, label: str, started: float) -> None:
        """The only thing this script varies: whether `gc.collect()` precedes the caching
        allocator release. `client.unload()` itself always drops the Python references first --
        exactly as every client's `unload()` already does today (set `self._model = None`) and
        exactly like the stage-level lifecycle will. `--current-unload` reproduces the real
        current pipeline's unload path (no `gc.collect()`); the default adds `gc.collect()`
        to measure what a deterministic stage release buys."""
        import torch

        client.unload()
        if not args.current_unload:
            gc.collect()
            torch.cuda.empty_cache()
        snapshots.append(_snapshot(f"{label} after {type(client).__name__} unload", started))

    started_at = time.perf_counter()
    snapshots: list[dict[str, object]] = []
    snapshots.append(_snapshot("initial", started_at))
    t_loads: dict[str, float] = {}
    t_infer: dict[str, float] = {}

    import torch

    for panel in range(1, args.panels + 1):
        prefix = f"panel{panel}"
        # 1. Qwen analysis stage
        t0 = time.perf_counter()
        vlm.generate(image, "Analyze this manga panel: list objects and their motion cues as JSON.")
        t_infer[f"{prefix} qwen_analysis"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after Qwen analysis infer", started_at))
        unload(vlm, prefix, started_at)

        # 2. DINO grounding stage
        t0 = time.perf_counter()
        dino.load()
        t_loads[f"{prefix} dino_load"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after DINO load", started_at))
        t0 = time.perf_counter()
        dino.detect(np_image, "a character")
        t_infer[f"{prefix} dino_detect"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after DINO detect", started_at))
        unload(dino, prefix, started_at)

        # 3. Qwen validation stage
        t0 = time.perf_counter()
        vlm.generate(image, "Does the image above show the target object? Reply JSON.")
        t_infer[f"{prefix} qwen_validation"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after Qwen validation infer", started_at))
        unload(vlm, prefix, started_at)

        # 4. SAM segmentation stage
        t0 = time.perf_counter()
        sam.load()
        t_loads[f"{prefix} sam_load"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after SAM load", started_at))
        t0 = time.perf_counter()
        sam.segment(np_image, bbox)
        t_infer[f"{prefix} sam_segment"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after SAM segment", started_at))
        unload(sam, prefix, started_at)

        # 5. Qwen mask_semantics stage
        t0 = time.perf_counter()
        vlm.generate(image, "Does the bright region show only the target object? Reply JSON.")
        t_infer[f"{prefix} qwen_mask_semantics"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after Qwen mask_semantics infer", started_at))
        unload(vlm, prefix, started_at)

        # 6. LaMa reconstruction stage
        t0 = time.perf_counter()
        lama.load()
        t_loads[f"{prefix} lama_load"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after LaMa load", started_at))
        t0 = time.perf_counter()
        lama.inpaint(image, Image.fromarray(np.ones_like(np_image[..., 0]) * 255))
        t_infer[f"{prefix} lama_inpaint"] = round(time.perf_counter() - t0, 2)
        snapshots.append(_snapshot(f"{prefix} after LaMa inpaint", started_at))
        unload(lama, prefix, started_at)

    snapshots.append(_snapshot("end", started_at))

    out_path = args.out or Path(f"outputs/experiments/phase14_profiling_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": "current_unload" if args.current_unload else "fixed_unload",
        "panels": args.panels,
        "page": str(args.page),
        "gpus": [
            {
                "device": i,
                "name": torch.cuda.get_device_name(i),
                "total_mb": round(torch.cuda.get_device_properties(i).total_memory / 2**20),
            }
            for i in range(torch.cuda.device_count())
        ],
        "load_seconds": t_loads,
        "infer_seconds": t_infer,
        "snapshots": snapshots,
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {round(time.perf_counter() - started_at, 1)}s", flush=True)


if __name__ == "__main__":
    main()
