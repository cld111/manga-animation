"""Phase 14 real GPU validation of the stage-level model lifecycle.

Runs the production page entry point (`run_page_panels`) on a real multi-panel manga page with
the REAL models (Qwen2.5-VL-7B-Instruct, Grounding DINO, SAM 2.1, LaMa), sampling GPU memory
every few seconds for the whole run so the stage-level residency claim ("each model loaded once
per stage, released deterministically, no cross-panel accumulation") is evidenced by a memory
timeline rather than asserted.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md's standing policy, ADR 0003).

Usage (models downloaded to local dirs on the worker):

    python scripts/run_phase14_gpu_validation.py \
        --page examples/realworld/villainess_ending_scuffle.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase14_gpu_validation_<ts>.json

Writes the manifest, per-panel statuses, the sampled memory timeline, ModelStage release
log lines, and measured wall-clock/peak figures as git-ignored experiment JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.harness import environment_metadata
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.reconstruction import LamaClient
from manga_animation.segmentation import Sam21Client

_MEMORY_SAMPLE_INTERVAL_S = 5.0


def _mem_sample_loop(stop: threading.Event, timeline: list[dict[str, object]]) -> None:
    import torch

    while not stop.wait(_MEMORY_SAMPLE_INTERVAL_S):
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 -- a failed sync must not kill the sampler
            break
        sample: dict[str, object] = {
            "elapsed_s": round(time.perf_counter() - timeline[0]["started_s"], 3),
            "gpus": [],
        }
        gpus: list[dict[str, float | int]] = []
        for i in range(torch.cuda.device_count()):
            gpus.append(
                {
                    "device": i,
                    "allocated_mb": round(torch.cuda.memory_allocated(i) / 2**20, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved(i) / 2**20, 1),
                }
            )
        sample["gpus"] = gpus
        timeline.append(sample)


def _capture_stage_release_logs() -> list[str]:
    import io

    import manga_animation.core.logging as core_logging

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    core_logging.get_logger("manga_animation.pipeline").addHandler(handler)
    return buffer, handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--qwen", type=str, required=True, help="local Qwen2.5-VL checkpoint")
    parser.add_argument(
        "--dino", type=str, required=True, help="local Grounding DINO checkpoint"
    )
    parser.add_argument("--sam", type=str, required=True, help="local SAM 2.1 checkpoint")
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    setup_logging(debug=False)
    config = load_config(args.env)
    device = config.resolve_device()

    vlm = Qwen25VLClient(source=args.qwen, dtype=config.dtype)
    dino = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    sam = Sam21Client(source=args.sam, device=device, dtype="float32")
    lama = LamaClient(device=device)

    import torch

    gpus = [
        {
            "device": i,
            "name": torch.cuda.get_device_name(i),
            "total_mb": round(torch.cuda.get_device_properties(i).total_memory / 2**20),
        }
        for i in range(torch.cuda.device_count())
    ]

    timeline: list[dict[str, object]] = [{"started_s": time.perf_counter(), "gpus": []}]
    stop = threading.Event()
    sampler = threading.Thread(
        target=_mem_sample_loop, args=(stop, timeline), daemon=True
    )
    sampler.start()

    buffer, handler = _capture_stage_release_logs()
    started_at = time.perf_counter()
    try:
        result = run_page_panels(
            args.page,
            config,
            vlm_client=vlm,
            grounding_client=dino,
            segmentation_client=sam,
            reconstruction_client=lama,
            out_dir=Path("outputs/videos/phase14_evidence"),
        )
        statuses = [panel.as_manifest_dict() for panel in result.panels]
        total_s = time.perf_counter() - started_at
    finally:
        stop.set()
        sampler.join(timeout=10)
        handler.flush()
        release_logs = buffer.getvalue().splitlines()

    peak_mb = 0.0
    for sample in timeline:
        for gpu in sample["gpus"]:
            peak_mb = max(peak_mb, float(gpu["allocated_mb"]))

    out_path = args.out or Path(
        f"outputs/experiments/phase14_gpu_validation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "environment": environment_metadata(device),
        "config_env": args.env,
        "model_variants": config.model_variants,
        "gpus": gpus,
        "page": str(args.page),
        "total_elapsed_s": round(total_s, 2),
        "peak_allocated_mb": peak_mb,
        "panels": statuses,
        "manifest_performance": json.loads(result.manifest_path.read_text())["performance"],
        "stage_release_logs": release_logs,
        "memory_timeline": timeline,
    }
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    print(f"total elapsed: {round(total_s, 1)}s peak allocated: {peak_mb:.0f}MB", flush=True)
    for panel in result.panels:
        print(
            f"[{panel.panel_id}] status={panel.status} runtime_s={panel.metrics.get('runtime_s')}"
            + (
                f" failing_stage={panel.failure_stage}"
                if panel.status in ("REJECTED", "ERROR")
                else ""
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
