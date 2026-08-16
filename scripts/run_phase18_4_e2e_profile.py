"""Phase 18.4 end-to-end GPU validation WITH PROFILING: DINO -> Qwen -> SAM pipeline.

Runs the production page entry point (`run_page_panels`) on REAL manga pages with the Phase
18.4 ordering (object description BEFORE segmentation: SAM segments only accepted bboxes)
and collects a GPU profile alongside: an nvidia-smi sampler thread records per-GPU
utilization and memory every `--sampler-interval-s` seconds for the whole run.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

Usage (models downloaded to local dirs on the worker, pages fetched):

    python scripts/run_phase18_4_e2e_profile.py \
        --pages examples/realworld/villainess_ending_scuffle.png \
                examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase18_4_e2e_profile_<ts>.json

Writes one git-ignored experiment JSON per invocation with per-panel statuses, the VLM call
count (one per panel, all bboxes in one prompt), per-stage wall-clock timings, and the GPU
sampler trace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient, VLMClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.segmentation import Sam21Client


class CountingVLMClient:
    """Wraps the real Qwen client and counts every generate() call, so the report can prove
    the Phase 18.3/18.4 contract: exactly one VLM call per panel (all its bboxes in one
    prompt)."""

    def __init__(self, inner: VLMClient):
        self._inner = inner
        self.call_count = 0
        self.boxes_per_call: list[int] = []

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        self.boxes_per_call.append(sum(1 for line in prompt.splitlines() if line.startswith("[")))
        return self._inner.generate(image, prompt)

    def unload(self) -> None:
        self._inner.unload()


class GpuSampler:
    """Background nvidia-smi sampler: appends one dict per tick into `samples`."""

    def __init__(self, interval_s: float = 3.0):
        self._interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            tick = {"t_s": round(time.perf_counter() - self._started_at, 2)}
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                gpus = []
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 4:
                        gpus.append(
                            {
                                "util_pct": float(parts[0]),
                                "mem_used_mib": float(parts[1]),
                                "mem_total_mib": float(parts[2]),
                                "power_w": float(parts[3]),
                            }
                        )
                tick["gpus"] = gpus
            except Exception as exc:  # noqa: BLE001 -- a missed sample must not kill the run
                tick["error"] = f"{type(exc).__name__}: {exc}"
            self.samples.append(tick)
            self._stop.wait(self._interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--sampler-interval-s", type=float, default=3.0)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate semantic labels to ground (default: the pipeline default list)",
    )
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": args.resolution})
    config.model_variants.update(
        {
            "vlm": "qwen2.5-vl-7b-instruct",
            "grounding": "grounding-dino-swin-l",
            "segmentation": "sam2.1-hiera-base",
        }
    )
    assert config.enable_object_description_validation, (
        "the object-description stage must be enabled"
    )

    vlm_client = CountingVLMClient(Qwen25VLClient(source=args.qwen, dtype="float16"))
    setup_logging("INFO")
    device = config.resolve_device()
    grounding_client = GroundingDinoClient(source=args.dino, device=device, dtype="float32")
    segmentation_client = Sam21Client(source=args.sam, device=device, dtype="float32")
    from manga_animation.reconstruction import LamaClient

    reconstruction_client = LamaClient(device=device, model_id="lama-large")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler(interval_s=args.sampler_interval_s)
    sampler.start()
    started = time.perf_counter()

    stage_timings: list[dict] = []
    report: dict = {
        "phase": "18.4-e2e-profile",
        "ordering": "grounding(DINO) -> object_description(Qwen) -> "
        "segmentation(SAM, accepted only)",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
    }
    try:
        for page in args.pages:
            page_started = time.perf_counter()
            page_result = run_page_panels(
                Path(page),
                config,
                vlm_client=vlm_client,
                grounding_client=grounding_client,
                segmentation_client=segmentation_client,
                reconstruction_client=reconstruction_client,
                out_dir=out_dir / "videos",
                labels=args.labels,
            )
            page_entry = {
                "page": page,
                "page_runtime_s": round(time.perf_counter() - page_started, 2),
                "panels": [
                    {
                        "panel_id": p.panel_id,
                        "status": p.status,
                        "failure_stage": p.failure_stage,
                        "failure_reason": p.failure_reason,
                        "output_video": str(p.output_video) if p.output_video else None,
                        "metrics": p.metrics,
                    }
                    for p in page_result.panels
                ],
            }
            report["pages"].append(page_entry)
            print(json.dumps(page_entry, indent=1))
            print(f"[profiler] page done in {page_entry['page_runtime_s']}s", flush=True)
    finally:
        sampler.stop()
        for client in (vlm_client, grounding_client, segmentation_client, reconstruction_client):
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown at the very end
                    pass

    report["elapsed_s"] = round(time.perf_counter() - started, 1)
    report["vlm_calls"] = vlm_client.call_count
    report["boxes_per_vlm_call"] = vlm_client.boxes_per_call
    report["stage_timings"] = stage_timings
    report["gpu_sampler_interval_s"] = args.sampler_interval_s
    report["gpu_samples"] = sampler.samples
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"VLM calls: {vlm_client.call_count}, boxes per call: {vlm_client.boxes_per_call}")
    print(f"GPU samples: {len(sampler.samples)}")


if __name__ == "__main__":
    main()
