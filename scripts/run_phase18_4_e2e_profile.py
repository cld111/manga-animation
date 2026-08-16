"""Phase 18.4 end-to-end GPU validation WITH PROFILING: DINO -> Qwen -> SAM pipeline.

Runs the production batch entry point (`run_pages`) on REAL manga pages with the Phase
18.4 ordering (object description BEFORE segmentation: SAM segments only accepted bboxes)
and phase 18.4 batch residency (each model loads ONCE and processes ALL pages before being
released) and collects a GPU profile alongside: an nvidia-smi sampler thread records
per-GPU utilization and memory every `--sampler-interval-s` seconds for the whole run.

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
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen25VLClient, VLMClient
from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.panels import run_pages
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
    """Background nvidia-smi + /proc/stat sampler: appends one dict per tick into `samples`.

    Each sample carries per-GPU utilization/memory/power and host CPU utilization (computed
    from `/proc/stat` deltas, the standard Linux kernel counters -- no psutil dependency).
    """

    def __init__(self, interval_s: float = 3.0):
        self._interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._prev_cpu: tuple[int, int] | None = None

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_s + 2.0)

    @staticmethod
    def _cpu_usage(prev: tuple[int, int] | None) -> tuple[float | None, tuple[int, int]]:
        """Read /proc/stat aggregate (jiffies), return (busy_pct_since_prev, new_counts)."""
        try:
            with open("/proc/stat", encoding="utf-8") as fh:
                parts = fh.readline().split()
            if not parts or parts[0] != "cpu":
                return None, (0, 0)
            vals = [int(v) for v in parts[1:9]]
            total = sum(vals)
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            if prev is None or total <= prev[0]:
                return None, (total, idle)
            dt_total = total - prev[0]
            dt_idle = idle - prev[1]
            pct = 100.0 * (1.0 - dt_idle / dt_total) if dt_total > 0 else 0.0
            return round(pct, 1), (total, idle)
        except (OSError, ValueError, IndexError):
            return None, (0, 0)

    @staticmethod
    def _ram_usage() -> dict | None:
        """Host RAM from /proc/meminfo (MemTotal/MemAvailable), in MiB."""
        try:
            fields: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    if key in ("MemTotal", "MemAvailable", "MemFree"):
                        fields[key] = int(rest.strip().split()[0]) // 1024
            if not fields:
                return None
            available = fields.get("MemAvailable", fields.get("MemFree", 0))
            return {
                "ram_used_mib": fields.get("MemTotal", 0) - available,
                "ram_total_mib": fields.get("MemTotal", 0),
            }
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _disk_usage() -> dict | None:
        """Working-dir disk usage from `shutil.disk_usage`, in MiB."""
        try:
            total, used, free = shutil.disk_usage("/kaggle")
            return {
                "disk_used_mib": used // 1048576,
                "disk_total_mib": total // 1048576,
                "disk_free_mib": free // 1048576,
            }
        except OSError:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            tick: dict[str, object] = {
                "t_s": round(time.perf_counter() - self._started_at, 2)
            }
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
                tick["gpu_error"] = f"{type(exc).__name__}: {exc}"
            cpu_pct, self._prev_cpu = self._cpu_usage(self._prev_cpu)
            if cpu_pct is not None:
                tick["cpu_util_pct"] = cpu_pct
            ram = self._ram_usage()
            if ram is not None:
                tick.update(ram)
            disk = self._disk_usage()
            if disk is not None:
                tick.update(disk)
            self.samples.append(tick)
            self._stop.wait(self._interval_s)


class StageMarker:
    """Records the wall-clock boundaries of each model stage (via client load/unload) and
    VLM calls, so the profile trace can be annotated with per-stage GPU/CPU load."""

    def __init__(self, sampler_started_at: float):
        self._t0 = sampler_started_at
        self.stage_events: list[dict] = []
        self.vlm_calls: list[dict] = []

    def _now(self) -> float:
        return round(time.perf_counter() - self._t0, 2)

    def wrap(self, client, *, name: str, is_vlm: bool = False):
        class Wrapper:
            def __init__(self, inner, marker, stage_name: str, vlm: bool):
                self._inner = inner
                self._marker = marker
                self._stage_name = stage_name
                self._vlm = vlm

            def load(self) -> None:
                if hasattr(self._inner, "load"):
                    self._inner.load()
                self._marker.stage_events.append(
                    {"t_s": self._marker._now(), "stage": self._stage_name, "event": "load"}
                )

            def unload(self) -> None:
                self._marker.stage_events.append(
                    {"t_s": self._marker._now(), "stage": self._stage_name, "event": "unload"}
                )
                if hasattr(self._inner, "unload"):
                    self._inner.unload()

            def __getattr__(self, attr):
                return getattr(self._inner, attr)

        return Wrapper(client, self, name, is_vlm)


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

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler(interval_s=args.sampler_interval_s)
    sampler.start()
    started = time.perf_counter()

    marker = StageMarker(started)

    def _wrap(client, name: str, *, is_vlm: bool = False):
        if is_vlm:
            client = CountingVLMClient(client)
        return marker.wrap(client, name=name, is_vlm=is_vlm)

    vlm_client = _wrap(
        Qwen25VLClient(source=args.qwen, dtype="float16"), "object_description", is_vlm=True
    )
    setup_logging("INFO")
    device = config.resolve_device()
    grounding_client = _wrap(
        GroundingDinoClient(source=args.dino, device=device, dtype="float32"), "grounding"
    )
    segmentation_client = _wrap(
        Sam21Client(source=args.sam, device=device, dtype="float32"), "segmentation"
    )
    from manga_animation.reconstruction import LamaClient

    reconstruction_client = _wrap(
        LamaClient(device=device, model_id="lama-large"), "reconstruction"
    )

    report: dict = {
        "phase": "18.4-e2e-profile",
        "ordering": "grounding(DINO) -> object_description(Qwen) -> "
        "segmentation(SAM, accepted only)",
        "timestamp": datetime.now(UTC).isoformat(),
        "pages": [],
    }
    try:
        page_results = run_pages(
            [Path(page) for page in args.pages],
            config,
            vlm_client=vlm_client,
            grounding_client=grounding_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            out_dir=out_dir / "videos",
            labels=args.labels,
        )
        for page_result in page_results:
            page_entry = {
                "page": page_result.source_image.name,
                "pages": len(page_results),
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
            print(
                f"[profiler] page {page_result.source_image.name} "
                f"runtime={page_result.panels[0].metrics.get('runtime_s', 0)}s",
                flush=True,
            )
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
    report["stage_events"] = marker.stage_events
    report["gpu_sampler_interval_s"] = args.sampler_interval_s
    report["gpu_samples"] = sampler.samples
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"VLM calls: {vlm_client.call_count}, boxes per call: {vlm_client.boxes_per_call}")
    print(f"GPU samples: {len(sampler.samples)}")


if __name__ == "__main__":
    main()
