"""Phase 15 real GPU regression validation of the Phase 14 stage-level model lifecycle.

Runs the production page entry point (`run_page_panels`) across MULTIPLE real manga pages in
ONE Python process (so cross-page model lifecycle is exercised, not just intra-page panel
lifecycle), sampling GPU memory continuously and capturing every ModelStage release log.

Deliberately answers the Phase 15 question -- "is the Phase 14 lifecycle stable across multiple
real pages and repeated GPU runs, without regressions?" -- with real GPU evidence, not mocks:

- Multiple pages in one session (detects cross-page state leakage / allocator growth).
- Repeated execution of the same page in the same session (detects stale GPU state / hidden
  model references / behavior differences caused by prior execution).
- Per-stage VRAM evidence via ModelStage release logs + a continuous memory timeline.
- A resume test (same out_dir twice) verifying PASS/STATIC panels are reused.
- An optional injected grounding failure (same class of exception as a CUDA OOM) verifying a
  real mid-pipeline exception isolates to its panel and still releases the model.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md's standing policy, ADR 0003).

Usage (models downloaded to local dirs on the worker, pages fetched):

    python scripts/run_phase15_gpu_regression.py \
        --pages examples/realworld/villainess_ending_scuffle.png \
                examples/realworld/space_monster_hypersenses.png \
                examples/realworld/villainess_ending_scuffle.png \
        --qwen /kaggle/working/models/qwen \
        --dino /kaggle/working/models/dino \
        --sam /kaggle/working/models/sam \
        --out outputs/experiments/phase15_gpu_regression_<ts>.json

    python scripts/run_phase15_gpu_regression.py --resume-test <page> \
        --qwen ... --dino ... --sam ... \
        --out outputs/experiments/phase15_resume_<ts>.json

    python scripts/run_phase15_gpu_regression.py --pages <page> --inject-grounding-failure 2 \
        --qwen ... --dino ... --sam ...

1xT4 lane (Phase 15 section 12): on a 2xT4 worker, restrict to one visible device for the
smoke/E2E run; on a genuine 1xT4 worker this is automatic:

    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase15_gpu_regression.py --pages <page> \
        --qwen ... --dino ... --sam ...

Writes one git-ignored experiment JSON per invocation.
"""

from __future__ import annotations

import argparse
import io
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


def _snapshot_vram(now: float, started_at: float) -> dict[str, object]:
    """Full VRAM snapshot across all visible devices (allocated/reserved/peak) plus a
    process-wide live-CUDA-tensor scan (the forensic complement to allocator stats: detects
    tensors still referenced from Python even when the caching allocator reports them free)."""
    import torch

    torch.cuda.synchronize()
    rec: dict[str, object] = {
        "elapsed_s": round(now - started_at, 3),
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
    import gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                count += 1
                bytes_live += obj.numel() * obj.element_size()
        except Exception:  # noqa: BLE001 -- a single weird object must not abort the scan
            continue
    rec["live_cuda_tensors"] = count
    rec["live_cuda_mb"] = round(bytes_live / 2**20, 1)
    return rec


def _mem_sample_loop(
    stop: threading.Event, timeline: list[dict[str, object]], started_at: float
) -> None:
    while not stop.wait(_MEMORY_SAMPLE_INTERVAL_S):
        try:
            timeline.append(_snapshot_vram(time.perf_counter(), started_at))
        except Exception:  # noqa: BLE001 -- a failed sync must not kill the sampler
            break


def _capture_stage_release_logs() -> tuple[io.StringIO, logging.Handler]:
    import manga_animation.core.logging as core_logging

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    core_logging.get_logger("manga_animation.pipeline").addHandler(handler)
    return buffer, handler


def _release_logs_from_buffer(buffer, logger_name: str = "manga_animation.pipeline") -> list[str]:
    """Pull and clear the captured pipeline-log lines that recorded a ModelStage release."""
    lines = buffer.getvalue().splitlines()
    buffer.truncate(0)
    buffer.seek(0)
    return [line for line in lines if "released" in line]


def _teardown_log_handler(handler: logging.Handler) -> None:
    import manga_animation.core.logging as core_logging

    core_logging.get_logger("manga_animation.pipeline").removeHandler(handler)


class _FailAfterGroundingCalls:
    """Wrap the real Grounding DINO client and raise an unexpected RuntimeError on the Nth
    `detect()` call -- the same class of failure as a CUDA OOM (raw, non-PipelineStageError)
    -- to prove a real mid-pipeline exception isolates to its panel and still releases the
    model via ModelStage's exception path."""

    def __init__(self, inner, fail_on_call: int):
        self._inner = inner
        self._fail_on_call = fail_on_call
        self.detect_calls = 0
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self._inner.load()

    def unload(self) -> None:
        self._inner.unload()

    def detect(self, image, text_prompt: str):
        self.detect_calls += 1
        if self.detect_calls == self._fail_on_call:
            raise RuntimeError("injected unexpected grounding failure (simulated CUDA OOM)")
        return self._inner.detect(image, text_prompt)


def _run_page(
    page: Path,
    config,
    vlm,
    dino,
    sam,
    lama,
    out_dir: Path,
    started_at: float,
    release_logs: list[str],
    timeline: list[dict[str, object]],
    *,
    inject_grounding_failure: int | None,
) -> dict[str, object]:
    grounding_client = dino
    if inject_grounding_failure is not None:
        grounding_client = _FailAfterGroundingCalls(dino, inject_grounding_failure)

    before = _snapshot_vram(time.perf_counter(), started_at)
    stop = threading.Event()
    sampler = threading.Thread(
        target=_mem_sample_loop, args=(stop, timeline, started_at), daemon=True
    )
    sampler.start()
    page_started = time.perf_counter()
    try:
        result = run_page_panels(
            page,
            config,
            vlm_client=vlm,
            grounding_client=grounding_client,
            segmentation_client=sam,
            reconstruction_client=lama,
            out_dir=out_dir,
        )
    finally:
        stop.set()
        sampler.join(timeout=10)
    after = _snapshot_vram(time.perf_counter(), started_at)

    peak_mb = 0.0
    for sample in timeline:
        for gpu in sample["gpus"]:
            peak_mb = max(peak_mb, float(gpu["peak_allocated_mb"]))

    return {
        "page": str(page),
        "elapsed_s": round(time.perf_counter() - page_started, 2),
        "out_dir": str(out_dir),
        "panels": [panel.as_manifest_dict() for panel in result.panels],
        "manifest_performance": json.loads(result.manifest_path.read_text())["performance"],
        "vram_before": before,
        "vram_after": after,
        "peak_allocated_mb": peak_mb,
        "release_logs": release_logs[-20:],
        "inject_grounding_failure": inject_grounding_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, nargs="+", default=[])
    parser.add_argument("--resume-test", type=Path, default=None)
    parser.add_argument("--inject-grounding-failure", type=int, default=None)
    parser.add_argument("--qwen", type=str, required=True, help="local Qwen2.5-VL checkpoint")
    parser.add_argument(
        "--dino", type=str, required=True, help="local Grounding DINO checkpoint"
    )
    parser.add_argument("--sam", type=str, required=True, help="local SAM 2.1 checkpoint")
    parser.add_argument("--env", default="kaggle")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.pages and args.resume_test is None:
        parser.error("provide --pages and/or --resume-test")

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

    started_at = time.perf_counter()
    timeline: list[dict[str, object]] = [_snapshot_vram(started_at, started_at)]
    buffer, handler = _capture_stage_release_logs()
    release_logs: list[str] = []
    page_runs: list[dict[str, object]] = []

    try:
        for page in args.pages:
            page = page.resolve()
            run_index = len(page_runs) + 1
            out_dir = Path("outputs/videos/phase15_evidence") / f"{page.stem}_run{run_index:02d}"
            run = _run_page(
                page,
                config,
                vlm,
                dino,
                sam,
                lama,
                out_dir,
                started_at,
                release_logs,
                timeline,
                inject_grounding_failure=args.inject_grounding_failure,
            )
            page_runs.append(run)
            release_logs.extend(_release_logs_from_buffer(buffer))
            print(
                f"[{page.name}] {run['elapsed_s']}s "
                f"statuses={[p['status'] for p in run['panels']]} "
                f"peak={run['peak_allocated_mb']:.0f}MB "
                f"vram_after={run['vram_after']['gpus'][0]['allocated_mb']}MB",
                flush=True,
            )

        if args.resume_test is not None:
            page = args.resume_test.resolve()
            out_dir = Path("outputs/videos/phase15_evidence") / f"{page.stem}_resume"
            first = _run_page(
                page,
                config,
                vlm,
                dino,
                sam,
                lama,
                out_dir,
                started_at,
                release_logs,
                timeline,
                inject_grounding_failure=None,
            )
            release_logs.extend(_release_logs_from_buffer(buffer))
            second = _run_page(
                page,
                config,
                vlm,
                dino,
                sam,
                lama,
                out_dir,
                started_at,
                release_logs,
                timeline,
                inject_grounding_failure=None,
            )
            release_logs.extend(_release_logs_from_buffer(buffer))
            page_runs.append(
                {
                    "resume_test": str(page),
                    "out_dir": str(out_dir),
                    "first_run": first,
                    "second_run": second,
                    "second_run_reused": [
                        {
                            "panel_id": p["panel_id"],
                            "status": p["status"],
                            "output_video": p["output_video"],
                        }
                        for p in second["panels"]
                    ],
                }
            )
            print(
                f"resume-test {page.name}: first={[p['status'] for p in first['panels']]} "
                f"second={[p['status'] for p in second['panels']]}",
                flush=True,
            )
    finally:
        handler.flush()
        _teardown_log_handler(handler)

    end = _snapshot_vram(time.perf_counter(), started_at)
    out_path = args.out or Path(
        f"outputs/experiments/phase15_gpu_regression_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "environment": environment_metadata(device),
        "config_env": args.env,
        "model_variants": config.model_variants,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "single_t4": len(gpus) == 1 and "T4" in (gpus[0]["name"] if gpus else ""),
        "pages": [str(p.resolve()) for p in args.pages],
        "total_elapsed_s": round(time.perf_counter() - started_at, 2),
        "page_runs": page_runs,
        "stage_release_logs": release_logs,
        "memory_timeline": timeline,
        "final_vram": end,
    }
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    print(
        f"total elapsed {record['total_elapsed_s']}s | gpus={gpus} | "
        f"final vram={end['gpus'][0]['allocated_mb']}MB live={end['live_cuda_tensors']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
