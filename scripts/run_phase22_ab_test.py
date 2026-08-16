"""Phase 22 panel-mode run with a per-GPU Qwen pool (ADR 0023).

Run the production `run_pages` panel pipeline on real pages with Qwen3-VL-4B fp16 as ONE
instance per CUDA device (`device_map={"": "cuda:N"}`), so the description stage splits the
panels between the cards. This replaces the A/B comparison (panel vs full-page) -- the
full-page mode was rejected by the Phase 22 A/B test (the model's single JSON covered only
10 of 52 boxes -> unparseable -> fail-closed), so only the panel mode is run here.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

Usage:

    python scripts/run_phase22_ab_test.py \
        --pages examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/models/qwen3_vl_4b \
        --dino /kaggle/models/dino \
        --sam /kaggle/models/sam \
        --out outputs/experiments/phase22_ab_test.json

Writes one git-ignored experiment JSON: per-panel statuses, wall-clock, VLM call counts, and
how many candidate descriptions were parsed per panel (JSON validity).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from manga_animation.analysis import Qwen3VLClient
from manga_animation.core.config import PipelineConfig, load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.pipeline.orchestrator import DEFAULT_ANIMATION_LABELS
from manga_animation.pipeline.panels import run_pages
from manga_animation.reconstruction import LamaClient
from manga_animation.segmentation import Sam21Client


class CountingVLM:
    """Counts generate() calls of the real Qwen client (one call per panel it processes)."""

    def __init__(self, inner):
        self._inner = inner
        self.call_count = 0

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        return self._inner.generate(image, prompt)

    def unload(self) -> None:
        self._inner.unload()


def _panel_report(page_result) -> list[dict]:
    return [
        {
            "panel_id": p.panel_id,
            "status": p.status,
            "failure_stage": p.failure_stage,
            "failure_reason": p.failure_reason,
            "runtime_s": p.metrics.get("runtime_s"),
            "frame_count": p.metrics.get("frame_count"),
        }
        for p in page_result.panels
    ]


def _checkpoint_counts(page_result) -> tuple[int, int]:
    """(grounded candidates, parsed descriptions) from the page's checkpoint files."""
    page_dir = page_result.manifest_path.parent
    n_grounded = 0
    grounding_path = page_dir / "grounding.json"
    if grounding_path.exists():
        payload = json.loads(grounding_path.read_text())
        n_grounded = sum(
            len(cands)
            for panel in payload.values()
            for cands in panel.get("candidates", {}).values()
        )
    n_parsed = 0
    descriptions_path = page_dir / "descriptions.json"
    if descriptions_path.exists():
        payload = json.loads(descriptions_path.read_text())
        n_parsed = sum(len(panel) for panel in payload.values())
    return n_grounded, n_parsed


def _run_panel_mode(
    page_path: Path,
    config: PipelineConfig,
    *,
    labels: list[str],
    qwen_pool: Sequence[Qwen3VLClient],
    dino: GroundingDinoClient,
    sam_source: str,
    aux_device: str,
    out_dir: Path,
) -> dict:
    """Production panel mode: `run_pages` with a per-GPU VLM pool -- one Qwen call per panel,
    panels split between the instances (ADR 0023)."""
    counting = [CountingVLM(client) for client in qwen_pool]
    started = time.perf_counter()
    page_result = run_pages(
        [page_path],
        config,
        vlm_client=counting,
        grounding_client=dino,
        segmentation_client=Sam21Client(
            source=sam_source, device=aux_device, dtype="float32"
        ),
        reconstruction_client=LamaClient(device=aux_device, model_id="lama-large"),
        out_dir=out_dir,
        labels=labels,
    )[0]
    n_grounded, n_parsed = _checkpoint_counts(page_result)
    return {
        "elapsed_s": round(time.perf_counter() - started, 1),
        "vlm_calls": sum(c.call_count for c in counting),
        "n_grounded_candidates": n_grounded,
        "n_parsed_descriptions": n_parsed,
        "panels": _panel_report(page_result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True, help="Qwen3-VL-4B model dir on the worker")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="kaggle")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="candidate semantic labels (default: the pipeline default list)",
    )
    args = parser.parse_args()

    config = load_config(args.env, overrides={"resolution": 1536})
    config.model_variants.update(
        {
            "grounding": "grounding-dino-swin-l",
            "segmentation": "sam2.1-hiera-base",
        }
    )
    setup_logging(False)
    # Pre-load transformers modules in the main thread BEFORE the panel pipeline spawns
    # worker threads: transformers 5.0.0 lazy-imports its submodules, and two workers
    # importing concurrently (Qwen describe + SAM load) can race and surface a spurious
    # "cannot import name 'Sam2Model' from 'transformers'" (observed on a real T4 run).
    import transformers  # noqa: F401
    from transformers import Sam2Model, Sam2Processor  # noqa: F401

    labels = list(args.labels or DEFAULT_ANIMATION_LABELS)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONE Qwen instance per GPU (device_map={"": "cuda:N"}, ADR 0023): Qwen3-VL-4B fp16
    # (~8.5 GiB) fits a single T4, so the description stage splits panels between the cards
    # instead of sharding one device_map="auto" model across them.
    try:
        import torch

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        n_gpus = 0
    qwen_pool = [
        Qwen3VLClient(
            source=args.qwen,
            dtype="float16",
            max_new_tokens=4096,
            device=f"cuda:{i}" if n_gpus else "cpu",
        )
        for i in range(max(1, n_gpus))
    ]
    # The auxiliary models (DINO/SAM/LaMa) are stage-owned and small (~2.6 GiB total), so they
    # share the last card with one Qwen instance: on a single T4, Qwen + SAM + LaMa + render
    # buffers OOM'd together (real run, Phase 22) -- the pool's fp16 instances (~8.5 GiB each)
    # leave enough headroom on their card for the small models.
    aux_device = "cuda:1" if n_gpus > 1 else ("cuda:0" if n_gpus == 1 else "cpu")
    dino = GroundingDinoClient(source=args.dino, device=aux_device, dtype="float32")

    report: dict = {
        "phase": "22-panel-mode",
        "timestamp": datetime.now(UTC).isoformat(),
        "page": [Path(p).name for p in args.pages],
        "resolution": 1536,
        "vlm_instances": len(qwen_pool),
        "vlm_instance_devices": [c.device for c in qwen_pool],
        "modes": {},
    }
    try:
        for page in args.pages:
            mode = _run_panel_mode(
                Path(page),
                config,
                labels=labels,
                qwen_pool=qwen_pool,
                dino=dino,
                sam_source=args.sam,
                aux_device=aux_device,
                out_dir=out_dir / "mode_a",
            )
            report["modes"]["A"] = mode
            print(json.dumps(mode, indent=1), flush=True)
    finally:
        for client in (*qwen_pool, dino):
            unload = getattr(client, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001 -- best-effort teardown
                    pass

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()