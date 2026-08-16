"""Phase 22 A/B: panel-mode vs full-page-mode VLM description on one manga page.

Compares two ways of feeding the VLM the same grounded candidates on the same page:

- **A (panel mode, the current production path)**: `run_pages` describes every panel's
  candidates in its own scene-crop image -- one Qwen call per panel, bboxes in crop-local
  coordinates.
- **B (full-page mode)**: ONE Qwen call for the whole page. All candidates of ALL panels
  are stated in PAGE coordinates (crop-local bbox + crop origin); the model sees the full
  page, so it gets the surrounding-panel context panel mode lacks. The descriptions are
  then split back per panel, checkpointed, and the rest of the pipeline (SAM ->
  plan/animate/reconstruct -> render) continues through the standard `run_pages` resume
  path (neither DINO nor Qwen reloads).

Both modes run on the SAME page with the same grounding (DINO), same labels and same
random seed, so the only difference is how the VLM sees the candidates.

**Run on the Kaggle/Jupyter GPU worker, never locally** (CLAUDE.md, ADR 0003).

Usage:

    python scripts/run_phase22_ab_test.py \
        --pages examples/realworld/wind_breaker_sprint.png \
        --qwen /kaggle/models/qwen3_vl_4b \
        --dino /kaggle/models/dino \
        --sam /kaggle/models/sam \
        --out outputs/experiments/phase22_ab_test.json \
        --resolution-b 2048

Writes one git-ignored experiment JSON: per-mode panel statuses, wall-clock, VLM call
counts, and how many candidate descriptions were parsed per panel (JSON validity).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.analysis import Qwen3VLClient
from manga_animation.core.config import PipelineConfig, load_config
from manga_animation.core.logging import setup_logging
from manga_animation.grounding import GroundingDinoClient
from manga_animation.object_description import CandidateBox, describe_objects
from manga_animation.pipeline.orchestrator import (
    DEFAULT_ANIMATION_LABELS,
    _ground_labels,
)
from manga_animation.pipeline.panels import (
    _crop_local_panel_bbox,
    _prepare_page_state,
    run_pages,
)
from manga_animation.pipeline.persistence import (
    save_descriptions,
    save_grounding,
)
from manga_animation.pipeline.types import BBoxPx
from manga_animation.reconstruction import LamaClient
from manga_animation.segmentation import Sam21Client


class CountingVLM:
    """Counts generate() calls of the real Qwen client (one call per panel in mode A,
    one call per page in mode B)."""

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


def _ground_and_checkpoint(state, labels, dino) -> None:
    """Ground every panel on its crop (the exact production `_ground_labels` path) and
    write the grounding checkpoint the pipeline resumes from."""
    for panel in state.panels:
        panel_id = panel.panel_id
        plans, grounded, dropped = _ground_labels(
            state.crops[panel_id],
            labels,
            dino,
            panel_bbox_px=_crop_local_panel_bbox(state, panel),
        )
        state.candidates_by_panel[panel_id] = grounded
        state.plan_by_object_by_panel[panel_id] = {p.object_id: p for p in plans}
        state.dropped_by_panel[panel_id] = dropped
    save_grounding(
        state.page_dir,
        state.candidates_by_panel,
        state.plan_by_object_by_panel,
        state.dropped_by_panel,
    )


def _run_mode_b(
    page_path: Path,
    config: PipelineConfig,
    *,
    labels: list[str],
    qwen: Qwen3VLClient,
    dino: GroundingDinoClient,
    sam_source: str,
    aux_device: str,
    out_dir: Path,
    max_long_edge: int,
    max_new_tokens: int,
) -> dict:
    """Full-page mode: ground per panel, then ONE Qwen call on the whole page with every
    candidate's box in PAGE coordinates, split the descriptions back per panel, checkpoint
    them, and let `run_pages` resume from the checkpoints (SAM -> render)."""
    qwen.max_new_tokens = max_new_tokens
    started = time.perf_counter()

    state = _prepare_page_state(page_path, out_dir, config)
    dino.load()  # run_pages unloaded DINO at the end of mode A (stage-owned, ADR 0023)
    _ground_and_checkpoint(state, labels, dino)

    # One candidate list for the whole page; bboxes translated to page coordinates.
    full_page = np.asarray(Image.open(page_path).convert("RGB"))
    candidates: list[CandidateBox] = []
    keys: list[tuple[str, str, int]] = []
    n_grounded = 0
    for panel in state.panels:
        panel_id = panel.panel_id
        ox, oy = panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0
        plans = state.plan_by_object_by_panel[panel_id]
        for object_id, ranked in state.candidates_by_panel[panel_id].items():
            for rank, cand in enumerate(ranked):
                candidates.append(
                    CandidateBox(
                        object_id=object_id,
                        semantic_label=plans[object_id].semantic_label,
                        bbox=BBoxPx(
                            x0=cand.bbox.x0 + ox,
                            y0=cand.bbox.y0 + oy,
                            x1=cand.bbox.x1 + ox,
                            y1=cand.bbox.y1 + oy,
                            score=cand.bbox.score,
                        ),
                    )
                )
                keys.append((panel_id, object_id, rank))
                n_grounded += 1

    descriptions_by_panel: dict[str, dict[tuple[str, int], object]] = {}
    if candidates:
        results = describe_objects(
            full_page, candidates, qwen, max_long_edge=max_long_edge
        )
        for (panel_id, object_id, rank), description in zip(keys, results, strict=True):
            descriptions_by_panel.setdefault(panel_id, {})[(object_id, rank)] = description
    save_descriptions(state.page_dir, descriptions_by_panel)

    n_parsed = sum(len(v) for v in descriptions_by_panel.values())

    # Continue from checkpoints: DINO and Qwen are checkpointed, so neither reloads; the
    # pipeline runs SAM -> plan/animate/reconstruct -> render on the same clients.
    page_result = run_pages(
        [page_path],
        config,
        vlm_client=CountingVLM(qwen),
        grounding_client=dino,
        segmentation_client=Sam21Client(
            source=sam_source, device=aux_device, dtype="float32"
        ),
        reconstruction_client=LamaClient(device=aux_device, model_id="lama-large"),
        out_dir=out_dir,
        labels=labels,
    )[0]
    return {
        "mode": "B (full page)",
        "elapsed_s": round(time.perf_counter() - started, 1),
        "vlm_calls": 1 if candidates else 0,
        "n_grounded_candidates": n_grounded,
        "n_parsed_descriptions": n_parsed,
        "panels": _panel_report(page_result),
    }


def _run_mode_a(
    page_path: Path,
    config: PipelineConfig,
    *,
    labels: list[str],
    qwen: Qwen3VLClient,
    dino: GroundingDinoClient,
    sam_source: str,
    aux_device: str,
    out_dir: Path,
) -> dict:
    """Panel mode: the production `run_pages` path, one Qwen call per panel."""
    counting = CountingVLM(qwen)
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
        "mode": "A (panel mode)",
        "elapsed_s": round(time.perf_counter() - started, 1),
        "vlm_calls": counting.call_count,
        "n_grounded_candidates": n_grounded,
        "n_parsed_descriptions": n_parsed,
        "panels": _panel_report(page_result),
    }


def _checkpoint_counts(page_result) -> tuple[int, int]:
    """(grounded candidates, parsed descriptions) from the page's checkpoint files."""
    import json as _json

    page_dir = page_result.manifest_path.parent
    n_grounded = 0
    grounding_path = page_dir / "grounding.json"
    if grounding_path.exists():
        payload = _json.loads(grounding_path.read_text())
        n_grounded = sum(
            len(cands)
            for panel in payload.values()
            for cands in panel.get("candidates", {}).values()
        )
    n_parsed = 0
    descriptions_path = page_dir / "descriptions.json"
    if descriptions_path.exists():
        payload = _json.loads(descriptions_path.read_text())
        n_parsed = sum(len(panel) for panel in payload.values())
    return n_grounded, n_parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", required=True)
    parser.add_argument("--qwen", required=True, help="Qwen3-VL-4B model dir on the worker")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution-b", type=int, default=2048)
    parser.add_argument("--max-new-tokens-b", type=int, default=16000)
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
    setup_logging("INFO")
    # Pre-load transformers modules in the main thread BEFORE the panel pipeline spawns
    # worker threads: transformers 5.0.0 lazy-imports its submodules, and two workers
    # importing concurrently (Qwen describe + SAM load) can race and surface a spurious
    # "cannot import name 'Sam2Model' from 'transformers'" (observed on a real T4 run).
    import transformers  # noqa: F401
    from transformers import Sam2Model, Sam2Processor  # noqa: F401

    labels = list(args.labels or DEFAULT_ANIMATION_LABELS)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # The auxiliary models (DINO/SAM/LaMa) must live on the SECOND GPU while the Qwen VLM
    # (device_map="auto", ~8.5 GiB fp16) occupies the first: on a single T4, Qwen + SAM +
    # LaMa + render buffers OOM'd together (real run, Phase 22 A/B).
    aux_device = "cuda:1"
    qwen = Qwen3VLClient(source=args.qwen, dtype="float16", max_new_tokens=4096)
    dino = GroundingDinoClient(source=args.dino, device=aux_device, dtype="float32")

    report: dict = {
        "phase": "22-ab-test",
        "timestamp": datetime.now(UTC).isoformat(),
        "page": [Path(p).name for p in args.pages],
        "resolution_a": 1536,
        "resolution_b": args.resolution_b,
        "modes": {},
    }
    try:
        for page in args.pages:
            page_path = Path(page)
            mode_a = _run_mode_a(
                page_path,
                config,
                labels=labels,
                qwen=qwen,
                dino=dino,
                sam_source=args.sam,
                aux_device=aux_device,
                out_dir=out_dir / "mode_a",
            )
            report["modes"]["A"] = mode_a
            print(json.dumps(mode_a, indent=1), flush=True)

            mode_b = _run_mode_b(
                page_path,
                config,
                labels=labels,
                qwen=qwen,
                dino=dino,
                sam_source=args.sam,
                aux_device=aux_device,
                out_dir=out_dir / "mode_b",
                max_long_edge=args.resolution_b,
                max_new_tokens=args.max_new_tokens_b,
            )
            report["modes"]["B"] = mode_b
            print(json.dumps(mode_b, indent=1), flush=True)
    finally:
        for client in (qwen, dino):
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
