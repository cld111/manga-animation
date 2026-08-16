"""Independent panel extraction, processing, outputs and page manifests.

This module is intentionally an orchestration boundary. It detects panels and creates scene
crops, then delegates every crop's actual animation to the existing ``run_pipeline`` stages.

Phase 18.3 architecture: the pipeline has NO Qwen analysis stage. Every panel is processed
with the same candidate label list: grounding (DINO) -> the pipeline's single VLM stage
(object description: full image + bbox coordinates) -> segmentation (SAM, only for accepted
bboxes) -> animation planning -> animation -> reconstruction -> compositing -> rendering.

Phase 18.4 batch mode: `run_pages` processes MANY pages with stage-level model residency
ACROSS pages. Each model loads ONCE, processes every eligible panel of EVERY page, saves its
results, and only then is released and the next model loads. `run_page_panels` is the
single-page convenience wrapper over the same code path.

Phase 20 co-residency: the sequential stage-by-stage residency (ADR 0020) is replaced by a
run-level residency (ADR 0021) -- ALL model clients are loaded together at the start of a
`run_pages` call and stay resident until it finishes, with deterministic unload only at the
end. Models are still only loaded when their stage has pending work (a fully checkpointed
stage is restored from disk and its model never loads).

Phase 21 panel pipeline (ADR 0022): the stages run CONCURRENTLY as five single-threaded
workers connected by bounded queues. A panel moves to the next model as soon as the previous
stage produced its result -- no stage barrier waits for every panel of every page. Checkpoint
resume is per-panel; PASS/STATIC panels from an earlier manifest are skipped entirely.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import numpy as np
from PIL import Image

from manga_animation.analysis import VLMClient, detect_panels
from manga_animation.analysis.panels import derive_scene_crop_bbox
from manga_animation.core.config import PipelineConfig
from manga_animation.core.logging import get_logger
from manga_animation.grounding import GroundingClient
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.orchestrator import (
    DEFAULT_ANIMATION_LABELS,
    DroppedObjectResult,
    _animate_objects,
    _build_plan,
    _composite_and_render,
    _describe_candidates,
    _ground_labels,
    _reconstruct_objects,
    _segment_candidates,
)
from manga_animation.pipeline.persistence import (
    has_descriptions,
    has_grounding,
    has_segmentation,
    load_descriptions,
    load_grounding,
    load_segmentation,
    save_descriptions,
    save_grounding,
    save_segmentation,
)
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    Layer,
    ObjectDescriptionResult,
    PanelStatus,
    PanelUnit,
    PipelineStageError,
    ReconstructionResult,
    SegmentationResult,
)
from manga_animation.reconstruction import ReconstructionClient
from manga_animation.schemas.animation_plan import AnimationPlan, ObjectPlan
from manga_animation.segmentation import SegmentationClient

logger = get_logger(__name__)


@dataclass
class PagePanelsResult:
    """Results for all detected panels, including panels that did not render."""

    page_id: str
    source_image: Path
    manifest_path: Path
    panels: list[PanelUnit]


_SAFE_REJECTION_STAGES = {"grounding", "segmentation", "object_description"}


def _failure_status(stage: str) -> PanelStatus:
    """Map a failing stage to the panel's status: safe model-gate rejections are REJECTED,
    everything else is ERROR."""
    return "REJECTED" if stage in _SAFE_REJECTION_STAGES else "ERROR"


def _write_manifest(
    manifest_path: Path,
    page_id: str,
    source_image: Path,
    panels: list[PanelUnit],
    *,
    started_at: float,
) -> None:
    payload = {
        "page_id": page_id,
        "source_image": str(source_image),
        "panels": [panel.as_manifest_dict() for panel in panels],
        "performance": {
            "detected_panel_count": len(panels),
            "scene_crop_pixels": sum(
                panel.scene_crop_bbox.width * panel.scene_crop_bbox.height for panel in panels
            ),
            "elapsed_s": round(time.perf_counter() - started_at, 6),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def _existing_resumable_panel(
    existing: dict[str, object], panel: PanelUnit
) -> PanelStatus | None:
    if existing.get("status") == "PASS":
        output = existing.get("output_video")
        if isinstance(output, str) and Path(output).exists():
            return "PASS"
    if existing.get("status") == "STATIC" and panel.scene_crop_path.exists():
        return "STATIC"
    return None


def _load_existing_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(item["panel_id"]): item
        for item in payload.get("panels", [])
        if isinstance(item, dict) and "panel_id" in item
    }


def _set_failure(panel: PanelUnit, status: PanelStatus, stage: str, reason: str) -> None:
    panel.status = status
    panel.failure_stage = stage
    panel.failure_reason = reason


@dataclass
class _PageRunState:
    """Everything one page accumulates across the run's model stages.

    Stage outputs live here across stages. Phase 20 co-residency (ADR 0021): ALL models with
    pending work stay resident for the whole `run_pages` call -- there is no per-stage
    load/unload. Phase 21 (ADR 0022) executes the stages concurrently as a panel pipeline;
    each field below is written by exactly one pipeline worker per panel, and readers only
    touch a panel's entries after the producing worker handed the token downstream (queue
    happens-before), so no cross-thread races exist.
    """

    image_path: Path
    page_id: str
    page_dir: Path
    manifest_path: Path
    existing: dict[str, dict[str, object]]
    panels: list[PanelUnit]
    crops: dict[str, np.ndarray]
    panel_started_at: dict[str, float]
    candidates_by_panel: dict[str, dict[str, list[GroundingResult]]] = field(default_factory=dict)
    plan_by_object_by_panel: dict[str, dict[str, ObjectPlan]] = field(default_factory=dict)
    dropped_by_panel: dict[str, list[DroppedObjectResult]] = field(default_factory=dict)
    descriptions_by_panel: dict[str, dict[tuple[str, int], ObjectDescriptionResult]] = field(
        default_factory=dict
    )
    segmentation_by_panel: dict[str, dict[tuple[str, int], SegmentationResult]] = field(
        default_factory=dict
    )
    plans_by_panel: dict[str, AnimationPlan] = field(default_factory=dict)
    animated_by_panel: dict[str, list[ObjectPlan]] = field(default_factory=dict)
    seg_by_object_by_panel: dict[str, dict[str, SegmentationResult]] = field(default_factory=dict)
    layers_by_panel: dict[str, list[Layer]] = field(default_factory=dict)
    layers_by_object_by_panel: dict[str, dict[str, Layer]] = field(default_factory=dict)
    reconstructions_by_panel: dict[str, dict[str, ReconstructionResult]] = field(
        default_factory=dict
    )


def _prepare_page_state(
    image_path: Path, out_dir: Path, config: PipelineConfig
) -> _PageRunState:
    """Detect panels, write scene crops and build the initial state for one page."""
    image_path = image_path.resolve()
    page_id = image_path.stem
    page_dir = out_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = page_dir / "page_manifest.json"
    existing = _load_existing_manifest(manifest_path)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    page_shape = image.shape[:2]
    candidates = detect_panels(image)
    if not candidates:
        raise PipelineStageError(
            stage="analysis",
            input_ref=str(image_path),
            detail="panel detector returned no usable candidates",
            root_cause="the source image is too small or degenerate for deterministic detection",
            architectural=False,
            proposed_fix="provide a larger source page or use an explicit panel annotation",
        )

    panel_bboxes = tuple(candidate.panel_bbox for candidate in candidates)
    panels: list[PanelUnit] = []
    crops: dict[str, np.ndarray] = {}
    for index, candidate in enumerate(candidates, start=1):
        panel_id = f"panel_{index:03d}"
        panel_bbox = candidate.panel_bbox
        scene_bbox = derive_scene_crop_bbox(
            panel_bbox,
            page_shape,
            neighboring_panel_bboxes=panel_bboxes,
        )
        crop_path = page_dir / "crops" / f"{panel_id}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop = image[scene_bbox.y0 : scene_bbox.y1, scene_bbox.x0 : scene_bbox.x1]
        crops[panel_id] = crop
        Image.fromarray(crop).save(crop_path)
        panel = PanelUnit(
            page_id=page_id,
            panel_id=panel_id,
            panel_order=index,
            panel_bbox=panel_bbox,
            scene_crop_bbox=scene_bbox,
            source_page=image_path,
            scene_crop_path=crop_path,
        )
        panel.metrics.update(
            {
                "scene_crop_width": scene_bbox.width,
                "scene_crop_height": scene_bbox.height,
                "scene_crop_pixels": scene_bbox.width * scene_bbox.height,
            }
        )
        panels.append(panel)

        resumed = _existing_resumable_panel(existing.get(panel_id, {}), panel)
        if resumed is not None:
            panel.status = resumed
            output = existing[panel_id].get("output_video")
            panel.output_video = Path(output) if isinstance(output, str) else None

    return _PageRunState(
        image_path=image_path,
        page_id=page_id,
        page_dir=page_dir,
        manifest_path=manifest_path,
        existing=existing,
        panels=panels,
        crops=crops,
        panel_started_at={},
    )


def _crop_local_panel_bbox(state: _PageRunState, panel: PanelUnit) -> BBoxPx:
    """The panel's logical bbox translated into its scene crop's local coordinates --
    grounding/description run on the crop canvas, so the region argument must be
    crop-local (the Phase 18.3 flow derives it from the crop geometry directly)."""
    crop = state.crops[panel.panel_id]
    ox, oy = panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0
    h, w = crop.shape[0], crop.shape[1]
    return BBoxPx(
        x0=max(0, panel.panel_bbox.x0 - ox),
        y0=max(0, panel.panel_bbox.y0 - oy),
        x1=min(w, panel.panel_bbox.x1 - ox),
        y1=min(h, panel.panel_bbox.y1 - oy),
    )


def _finalize(
    state: _PageRunState, panel: PanelUnit, status: PanelStatus, stage: str, reason: str
) -> None:
    _set_failure(panel, status, stage, reason)
    start = state.panel_started_at.get(panel.panel_id)
    if start is not None:
        panel.metrics["runtime_s"] = round(time.perf_counter() - start, 6)
    state.panel_started_at.pop(panel.panel_id, None)


def _write_all_manifests(states: list[_PageRunState]) -> None:
    for state in states:
        _write_manifest(
            state.manifest_path,
            state.page_id,
            state.image_path,
            state.panels,
            started_at=0.0,
        )


class _PanelPipelineToken:
    """One panel flowing through the concurrent panel pipeline (Phase 21).

    A token is created per panel with `start_stage` = the first stage that still has work
    for it (0=grounding, 1=object description, 2=segmentation, 3=plan/animate/reconstruct,
    4=render). A worker whose stage is below `start_stage` passes the token through without
    processing it -- which is how resume (checkpointed stages) and the fresh-run path share
    one pipeline.
    """

    __slots__ = ("state", "panel", "start_stage")

    def __init__(self, state: _PageRunState, panel: PanelUnit, start_stage: int) -> None:
        self.state = state
        self.panel = panel
        self.start_stage = start_stage


def _panel_start_stage(state: _PageRunState, panel: PanelUnit) -> int:
    """The first pipeline stage that still has work for this panel, given the checkpoints
    already loaded into `state` (Phase 18.4 persistence): a panel absent from a stage's
    checkpoint re-runs that stage -- per-panel resume, not per-page."""
    panel_id = panel.panel_id
    if panel_id not in state.candidates_by_panel:
        return 0
    if panel_id not in state.descriptions_by_panel:
        return 1
    if panel_id not in state.segmentation_by_panel:
        return 2
    return 3


def _persist_stage(state: _PageRunState, stage: int) -> None:
    """Re-write the checkpoint file of one completed stage for one page. Each stage's
    checkpoint is written only by that stage's pipeline worker (one writer per file)."""
    if stage == 0:
        save_grounding(
            state.page_dir,
            state.candidates_by_panel,
            state.plan_by_object_by_panel,
            state.dropped_by_panel,
        )
    elif stage == 1:
        save_descriptions(state.page_dir, state.descriptions_by_panel)
    elif stage == 2:
        save_segmentation(state.page_dir, state.segmentation_by_panel)


def _pipeline_stage_grounding(
    token: _PanelPipelineToken,
    labels: Sequence[str],
    grounding_client: GroundingClient,
) -> bool:
    """DINO grounds every candidate label on this panel's crop (stage 0)."""
    state, panel = token.state, token.panel
    panel_id = panel.panel_id
    state.panel_started_at[panel_id] = time.perf_counter()
    try:
        plans, grounded, dropped = _ground_labels(
            state.crops[panel_id],
            labels,
            grounding_client,
            panel_bbox_px=_crop_local_panel_bbox(state, panel),
        )
    except PipelineStageError as exc:
        _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
        return False
    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
        return False
    state.candidates_by_panel[panel_id] = grounded
    state.plan_by_object_by_panel[panel_id] = {p.object_id: p for p in plans}
    state.dropped_by_panel[panel_id] = dropped
    _persist_stage(state, 0)
    return True


def _pipeline_stage_description(
    token: _PanelPipelineToken,
    vlm_client: VLMClient,
    config: PipelineConfig,
    persist_lock: Lock | None = None,
) -> bool:
    """The pipeline's single VLM stage: Qwen describes this panel's grounded candidates
    (stage 1).

    In the Phase 22 int8 scheme several VLM workers (one per GPU instance) share this stage
    concurrently, so the per-panel checkpoint write is guarded by `persist_lock` (one file
    per page, several writers)."""
    state, panel = token.state, token.panel
    panel_id = panel.panel_id
    if panel_id not in state.candidates_by_panel:
        return False  # failed at grounding
    try:
        desc, dropped = _describe_candidates(
            state.crops[panel_id],
            state.candidates_by_panel[panel_id],
            state.plan_by_object_by_panel[panel_id],
            vlm_client,
            config,
            panel_bbox_px=_crop_local_panel_bbox(state, panel),
        )
    except PipelineStageError as exc:
        _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
        return False
    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
        return False
    state.descriptions_by_panel[panel_id] = desc
    state.dropped_by_panel[panel_id].extend(dropped)
    if persist_lock is not None:
        with persist_lock:
            _persist_stage(state, 1)
    else:
        _persist_stage(state, 1)
    return True


def _pipeline_stage_segmentation(
    token: _PanelPipelineToken,
    segmentation_client: SegmentationClient,
) -> bool:
    """SAM segments ONLY the accepted candidates of this panel (stage 2)."""
    state, panel = token.state, token.panel
    panel_id = panel.panel_id
    if panel_id not in state.descriptions_by_panel:
        return False  # failed at grounding or object description
    accepted_keys = {
        key
        for key, description in state.descriptions_by_panel[panel_id].items()
        if description.accepted
    }
    try:
        seg, dropped = _segment_candidates(
            state.crops[panel_id],
            state.candidates_by_panel[panel_id],
            state.plan_by_object_by_panel[panel_id],
            segmentation_client,
            accepted_keys=accepted_keys,
        )
    except PipelineStageError as exc:
        _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
        return False
    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure
        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
        return False
    state.segmentation_by_panel[panel_id] = seg
    state.dropped_by_panel[panel_id].extend(dropped)
    _persist_stage(state, 2)
    return True


def _pipeline_stage_plan(
    token: _PanelPipelineToken,
    config: PipelineConfig,
    reconstruction_client: ReconstructionClient,
) -> bool:
    """Deterministic plan ranking + animation + LaMa reconstruction for this panel
    (stage 3). CV work is CPU-only; LaMa is the only model used here."""
    state, panel = token.state, token.panel
    panel_id = panel.panel_id
    if panel_id not in state.descriptions_by_panel:
        return False  # failed earlier
    state.panel_started_at.setdefault(panel_id, time.perf_counter())
    try:
        accepted = []
        for (object_id, rank), description in state.descriptions_by_panel[panel_id].items():
            if not description.accepted:
                continue
            if (object_id, rank) not in state.segmentation_by_panel[panel_id]:
                continue  # accepted by the VLM but dropped at segmentation
            accepted.append(
                (
                    object_id,
                    rank,
                    state.plan_by_object_by_panel[panel_id][object_id],
                    state.candidates_by_panel[panel_id][object_id][rank],
                    state.segmentation_by_panel[panel_id][(object_id, rank)],
                    description,
                )
            )
        plan, primary, kept = _build_plan(
            panel.scene_crop_path,
            state.crops[panel_id].shape[:2],
            config,
            accepted=accepted,
            global_origin=(panel.scene_crop_bbox.x0, panel.scene_crop_bbox.y0),
            logical_panel_bbox_px=panel.panel_bbox,
            neighboring_panel_bboxes=tuple(p.panel_bbox for p in state.panels),
        )
        animated_objects = [item[0] for item in kept]
        segmentation_by_object = {item[0].object_id: item[2] for item in kept}
        panel_bbox_px_by_object = {
            obj.object_id: panel.panel_bbox for obj in animated_objects
        }
        layers, layers_by_object = _animate_objects(
            state.crops[panel_id],
            animated_objects,
            segmentation_by_object,
            panel_bbox_px_by_object,
            plan,
        )
        reconstructions = _reconstruct_objects(
            state.crops[panel_id],
            animated_objects,
            segmentation_by_object,
            layers_by_object,
            reconstruction_client,
            config,
        )
        state.plans_by_panel[panel_id] = plan
        state.animated_by_panel[panel_id] = animated_objects
        state.seg_by_object_by_panel[panel_id] = segmentation_by_object
        state.layers_by_panel[panel_id] = layers
        state.layers_by_object_by_panel[panel_id] = layers_by_object
        state.reconstructions_by_panel[panel_id] = reconstructions
    except PipelineStageError as exc:
        _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
        return False
    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
        return False
    return True


def _pipeline_stage_render(token: _PanelPipelineToken, config: PipelineConfig) -> bool:
    """Render this panel's frames to H.264, update its status, and re-write the page
    manifest (stage 4, CPU)."""
    state, panel = token.state, token.panel
    panel_id = panel.panel_id
    if panel_id not in state.plans_by_panel:
        return False  # failed earlier
    try:
        render_result = _composite_and_render(
            state.crops[panel_id],
            state.layers_by_panel[panel_id],
            state.reconstructions_by_panel[panel_id],
            state.plans_by_panel[panel_id],
            state.page_dir,
            config,
            video_filename=f"{panel_id}.mp4",
            frames_dir=state.page_dir / "frames" / panel_id,
        )
        panel.status = "PASS"
        panel.output_video = render_result.output_path
        panel.metrics["frame_count"] = render_result.frame_count
        panel.metrics["runtime_s"] = round(
            time.perf_counter() - state.panel_started_at.pop(panel_id), 6
        )
        # Crash-safe manifest: a killed run keeps every rendered panel reusable (PASS).
        _write_manifest(
            state.manifest_path,
            state.page_id,
            state.image_path,
            state.panels,
            started_at=0.0,
        )
    except PipelineStageError as exc:
        _finalize(state, panel, _failure_status(exc.stage), exc.stage, exc.detail)
    except Exception as exc:  # noqa: BLE001 -- isolate unexpected failure to this panel
        _finalize(state, panel, "ERROR", type(exc).__name__, str(exc))
    return True


def _pipeline_worker(
    in_q: Queue,
    out_q: Queue | None,
    stage: int,
    process,
    errors: list[BaseException],
    *,
    end_expected: int = 1,
    end_sent: int = 1,
    owned_client: object | None = None,
    owned_name: str | None = None,
) -> None:
    """One pipeline stage's worker: pull tokens from `in_q`, process the ones that still
    need this stage, forward the survivors to `out_q`. `None` is the shutdown sentinel.

    The description stage (Phase 22, ADR 0023) runs as a WORKER POOL -- one worker per VLM
    instance, all consuming the same input queue and feeding the same output queue. To make
    that safe, `end_expected` is how many sentinels this worker must observe before it
    stops (1 for a single producer; N when N workers feed this queue), and `end_sent` is how
    many sentinels it emits on shutdown (N when N consumers wait on `out_q`). The upstream
    stage emits one sentinel PER downstream consumer. Any worker failure is recorded and
    still emits its sentinels, so no worker can deadlock waiting on a queue whose producer
    died.

    `owned_client` gives the worker stage-level residency for the NON-Qwen models (Phase 22
    memory split): DINO/SAM/LaMa load when their worker starts and unload when it finishes,
    so the GPU0 Qwen instance is NOT permanently joined by 2.6 GiB of small models -- a full
    int8 Qwen + KV cache + prefill does not fit with them co-resident (real OOM on the
    worker). The VLM instances stay run-level resident (ADR 0021)."""
    def run_loop() -> None:
        ended = 0
        while ended < end_expected:
            token = in_q.get()
            if token is None:
                ended += 1
                continue
            if token.start_stage <= stage:
                if not process(token):
                    continue  # panel failed at this stage -- it leaves the pipeline
            if out_q is not None:
                out_q.put(token)

    try:
        if owned_client is not None:
            with ModelStage(owned_client, name=owned_name or f"stage-{stage}"):
                run_loop()
        else:
            run_loop()
    except BaseException as exc:  # noqa: BLE001 -- worker-level failure (not panel-level)
        errors.append(exc)
    finally:
        if out_q is not None:
            for _ in range(end_sent):
                out_q.put(None)


def _run_panel_pipeline(
    states: list[_PageRunState],
    *,
    labels: Sequence[str],
    config: PipelineConfig,
    grounding_client: GroundingClient,
    vlm_client: VLMClient | Sequence[VLMClient],
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    need_dino: bool,
    need_qwen: bool,
    need_sam: bool,
) -> None:
    """Run the five-stage concurrent panel pipeline over every eligible panel.

    Each stage has one or more worker threads pulling from its input queue and pushing
    survivors to the next stage's queue (bounded, giving backpressure). A panel moves to
    the next model as soon as the previous stage produced its result -- no stage barrier
    (Phase 21). The object-description stage runs as a WORKER POOL (Phase 22, ADR 0023):
    one worker per VLM instance (one int8 Qwen per GPU), all consuming the shared panel
    queue, so panels are split between the GPUs. Per-panel results are identical to the
    sequential scheme regardless of which worker processed a panel: each worker computes
    the same per-panel stage function, and checkpoints are written per panel.

    Memory split (Phase 22): the VLM instances are run-level resident (ADR 0021); the
    smaller models (DINO/SAM/LaMa) are stage-owned -- loaded when their worker starts,
    unloaded when it finishes -- so a full int8 Qwen + KV cache + prefill is not joined by
    2.6 GiB of permanently resident small models on the same card (a real OOM otherwise).
    """
    vlm_clients = list(vlm_client) if isinstance(vlm_client, Sequence) else [vlm_client]
    n_desc = len(vlm_clients)
    q_ground: Queue = Queue(maxsize=8)
    q_desc: Queue = Queue(maxsize=8)
    q_seg: Queue = Queue(maxsize=8)
    q_plan: Queue = Queue(maxsize=8)
    q_render: Queue = Queue(maxsize=8)
    errors: list[BaseException] = []
    persist_lock = Lock()

    workers: list[Thread] = [
        Thread(
            target=_pipeline_worker,
            args=(
                q_ground,
                q_desc,
                0,
                partial(
                    _pipeline_stage_grounding,
                    labels=labels,
                    grounding_client=grounding_client,
                ),
                errors,
            ),
            kwargs={
                "end_sent": n_desc,  # one sentinel per description worker
                "owned_client": grounding_client if need_dino else None,
                "owned_name": "grounding",
            },
            name="pipeline-grounding",
            daemon=True,
        ),
    ]
    for index, vlm in enumerate(vlm_clients):
        workers.append(
            Thread(
                target=_pipeline_worker,
                args=(
                    q_desc,
                    q_seg,
                    1,
                    partial(
                        _pipeline_stage_description,
                        vlm_client=vlm,
                        config=config,
                        persist_lock=persist_lock,
                    ),
                    errors,
                ),
                kwargs={"end_expected": 1, "end_sent": 1},
                name=f"pipeline-description-{index}",
                daemon=True,
            )
        )
    workers.extend(
        [
            Thread(
                target=_pipeline_worker,
                args=(
                    q_seg,
                    q_plan,
                    2,
                    partial(
                        _pipeline_stage_segmentation,
                        segmentation_client=segmentation_client,
                    ),
                    errors,
                ),
                kwargs={
                    "end_expected": n_desc,  # every description worker terminates us
                    "owned_client": segmentation_client if need_sam else None,
                    "owned_name": "segmentation",
                },
                name="pipeline-segmentation",
                daemon=True,
            ),
            Thread(
                target=_pipeline_worker,
                args=(
                    q_plan,
                    q_render,
                    3,
                    partial(
                        _pipeline_stage_plan,
                        config=config,
                        reconstruction_client=reconstruction_client,
                    ),
                    errors,
                ),
                kwargs={
                    "owned_client": reconstruction_client,
                    "owned_name": "reconstruction",
                },
                name="pipeline-plan-animate-reconstruct",
                daemon=True,
            ),
            Thread(
                target=_pipeline_worker,
                args=(
                    q_render,
                    None,
                    4,
                    partial(_pipeline_stage_render, config=config),
                    errors,
                ),
                name="pipeline-render",
                daemon=True,
            ),
        ]
    )
    for worker in workers:
        worker.start()
    try:
        for state in states:
            for panel in state.panels:
                if panel.status in ("PASS", "STATIC"):
                    continue  # resumed from an earlier manifest
                q_ground.put(
                    _PanelPipelineToken(
                        state, panel, _panel_start_stage(state, panel)
                    )
                )
    finally:
        q_ground.put(None)
    for worker in workers:
        worker.join(timeout=600)
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError("panel pipeline workers failed to terminate")
    if errors:
        raise errors[0]


def run_pages(
    image_paths: Sequence[Path],
    config: PipelineConfig,
    *,
    vlm_client: VLMClient | Sequence[VLMClient],
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    labels: Sequence[str] | None = None,
) -> list[PagePanelsResult]:
    """Process MANY pages through the concurrent panel pipeline (Phase 21).

    ALL model clients that have pending work are loaded TOGETHER at the start of the call
    and stay resident until the whole run finishes (Phase 20 co-residency, ADR 0021), then
    pipeline stages (grounding -> object description -> segmentation ->
    plan/animate/reconstruct -> render) process panels through bounded queues: a panel
    moves to the next model as soon as the previous stage produced its result, with NO
    stage barrier waiting for all pages (Phase 21, ADR 0022). The object-description stage
    accepts ONE VLM client OR a pool of them (Phase 22, ADR 0023): one int8 Qwen instance
    per GPU, all consuming the shared panel queue, so panels are split across the GPUs.

    Resume is per-panel (Phase 18.4 persistence): a panel whose checkpoint entry exists for
    a stage skips that stage -- and its model is never loaded if NO panel needs it -- so a
    killed session resumes from the last completed stage instead of re-running
    DINO/Qwen/SAM from scratch. The VLM runs exactly ONCE per panel, before segmentation
    (DINO -> Qwen -> SAM): SAM segments only the bboxes that earned an action description.
    A panel failure is recorded and processing continues; manifests are written after each
    rendered panel so completed PASS panels are reusable on a later invocation.
    """
    active_labels = list(labels or DEFAULT_ANIMATION_LABELS)
    states = [_prepare_page_state(path, out_dir, config) for path in image_paths]
    _write_all_manifests(states)

    # -------------------------------------------------------------------------------------
    # Restore every completed model stage from disk FIRST (Phase 18.4 persistence): the
    # restored stages must not load, and the remaining pending panels define what loads.
    # -------------------------------------------------------------------------------------
    resume_grounding = [s for s in states if has_grounding(s.page_dir)]
    for state in resume_grounding:
        (
            state.candidates_by_panel,
            state.plan_by_object_by_panel,
            state.dropped_by_panel,
        ) = load_grounding(state.page_dir)
        logger.info(
            "grounding: restored %d panel(s) from %s (no DINO load)",
            len(state.candidates_by_panel),
            state.page_dir / "grounding.json",
        )
    resume_descriptions = [s for s in states if has_descriptions(s.page_dir)]
    for state in resume_descriptions:
        state.descriptions_by_panel = load_descriptions(state.page_dir)
        logger.info(
            "object description: restored %d panel(s) from %s (no Qwen load)",
            len(state.descriptions_by_panel),
            state.page_dir / "descriptions.json",
        )
    resume_segmentation = [s for s in states if has_segmentation(s.page_dir)]
    for state in resume_segmentation:
        state.segmentation_by_panel = load_segmentation(state.page_dir)
        logger.info(
            "segmentation: restored %d panel(s) from %s (no SAM load)",
            len(state.segmentation_by_panel),
            state.page_dir / "segmentation.json",
        )

    starts = [
        _panel_start_stage(state, panel)
        for state in states
        for panel in state.panels
        if panel.status not in ("PASS", "STATIC")
    ]
    need_dino = any(start == 0 for start in starts)
    need_qwen = any(start <= 1 for start in starts)
    need_sam = any(start <= 2 for start in starts)

    # -------------------------------------------------------------------------------------
    # Phase 20/22 residency: the VLM instances are run-level co-resident (ADR 0021) --
    # loaded together, resident for the whole run, released together at the end. DINO/SAM/
    # LaMa are stage-owned instead (loaded in their worker, unloaded when its stage
    # finishes): a full int8 Qwen per GPU needs the card's headroom for its KV cache and
    # prefill, and 2.6 GiB of permanently resident small models OOM'd the card (real OOM
    # on the worker). ExitStack exits unwind on completion AND on exception, so a failed
    # run still deterministically releases the VLM instances.
    # -------------------------------------------------------------------------------------
    with ExitStack() as residency:
        if need_qwen:
            vlm_clients = (
                list(vlm_client) if isinstance(vlm_client, Sequence) else [vlm_client]
            )
            for index, vlm in enumerate(vlm_clients):
                residency.enter_context(
                    ModelStage(vlm, name=f"object_description_{index}")
                )

        # ---------------------------------------------------------------------------------
        # The concurrent panel pipeline: five stages, bounded queues between them (Phase
        # 21, ADR 0022). No stage barrier: each panel moves forward as soon as the
        # previous stage produced its result. The description stage is a worker pool of
        # one int8 Qwen per GPU (Phase 22, ADR 0023).
        # ---------------------------------------------------------------------------------
        _run_panel_pipeline(
            states,
            labels=active_labels,
            config=config,
            grounding_client=grounding_client,
            vlm_client=vlm_client,
            segmentation_client=segmentation_client,
            reconstruction_client=reconstruction_client,
            need_dino=need_dino,
            need_qwen=need_qwen,
            need_sam=need_sam,
        )

    _write_all_manifests(states)

    return [
        PagePanelsResult(
            page_id=state.page_id,
            source_image=state.image_path,
            manifest_path=state.manifest_path,
            panels=state.panels,
        )
        for state in states
    ]


def run_page_panels(
    image_path: Path,
    config: PipelineConfig,
    *,
    vlm_client: VLMClient,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
    reconstruction_client: ReconstructionClient,
    out_dir: Path,
    labels: Sequence[str] | None = None,
) -> PagePanelsResult:
    """Single-page convenience wrapper over `run_pages` (Phase 18.4 batch residency:
    each model loads once per call -- here, once for this one page)."""
    results = run_pages(
        [image_path],
        config,
        vlm_client=vlm_client,
        grounding_client=grounding_client,
        segmentation_client=segmentation_client,
        reconstruction_client=reconstruction_client,
        out_dir=out_dir,
        labels=labels,
    )
    return results[0]
