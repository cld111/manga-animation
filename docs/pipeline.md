# Pipeline

This is the current pipeline contract. Historical phase-specific behavior and experiment
results live in `docs/phase*-results.md` and are not normative unless linked here.

```text
Manga page
    │
    ▼
Deterministic panel extraction  — src/manga_animation/analysis/panels.py
    │  logical panel_bbox -> bounded scene_crop_bbox
    ▼
Independent panel unit          — src/manga_animation/pipeline/panels.py
    │  one crop, status, output and manifest record per panel
    ▼
Object grounding                — src/manga_animation/grounding
    │  DINO, labels supplied by the caller (never invented); panel crop when
    │  available, otherwise full page; returns page coordinates
    ▼
Per-candidate VLM description    — src/manga_animation/object_description
    │  THE pipeline's single VLM call (Phase 18.3): Qwen sees the FULL pipeline image
    │  plus ALL grounded candidate bboxes as pixel coordinates (never a crop of a
    │  candidate, never a mask), judges each candidate (pass/ambiguous/partial/
    │  reject/not_animatable) and produces a structured animation description whose
    │  deterministically-mapped MotionSpec drives the animation stage. Fail-closed:
    │  PRIMARY non-PASS rejects the run, SECONDARY/MICRO drops the object.
    ▼
Generative per-object animation  — src/manga_animation/animation_anything  (2026 architecture)
    │  NO SAM segmentation. For each ACCEPTED candidate: crop the panel at its DINO bbox,
    │  build the prompt from the accepted Qwen description, and animate the crop directly
    │  with AnimateAnything (image + prompt -> frame sequence).
    ▼
Per-object rendering            — src/manga_animation/rendering
    │  each accepted object renders to its OWN H.264 MP4; panel manifest lists them
    ▼
Decoded-output validation        — src/manga_animation/rendering
    │  verify dimensions, timing, decoded frame count and loop metrics
    ▼
H.264 video                      — src/manga_animation/rendering
```

The 2026 architecture change (video-generation-kaggle) replaced the deterministic
SAM + CV animation + LaMa reconstruction + compositing engine with the generative
AnimateAnything engine: the pipeline no longer uses SAM segmentation, per-object CV
transforms, LaMa hole reconstruction, or CV compositing. The animation unit is the
ACCEPTED OBJECT (a DINO bbox crop), not the whole panel. The deterministic
`src/manga_animation/animation`, `segmentation`, `reconstruction` and `compositing` packages
remain in the codebase as the legacy/regression path, selected by NOT passing
`animation_clients` to `run_pages`/`run_page_panels`.

`run_page_panels` is the page-level production entry point. It does not duplicate the stage
implementation: it writes each scene crop and invokes the existing `run_pipeline` once per
panel. The scene crop, not the strict logical `panel_bbox`, is the source image for grounding,
segmentation, CV transforms, reconstruction, compositing and video rendering. Page-space
coordinates are recovered for cross-panel safety checks by adding the scene crop origin.

`run_pages` is the batch entry point (Phase 18.4): it processes MANY pages in one call.
Every model stage persists its outputs to disk (`grounding.json`, `descriptions.json`,
`segmentation.json` + mask `.npz` per page -- the last two only on the deterministic path):
a later invocation loads the completed stages from disk and never re-loads their models, so a
killed session resumes from the last completed stage instead of re-running DINO/Qwen/SAM from
scratch. `run_page_panels` is the single-page convenience wrapper over the same code path.

Since Phase 21 the stage execution is a CONCURRENT PANEL PIPELINE (ADR 0022): five
single-threaded workers (grounding -> object description -> segmentation/animate -> render)
pass panels through bounded queues. A panel moves to the next stage as soon as the previous
stage produced ITS result -- there is no stage barrier that waits for every panel of every
page. Per-panel results are byte-identical to the sequential scheme: tokens enter the pipeline
in fixed page/panel order and each worker processes its queue FIFO. Checkpoints are written
per panel, so resume is per-panel too: a panel absent from a stage's checkpoint re-runs
exactly that stage. On the generative path the SAM segmentation worker is replaced by a
no-op pass-through and the animation stage is a worker pool (one AnimateAnything client per
GPU), so the pipeline effectively runs grounding -> description -> animate -> render.

Each panel is recorded as `PASS`, `STATIC`, `REJECTED` or `ERROR`. A page manifest is written
after every panel so successful outputs can be reused and a later panel failure cannot erase
earlier results. A materially ambiguous grounded bbox crossing another logical panel is safely
rejected; no object splitting, synchronization or ownership graph is attempted.

The implementation order on the deterministic path is intentionally `animation -> reconstruction`:
reconstruction needs transformed masks to know what motion reveals. The layer and safety-gate
blocks are pipeline boundaries, not independent model stages.

## Stage Ownership

- `grounding` and `segmentation`: `segmentation-agent`.
- `animation`, `reconstruction`, `compositing` and layer mechanics: `cv-agent`.
- `animation_anything` (generative engine) and its vendored model code: `cv-agent` together
  with the orchestrating session.
- `rendering`: `video-agent`.
- Cross-stage orchestration and evaluation wiring: the orchestrating session and `qa-agent`.

The **per-candidate VLM description** stage (`src/manga_animation/object_description`,
Phase 18.3) is owned by the orchestrating session/`qa-agent` together with `cv-agent`: it
consumes the accepted grounding bbox (the mask stays downstream-only), and its output --
the mapped `MotionSpec` -- drives the animation stage on the deterministic path and the prompt
builder on the generative path. On the 2026 generative path it runs BEFORE the crop-based
animation: AnimateAnything animates only the bboxes that earned an accepted description.

## Model Lifecycle

Model residency is run-level and explicitly owned (Phase 20, ADR 0021, superseding ADR 0020's
stage-level scheme). `run_pages` loads ALL model clients that have pending work TOGETHER at the
start of the call and each stage processes every eligible panel of every page in turn. Release
happens exactly once, deterministically, at the end: every model-backed stage runs inside a
`ModelStage` context manager (`src/manga_animation/pipeline/lifecycle.py`, composed via
`ExitStack`) that loads the client on entry and releases it on exit -- on success AND on
exception -- by dropping references, collecting cyclic garbage, and flushing the CUDA caching
allocator. The stages themselves execute concurrently as a panel pipeline (Phase 21, ADR 0022)
on top of this residency: each worker calls only its own client, so the co-resident models are
never shared between threads. Resume is preserved: a stage whose checkpoint exists on disk is
restored from it and its model is never loaded, so a killed session resumes from the last
completed stage.

The AnimateAnything engine is SUBPROCESS-BACKED (2026 architecture): `AnimateAnythingClient`
does not hold a model in the main process -- `load()` only verifies the isolated worker
environment, and `animate()` launches `worker.py` under a dedicated Python interpreter (the
pinned diffusers==0.24.0/transformers==4.36.2/torch==2.0.0 stack that conflicts with the
project's `ml` extra). The worker loads the checkpoint on its own device, generates the
frames, writes them, and exits; the client reads them back. It is therefore stage-owned like
the other small models, with no resident GPU footprint in the pipeline process.

The VLM's `device_map="auto"` client specifically requires `gc.collect()` before
`torch.cuda.empty_cache()`; without it the ~16 GiB model survives `unload()` inside cyclic
Python references until an opportunistic GC, racing the next load into a CUDA OOM (see
docs/phase14-results.md). Phase 22 (ADR 0023) builds the runtime VLM as ONE instance per
GPU instead of one sharded model: the default `qwen3-vl-4b` (fp16) and the int8
`qwen3-vl-8b-int8` each load with `device_map={"": "cuda:N"}` per card and the description
stage runs them as a worker pool (panels split between the cards). The 8B fp16 sharded
client (`qwen3-vl-8b`) remains for comparison.

The production client factory accepts only candidates with an implemented adapter. Entries
in `configs/benchmark_candidates.yaml` without an adapter remain research candidates and
must not be reported as if they were active runtime models.

## Safety Contracts

- A plan contains at most one `PRIMARY` object.
- PRIMARY failure rejects the run. SECONDARY/MICRO failure drops only that object.
- On the generative path there is no segmentation: the animation unit is the accepted DINO
  bbox crop, clamped to the panel crop bounds. A crop that degenerates after clamping is
  skipped; a panel with no accepted candidate is REJECTED.
- Segmentation masks are full-source-image, 2D and `uint8` (deterministic path only).
- Cross-object mask overlap may drop a secondary object to prevent duplicate silhouettes
  (deterministic path only).
- Raw composited frames preserve pixels outside transformed masks exactly (deterministic
  path only).
- H.264 decoding may introduce bounded codec noise; decoded validation is separate from the
  raw-frame pixel invariant.
- `panel_bbox` describes logical panel geometry; `scene_crop_bbox` is the bounded processing and
  output canvas. Scene crops contain their logical panel, remain in page bounds, and stop at
  nearby-panel midpoints where a gutter exists.

## STATIC Results

Semantic analysis may correctly find no justified motion. That is a valid analysis result,
not evidence that the VLM failed. The current render pipeline still rejects an all-STATIC
plan because its output contract requires an animated target. If static-video output becomes
a product requirement, it should be added as an explicit pipeline outcome rather than
silently treating the current analysis failure as success.

## Plan Boundary

The Animation Plan is generated before grounding and segmentation. It records what should
move and why; downstream stages determine where the object is and how pixels are transformed.
`parent_id`/`children_ids` are validated structurally, but automatic transform inheritance
is not implemented. On the generative path each accepted object is animated independently
from its own DINO bbox crop and Qwen description prompt.
