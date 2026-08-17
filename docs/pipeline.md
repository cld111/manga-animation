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
    │  PRIMARY non-PASS rejects the run, SECONDARY/MICRO drops the object. Runs BEFORE
    │  segmentation (Phase 18.4 ordering: DINO -> Qwen -> SAM).
    ▼
Precise segmentation            — src/manga_animation/segmentation
    │  SAM2, run ONLY on the bboxes the description stage accepted (Phase 18.4);
    │  full-source-image 2D uint8 mask for each accepted object
    ▼
Post-segmentation safety gates  — segmentation/orchestration boundary
    │  mask shape and cross-object overlap checks
    ▼
Animation engine (config-selected) — src/manga_animation/wan2 OR
    │  animation        • Wan2.2-TI2V-5B (ADR 0024, model_variants.animation):
    │                     original panel image + prompt built from the accepted Qwen
    │                     descriptions -> generated frame sequence via I2V mode
    │                     (isolated subprocess, pinned model env).
    │                   • Deterministic CV (the original engine): apply MotionSpec
    │                     transforms locally to per-object masks.
    ▼
Layer decomposition             — pipeline/types.py
    │  collect per-object frames and deterministic z-order (deterministic engine only)
    ▼
Hidden-region reconstruction     — src/manga_animation/reconstruction
    │  fill only background revealed by an object's motion (deterministic engine only)
    ▼
Original-image compositing       — src/manga_animation/compositing
    │  blend layers onto a fresh copy of the original plate (deterministic engine only)
    ▼
Decoded-output validation        — src/manga_animation/rendering
    │  verify dimensions, timing, decoded frame count and loop metrics
    ▼
H.264 video                      — src/manga_animation/rendering
```

`run_page_panels` is the page-level production entry point. It does not duplicate the stage
implementation: it writes each scene crop and invokes the existing `run_pipeline` once per
panel. The scene crop, not the strict logical `panel_bbox`, is the source image for grounding,
segmentation, CV transforms, reconstruction, compositing and video rendering. Page-space
coordinates are recovered for cross-panel safety checks by adding the scene crop origin.

`run_pages` is the batch entry point (Phase 18.4): it processes MANY pages in one call.
Every model stage persists its outputs to disk (`grounding.json`, `descriptions.json`,
`segmentation.json` + mask `.npz` per page): a later invocation loads the completed stages
from disk and never re-loads their models, so a killed session resumes from the last
completed stage instead of re-running DINO/Qwen/SAM. `run_page_panels` is the single-page
convenience wrapper over the same code path.

Since Phase 21 the stage execution is a CONCURRENT PANEL PIPELINE (ADR 0022): five
single-threaded workers (grounding -> object description -> segmentation ->
plan/animate/reconstruct -> render) pass panels through bounded queues. A panel moves to the
next model as soon as the previous stage produced ITS result -- there is no stage barrier
that waits for every panel of every page. Per-panel results are byte-identical to the
sequential scheme: tokens enter the pipeline in fixed page/panel order and each worker
processes its queue FIFO. Checkpoints are written per panel, so resume is per-panel too: a
panel absent from a stage's checkpoint re-runs exactly that stage.

Each panel is recorded as `PASS`, `STATIC`, `REJECTED` or `ERROR`. A page manifest is written
after every panel so successful outputs can be reused and a later panel failure cannot erase
earlier results. A materially ambiguous grounded bbox crossing another logical panel is safely
rejected; no object splitting, synchronization or ownership graph is attempted.

The implementation order is intentionally `animation -> reconstruction`: reconstruction
needs transformed masks to know what motion reveals. The layer and safety-gate blocks are
pipeline boundaries, not independent model stages.

## Animation Engine Selection (ADR 0024)

The animation stage is config-selected. `model_variants.animation = wan2.2-ti2v-5b`
registers Wan2.2-TI2V-5B as the generative animation engine; the engine is ACTIVATED by
passing a `Wan2Client` to `run_pages`/`run_page_panels` (it needs the worker-side
checkpoint path and isolated interpreter, which are not buildable from config alone). When
active, stage 3 is the generative engine: `(original panel crop,
prompt from the accepted Qwen descriptions) -> FrameSequence` via I2V mode, rendered directly
by stage 4. No LaMa reconstruction, no per-object CV transforms, and no compositing run on
this path -- the model produces the whole frame sequence. When no `animation_client` is
passed, the deterministic plan/animate/reconstruct + compositing engine runs unchanged.

The generative engine lives in `src/manga_animation/wan2/`; its worker runs in
Wan2.2's own Python environment (diffusers main branch, which may conflict with
the project's `ml` extra -- see docs/decisions/0024). The model's native output is a 121-frame
@ 24 fps clip at 720P, rendered as-is; loop metrics are measured and reported, not guaranteed
seamless by construction.

## Stage Ownership

- `grounding` and `segmentation`: `segmentation-agent`.
- `animation`, `reconstruction`, `compositing` and layer mechanics: `cv-agent`.
- `rendering`: `video-agent`.
- Cross-stage orchestration and evaluation wiring: the orchestrating session and `qa-agent`.

The **per-candidate VLM description** stage (`src/manga_animation/object_description`,
Phase 18.3) is owned by the orchestrating session/`qa-agent` together with `cv-agent`: it
consumes the accepted grounding bbox (the mask stays downstream-only), and its output —
the mapped `MotionSpec` — is what the animation stage applies. It runs before segmentation
(Phase 18.4): SAM segments only the bboxes that earned an action description.

## Model Lifecycle

Model residency is run-level and explicitly owned (Phase 20, ADR 0021, superseding ADR 0020's
stage-level scheme). `run_pages` loads ALL model clients that have pending work TOGETHER at the
start of the call -- grounding (DINO), object description (Qwen3-VL), segmentation (SAM 2.1)
and reconstruction (LaMa) co-reside in GPU memory for the whole run, and each stage processes
every eligible panel of every page in turn. Release happens exactly once, deterministically,
at the end: every model-backed stage runs inside a `ModelStage` context manager
(`src/manga_animation/pipeline/lifecycle.py`, composed via `ExitStack`) that loads the client
on entry and releases it on exit -- on success AND on exception -- by dropping references,
collecting cyclic garbage, and flushing the CUDA caching allocator. The stages themselves
execute concurrently as a panel pipeline (Phase 21, ADR 0022) on top of this residency:
each worker calls only its own client, so the co-resident models are never shared between
threads. Resume is preserved: a stage whose checkpoint exists on disk is restored from it and
its model is never loaded, so a killed session resumes from the last completed stage.

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
- Segmentation masks are full-source-image, 2D and `uint8`.
- Cross-object mask overlap may drop a secondary object to prevent duplicate silhouettes.
- Raw composited frames preserve pixels outside transformed masks exactly.
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
is not implemented. Current animation applies each object's own `MotionSpec` independently.
