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
Deterministic / kinematic       — src/manga_animation/animation
    animation                    │  apply MotionSpec transforms locally
    ▼
Layer decomposition             — pipeline/types.py
    │  collect per-object frames and deterministic z-order
    ▼
Hidden-region reconstruction     — src/manga_animation/reconstruction
    │  fill only background revealed by an object's motion
    ▼
Original-image compositing       — src/manga_animation/compositing
    │  blend layers onto a fresh copy of the original plate
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

`run_pages` is the batch entry point (Phase 18.4): it processes MANY pages with stage-level
model residency ACROSS pages. Each model loads ONCE, processes every eligible panel of EVERY
page (saving its outputs into that page's state), and only then is released and the next
model loads -- never a per-page load/unload cycle. `run_page_panels` is the single-page
convenience wrapper over the same code path.

Each panel is recorded as `PASS`, `STATIC`, `REJECTED` or `ERROR`. A page manifest is written
after every panel so successful outputs can be reused and a later panel failure cannot erase
earlier results. A materially ambiguous grounded bbox crossing another logical panel is safely
rejected; no object splitting, synchronization or ownership graph is attempted.

The implementation order is intentionally `animation -> reconstruction`: reconstruction
needs transformed masks to know what motion reveals. The layer and safety-gate blocks are
pipeline boundaries, not independent model stages.

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

Model residency is stage-level and explicitly owned (Phase 14, ADR 0020). Each model-backed
stage runs inside a `ModelStage` context manager (`src/manga_animation/pipeline/lifecycle.py`)
that loads the client on entry and deterministically releases it on exit -- on success AND on
exception -- by dropping references, collecting cyclic garbage, and flushing the CUDA caching
allocator. `run_pages` (and the single-page wrapper `run_page_panels`) processes panels
stage-by-stage: grounding (DINO) for all eligible panels of ALL pages, then object
description (VLM, the pipeline's single Qwen call), then segmentation (SAM, only for
accepted bboxes), then animation/reconstruction/compositing/rendering (LaMa loaded once).
One model family is resident at a time, never per-panel and never per-page.

The VLM's `device_map="auto"` client specifically requires `gc.collect()` before
`torch.cuda.empty_cache()`; without it the ~16 GiB model survives `unload()` inside cyclic
Python references until an opportunistic GC, racing the next load into a CUDA OOM (see
docs/phase14-results.md).

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
