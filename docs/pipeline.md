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
Panel / scene analysis          — src/manga_animation/analysis
    │  establish per-panel semantic context
    ▼
VLM semantic understanding      — src/manga_animation/analysis
    │  identify objects, action cues and justified motion
    ▼
Structured Animation Plan       — src/manga_animation/schemas
    │  validated semantic contract; no pixel-space data
    ▼
Object grounding                — src/manga_animation/grounding
    │  panel crop when available, otherwise full page; returns page coordinates
    ▼
Target validation               — src/manga_animation/validation
    │  bbox plausibility, semantic agreement and transform geometry
    ▼
Precise segmentation            — src/manga_animation/segmentation
    │  full-source-image 2D uint8 mask for each accepted object
    ▼
Post-segmentation safety gates  — segmentation/orchestration boundary
    │  mask shape and cross-object overlap checks
    ▼
Semantic mask validation         — src/manga_animation/validation (mask_semantics.py)
    │  ACCEPT/REJECT/ABSTAIN: does the real mask's own pixel content match the intended
    │  target, not just "is this box a plausible location for it" (Phase 12 — see
    │  docs/decisions/0018-semantic-mask-validation.md; a geometrically unremarkable mask
    │  is not automatically a semantically correct one)
    ▼
Per-candidate VLM description    — src/manga_animation/object_description
    │  Phase 18.3: for every still-animated object, the VLM sees the FULL pipeline image
    │  plus the accepted grounding bbox as pixel coordinates (never a crop of the
    │  candidate, never the mask), judges the candidate itself (pass/ambiguous/partial/
    │  reject/not_animatable) and produces a structured animation description whose
    │  deterministically-mapped MotionSpec drives the animation stage. Fail-closed:
    │  PRIMARY non-PASS rejects the run, SECONDARY/MICRO drops the object.
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

Each panel is recorded as `PASS`, `STATIC`, `REJECTED` or `ERROR`. A page manifest is written
after every panel so successful outputs can be reused and a later panel failure cannot erase
earlier results. A materially ambiguous grounded bbox crossing another logical panel is safely
rejected; no object splitting, synchronization or ownership graph is attempted.

The implementation order is intentionally `animation -> reconstruction`: reconstruction
needs transformed masks to know what motion reveals. The layer and safety-gate blocks are
pipeline boundaries, not independent model stages.

## Stage Ownership

- `analysis` and the semantic part of `schemas`: `vision-agent`.
- `grounding`, `segmentation` and `validation`: `segmentation-agent`.
- `animation`, `reconstruction`, `compositing` and layer mechanics: `cv-agent`.
- `rendering`: `video-agent`.
- Cross-stage orchestration and evaluation wiring: the orchestrating session and `qa-agent`.

Target validation is owned by `segmentation-agent` because it decides whether a grounded
region is a plausible target before segmentation. It does not change the VLM's STATIC vs.
animated decision.

The **semantic mask validation** stage (`src/manga_animation/validation/mask_semantics.py`,
Phase 12 — see [ADR 0018](decisions/0018-semantic-mask-validation.md)) is owned by
`segmentation-agent` too, for the same reason: it lives in the same `validation` package,
consumes `segmentation`'s own real mask output, and its job ("does the real mask's content
match the label") is a mask-quality question, distinct from `validate_target`'s pre-segmentation
bbox-plausibility question but structurally the same kind of gate.

The **per-candidate VLM description** stage (`src/manga_animation/object_description`,
Phase 18.3) is owned by the orchestrating session/`qa-agent` together with `cv-agent`: it
sits between `mask_semantics` and `animation`, consumes the accepted grounding bbox (the
mask stays downstream-only), and its output — the mapped `MotionSpec` — is what the
animation stage applies.

## Model Lifecycle

Model residency is stage-level and explicitly owned (Phase 14, ADR 0020). Each model-backed
stage runs inside a `ModelStage` context manager (`src/manga_animation/pipeline/lifecycle.py`)
that loads the client on entry and deterministically releases it on exit -- on success AND on
exception -- by dropping references, collecting cyclic garbage, and flushing the CUDA caching
allocator. `run_page_panels` processes panels stage-by-stage: analysis (VLM) for all eligible
panels, then grounding (DINO), then validation (VLM), then segmentation (SAM), then semantic
mask validation (VLM), then object description (VLM, Phase 18.3), then
animation/reconstruction/compositing/rendering (LaMa loaded once).
One model family is resident at a time, never per-panel. A benchmark adapter is also unloaded
after success or failure.

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
