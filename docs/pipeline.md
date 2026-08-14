# Pipeline

This is the current pipeline contract. Historical phase-specific behavior and experiment
results live in `docs/phase*-results.md` and are not normative unless linked here.

```text
Manga page
    │
    ▼
Panel / scene analysis          — src/manga_animation/analysis
    │  detect panels and establish per-panel context
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

## Model Lifecycle

Each model-backed stage owns its memory lifecycle. The VLM is released after analysis and
after target validation; grounding, segmentation and reconstruction release their clients
in `finally` blocks. A benchmark adapter is also unloaded after success or failure.

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
