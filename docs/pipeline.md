# Pipeline (planned)

This describes the pipeline manga-animation will implement in Phase 2 onward. **None of
this is implemented yet** — Phase 1 only establishes the project structure these stages
will live in (`src/manga_animation/<stage>/`) and the schema that connects them (the
Animation Plan). Model choices below (Qwen, SAM, etc.) are illustrative of the *kind* of
model each stage needs, not commitments — Phase 2 exists specifically to benchmark and
select them.

```text
Manga page
    │
    ▼
Panel / scene analysis          — src/manga_animation/analysis
    │  split the page into panels; identify per-panel scene context
    ▼
VLM semantic understanding      — src/manga_animation/analysis
    │  what's happening in each panel: objects, actions, implied motion
    ▼
Structured Animation Plan       — src/manga_animation/schemas
    │  the VLM's reasoning, made machine-readable and validated
    │  (see animation-plan-schema.md) — the contract point of the whole pipeline
    ▼
Object grounding                — src/manga_animation/grounding
    │  map each Animation Plan object to a region of the actual image
    ▼
Precise segmentation            — src/manga_animation/segmentation
    │  pixel-accurate masks per grounded object (e.g. SAM-family model)
    ▼
Layer decomposition             — src/manga_animation/layers
    │  separate each animated object into an independently transformable layer
    ▼
Optional hidden-region          — src/manga_animation/reconstruction
reconstruction                  │  fill in areas a layer's motion will reveal, only where needed
    ▼
Deterministic / kinematic       — src/manga_animation/animation
animation                       │  apply each object's MotionSpec: translate/rotate/scale/
    │                             shear/mesh_warp/opacity, via OpenCV/NumPy transforms
    ▼
Secondary motion                — src/manga_animation/animation
    │  motion that follows from a primary mover (cloth, hair) — SECONDARY/MICRO objects
    ▼
Original-image compositing      — src/manga_animation/compositing
    │  alpha-composite animated layers back over the untouched original page
    ▼
Seamless loop                   — src/manga_animation/rendering
    │  ensure frame N+1 after the last frame matches frame 0
    ▼
H.264 video                     — src/manga_animation/rendering
       encode the frame sequence via FFmpeg
```

`src/manga_animation/pipeline` will hold the orchestration that wires these stages
together end-to-end once enough of them exist to connect (see the phases table in
[README.md](../README.md)).

## Why the Animation Plan sits where it does

The Animation Plan is deliberately generated *before* grounding/segmentation, not after.
Semantic reasoning ("does this need to move, and why") is a VLM/analysis problem; turning
that reasoning into precise pixel regions is a grounding/segmentation problem. Keeping the
plan upstream and pixel-space-free means:

- the plan can be validated, diffed, and unit-tested independently of any image;
- a grounding/segmentation model swap (Phase 2 decision) never requires re-deriving *what*
  should move, only *where* it is in a specific image;
- QA can compare "what was planned" against "what was grounded/animated" as two distinct,
  inspectable artifacts.

## Stage boundaries and GPU residency

Each stage above is expected to own its model's lifecycle: load it, run inference for that
stage across the page, release it. See the "GPU Awareness" principle in
[architecture.md](architecture.md). This is what makes the same pipeline code viable on a
memory-constrained Kaggle T4 as well as a local machine with no discrete GPU at all —
stages that aren't currently running hold no VRAM/unified-memory footprint.

## What decides STATIC vs. animated

That decision is made once, by the VLM semantic understanding stage, and recorded in the
Animation Plan (`MotionType.STATIC` vs. `PRIMARY`/`SECONDARY`/`MICRO`). No downstream stage
re-derives or overrides it — grounding, segmentation, and animation all operate on the
plan's decision, they don't re-litigate it. See "Static Is a Valid Result" in
[architecture.md](architecture.md).
