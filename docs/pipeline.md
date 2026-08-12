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
    │  map each Animation Plan object to a region of the actual image; grounds against the
    │  object's real panel crop when one is known, else the full page (Phase 5.1, see
    │  docs/decisions/0011-panel-aware-grounding.md) — always returns full-page coordinates
    ▼
Target validation                — src/manga_animation/validation
    │  ACCEPT/REJECT: does the grounded region actually depict the intended target?
    │  (Phase 3.2 — see docs/decisions/0006-grounding-target-validation.md; a
    │  technically valid detection is not automatically a semantically correct one)
    ▼
Precise segmentation            — src/manga_animation/segmentation
    │  pixel-accurate masks per grounded object (e.g. SAM-family model)
    ▼
Deterministic / kinematic       — src/manga_animation/animation
animation                       │  apply each animated object's MotionSpec: translate/rotate/
    │                             scale/shear/mesh_warp/opacity, via OpenCV/NumPy transforms.
    │                             Runs for every non-STATIC object in the plan, not only the
    │                             PRIMARY one — a real PRIMARY + SECONDARY/MICRO plan animates
    │                             all of them (Phase 4, see
    │                             docs/decisions/0010-multi-object-layer-decomposition.md).
    ▼
Layer decomposition             — src/manga_animation/pipeline/types.py (the `Layer` type)
    │  wrap each animated object's per-frame (image, mask) output into a `Layer` with a
    │  compositing z_order — thinner than a standalone stage; see
    │  src/manga_animation/layers/__init__.py for why it isn't its own module
    ▼
Optional hidden-region          — src/manga_animation/reconstruction
reconstruction                  │  fill in areas a layer's motion will reveal, only where
    │                             needed — computed per object, after animation (not before —
    │                             it needs to know what each object's motion actually reveals)
    ▼
Original-image compositing      — src/manga_animation/compositing
    │  alpha-composite every animated layer back over the untouched original page, in
    │  deterministic z_order (Phase 4: N layers via `composite_frame_stack`, not just one)
    ▼
Seamless loop                   — src/manga_animation/rendering
    │  ensure frame N+1 after the last frame matches frame 0
    ▼
H.264 video                     — src/manga_animation/rendering
       encode the frame sequence via FFmpeg
```

`src/manga_animation/pipeline` holds the orchestration that wires these stages together
end-to-end (`pipeline/orchestrator.py::run_pipeline`), plus the shared cross-stage type
contracts every stage converts to/from at its boundary (`pipeline/types.py`, including the
`Layer` type — see below).

## Stage ownership

Most stages above map to one specialist agent in `.claude/agents/`, matched to the `src/`
package each owns (`vision-agent` → `analysis`/`schemas`; `segmentation-agent` →
`grounding`/`segmentation`; `animation-agent` → motion parameters on top of the schema;
`video-agent` → `rendering`; `qa-agent` → cross-cutting checks in `tests/`). The
**hidden-region reconstruction** stage (`src/manga_animation/reconstruction`) is owned by
`cv-agent`, alongside the deterministic animation (`animation`) and compositing
(`compositing`) stages it already implements — it is not a separate specialist, since
reconstruction exists specifically to prepare pixels that `cv-agent`'s own compositing
step then consumes. See `.claude/agents/cv-agent.md` for the full ownership boundary
(input/output/downstream consumer). **Layer decomposition** (Phase 4, `pipeline.types.Layer`
plus `compositing.composite_frame_stack`) follows the same ownership: it's `cv-agent`'s
territory too, not a separate specialist — see
[ADR 0010](decisions/0010-multi-object-layer-decomposition.md). Multi-object orchestration
itself (looping grounding/validation/segmentation over every non-STATIC object, not just
PRIMARY) lives in `pipeline/orchestrator.py`, owned by the orchestrating session per this
project's general pipeline-wiring convention, not any single stage specialist.

The **target validation** stage (`src/manga_animation/validation`, Phase 3.2 — see
[ADR 0006](decisions/0006-grounding-target-validation.md)) is owned by `segmentation-agent`:
it sits structurally between `grounding` and `segmentation` (both already that agent's
packages), and its job — "is this specific candidate region a plausible match for the
target" — is a grounding-quality question, not a "should this object move at all" one (that
remains `vision-agent`'s call, made once, upstream, and not re-litigated here). Mechanically,
though, it calls into the VLM through the same `VLMClient` protocol `analysis/client.py`
already defines (a cheap crop-verification call, not a full-page analysis call) rather than
adding a new model dependency — so a validation-quality question is fair game for
`vision-agent` to weigh in on too, even though `segmentation-agent` owns the code.

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
