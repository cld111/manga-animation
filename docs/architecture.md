# Architecture

This document is the set of engineering principles the manga-animation pipeline is built
against. They exist to keep a system with many replaceable ML components (VLM, grounding,
segmentation, inpainting) from turning into either an unmaintainable pile of hardcoded
glue, or a full generative image-to-video system that no longer respects the source
artwork. Every later phase should be checked against these before merging.

## Original Image Is the Source of Truth

The manga page the user provides is not a reference for generation — it *is* the output,
minus motion. Wherever a region of the frame is not being deliberately animated, its pixels
must be the original pixels, unchanged. Compositing stages should be structured so that
"do nothing" is the default per-pixel behavior, and animated regions are the exception
requiring justification (a segmented layer, a transform, a reason).

## Minimal Motion Principle

Only objects with a clear, visually justified reason to move should move. The system is
not trying to animate "the page" — it's trying to animate the handful of elements that
carry the page's action (a swung weapon, blowing hair, an impact). Everything else,
including plausible-but-unjustified motion ("wouldn't it look nice if the background
moved a little"), is out of scope by design.

## Static Is a Valid Result

A page — or an object, or even an entire panel — that ends up with no motion at all is not
a failure state. It's the correct output when there's no visually justified reason to
animate. Every decision point in the pipeline (see the Animation Plan's `STATIC` motion
type in [`animation-plan-schema.md`](animation-plan-schema.md)) should default toward
STATIC and require a positive reason to do otherwise, not the other way around.

## Deterministic First

Prefer deterministic CV transforms (affine transforms, mesh warps, alpha compositing —
see the planned `cv-agent`/`src/manga_animation/animation` and `compositing` stages) over
generative video models wherever the desired motion can be expressed that way. Determinism
gives reproducibility, speed, and — critically — a mechanical guarantee that unrelated
pixels aren't touched. Generative techniques (e.g. inpainting for hidden-region
reconstruction) are reserved for the specific sub-problems that are not expressible as a
transform of existing pixels, such as revealing an area that motion uncovers.

## Local Modification

Every stage should touch the smallest region of the frame necessary for its job:
segmentation should not over-segment beyond what animation needs, warps should be local to
the object being animated, and reconstruction should only fill the specific hole that
motion reveals. Global operations over the whole page are avoided by default.

## Semantic Coordination

Independent objects may have independent motion parameters (phase, speed, amplitude — see
`MotionSpec` in the Animation Plan schema), but "independent" refers to their timing
curves, not their meaning. A character's hand and the object it's holding must stay
kinematically attached (parent/child relationships in the schema); a flag and the pole it
hangs from must move in a way that stays physically plausible together. Coordination is
expressed structurally via `parent_id`/`children_ids`, not left to chance.

## Model Abstraction

VLM, grounding, segmentation, and reconstruction are all treated as swappable components
behind a stage boundary, not as fixed choices baked into pipeline code. Phase 2 will
benchmark and select specific models; nothing about the pipeline's structure (or the
Animation Plan schema, which is model-agnostic) should need to change if a model is later
swapped out. Concretely: model identity lives in `PipelineConfig.model_variants` (a
`dict[str, str]`), not as an import or a hardcoded model name inside a stage module.

## GPU Awareness

Models should not sit loaded in VRAM/unified memory longer than the stage that needs them
is running. This matters especially on the hardware this project actually targets: an
Apple Silicon machine with shared CPU/GPU memory locally, and memory-constrained T4/L4 GPUs
on Kaggle remotely. Stages should load their model, do their work, and release it, rather
than the pipeline holding every model resident for the full run.

## Remote Compute Is Disposable

See [`decisions/0002-local-canonical-source.md`](decisions/0002-local-canonical-source.md)
and [`decisions/0003-remote-compute-workers.md`](decisions/0003-remote-compute-workers.md).
In short: Kaggle/Jupyter GPU sessions can disappear at any time (session expiry, quota,
disconnect) without warning. Nothing about the project's state should depend on a remote
session surviving — code lives in git, and the local checkout is always the complete,
canonical copy.
