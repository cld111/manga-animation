# Architecture

This document defines the engineering principles of the manga-animation pipeline. They
keep replaceable ML components from turning the system into hardcoded glue or a generative
image-to-video system that no longer respects the source artwork.

## Original Image Is the Source of Truth

The manga page the user provides is not a reference for generation - it is the output, minus
motion. Wherever a region of the raw composited frame is not being deliberately animated,
its pixels must be the original pixels, unchanged. After lossy H.264 encoding/decoding,
bounded codec-dependent noise is expected and checked separately. Compositing stages should
make "do nothing" the default per-pixel behavior.

## Minimal Motion Principle

Only objects with a clear, visually justified reason to move should move. The system is not
trying to animate the page; it animates the handful of elements that carry the action. Every
plausible-but-unjustified background motion is out of scope.

## Static Is a Valid Result

A page, object or panel with no justified motion is a valid semantic result. The current
render pipeline still rejects an all-STATIC plan because its output contract requires an
animated target. If static-video output becomes a product requirement, it must be added as
an explicit pipeline outcome rather than silently treated as success.

## Deterministic First

Prefer deterministic CV transforms (affine transforms, mesh warps and alpha compositing in
`src/manga_animation/animation` and `compositing`) over generative video models whenever the
desired motion can be expressed that way. Generative techniques such as inpainting are
reserved for content that motion reveals but that was never present in the source image.

## Local Modification

Every stage should touch the smallest region necessary for its job. Segmentation should not
over-segment beyond what animation needs, warps should be local to the object, and
reconstruction should only fill the specific hole motion reveals. Global operations are
avoided by default.

## Semantic Coordination

Independent objects may have independent timing parameters, but "independent" refers to
timing, not physical meaning. The Animation Plan validates `parent_id`/`children_ids`, but
the current animation implementation does not propagate a parent's transform to children.
Automatic kinematic coordination is future capability and must not be assumed by callers.

## Model Abstraction

VLM, grounding, segmentation and reconstruction are stage-boundary components. The active
candidate is configured through `PipelineConfig.model_variants`; the production factory
must reject candidate IDs for which no client adapter is implemented. Benchmark candidates
without adapters remain research entries, not fake runtime substitutions.

## GPU Awareness

Models must not remain loaded in VRAM/unified memory longer than the stage that needs them.
The VLM is used by both analysis and target validation, but each stage releases it before the
next stage. Grounding, segmentation, reconstruction and benchmark adapters also clean up on
failure paths.

## Remote Compute Is Disposable

See [`decisions/0002-local-canonical-source.md`](decisions/0002-local-canonical-source.md)
and [`decisions/0003-remote-compute-workers.md`](decisions/0003-remote-compute-workers.md).
Kaggle/Jupyter sessions can disappear without warning. Code, tests, config and docs live in
the local git checkout; generated artifacts remain reproducible and git-ignored.
