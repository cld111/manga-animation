# 1. Hybrid VLM + deterministic-CV architecture, not end-to-end generative video

Status: Accepted

## Context

The project goal is to animate a manga page with the minimum motion needed to express the
action already drawn on it, while keeping the original artwork as the source of truth for
every unanimated pixel (see [architecture.md](../architecture.md)). A full
image-to-video generative model (e.g. a diffusion video model conditioned on the page) was
considered and rejected as the primary mechanism.

Generative video models:

- do not guarantee pixel fidelity to the source artwork in unanimated regions — they
  regenerate the frame, they don't composite over it;
- do not give a mechanical guarantee of a seamless loop;
- do not give per-object control over which specific things move, by how much, or why;
- do not degrade gracefully to "nothing moves" — they generally produce *some* motion
  everywhere unless heavily constrained;
- are expensive to run repeatedly and hard to make deterministic/reproducible.

A single VLM pass to plan motion, followed by segmentation and deterministic CV transforms
to execute it, addresses all of the above directly.

## Decision

The pipeline is a **hybrid**: one VLM call for *semantic understanding and planning* (what
should move, and why — captured as the Animation Plan, see
[animation-plan-schema.md](../animation-plan-schema.md)), followed by *deterministic*
grounding, segmentation, and CV-based animation (OpenCV/NumPy affine transforms, mesh
warps, alpha compositing) to execute that plan. See [pipeline.md](../pipeline.md) for the
full stage sequence.

Generative techniques are not banned outright — hidden-region reconstruction (filling an
area a layer's motion reveals) is a case where no amount of transforming existing pixels
can produce the needed content, so a generative inpainting step is planned there
specifically. But it's scoped to that one sub-problem, not used as the pipeline's general
mechanism (see "Deterministic First" in [architecture.md](../architecture.md)).

## Consequences

- The Animation Plan schema becomes the pipeline's central contract and must be designed
  independently of any single model (see "Model Abstraction").
- Motion quality is bounded by what deterministic transforms can express (translation,
  rotation, scale, shear, mesh warp, opacity) — sufficient for the kind of subtle,
  justified motion this project targets, not for arbitrary scene changes.
- The VLM's job is narrowed to planning/classification (STATIC vs. ANIMATED, and motion
  parameters), which is a much more constrained and evaluable task than "generate a video,"
  making Phase 2 model benchmarking tractable.
