# ADR 0024: AnimateAnything as the Generative Animation Engine

- **Status**: accepted (implemented, integration run pending)
- **Date**: 2026-08-17
- **Context**: the user selected `alibaba/animate-anything` as THE animation model for the
  project ("для анимации должна использоваться только эта модель"), overriding the
  deterministic-first default for the animation stage.

## Decision

Add a generative animation engine to the pipeline, selected by
`model_variants.animation = animate-anything-512-v1.02` and activated by passing an
`AnimateAnythingClient` to `run_pages`/`run_page_panels`. When active, the animation stage
consumes exactly the integration contract the user specified:

1. the ORIGINAL panel image (the scene crop, unchanged),
2. the SAM segmentation masks of the accepted objects (merged into one binary motion mask
   marking the region allowed to move),
3. the accepted Qwen object descriptions (composed into one text prompt, PRIMARY first).

AnimateAnything then generates the panel's frame sequence directly from `(image, mask,
prompt)`. The deterministic plan/animate/reconstruct engine and its LaMa reconstruction /
compositing are NOT used on this path: the generative model produces the whole frame
sequence, so there are no per-object layers, no motion-revealed holes to reconstruct, and no
CV compositing.

## Architecture

- `src/manga_animation/animation_anything/` — the engine:
  - `client.py` `AnimateAnythingClient` — the pipeline-facing model client (subprocess-backed).
  - `worker.py` — the isolated inference entrypoint that loads the checkpoint and generates
    frames; runs under AnimateAnything's OWN Python environment.
  - `prompt.py` — deterministic prompt builder from `ObjectDescriptionResult`s.
  - `mask.py` — union merger of accepted SAM masks into one 0/255 motion mask.
  - `spec.py` — the serialized (image, mask, prompt, hyper-parameters) contract between client
    and worker.
  - `vendored/` — the upstream repository's inference subset (Apache-2.0), imported ONLY by
    `worker.py`.
- Pipeline wiring in `pipeline/panels.py`: stage 3 becomes the generative engine when an
  `animation_client` is passed; stage 4 renders the generated `FrameSequence` directly.
  Fail-closed semantics are unchanged (`_build_plan` rejects a no-acceptance panel, unexpected
  failures isolate to their panel).
- Config: `model_variants.animation` + `animation_*` hyper-parameters in `default.yaml`;
  the candidate is registered in `benchmark_candidates.yaml` and `_RUNTIME_CANDIDATES`.

## Dependency isolation (why a subprocess)

AnimateAnything was released against an old stack (`diffusers==0.24.0`,
`transformers==4.36.2`, `torch==2.0.0` — its own `requirements.txt`), which cannot coexist
with the project's `ml` extra (`transformers>=5.0`, `torch>=2.3`). The worker therefore runs
in a dedicated Python environment on the remote worker; the client shells out per panel. The
heavy model and its pinned environment are remote-GPU concerns (ADR 0003), not local
dependencies.

## Output contract

The model's native output is a 16-frame @ 8 fps clip (~2 s), which the pipeline renders
as-is (H.264). This differs from the deterministic engine's 4 s @ 24 fps seamless-loop
default; the generative clip is rendered and its loop metrics are measured and reported, not
guaranteed seamless by construction.

## Consequences

- The deterministic animation engine remains available (regression path); the generative
  engine is the selected animation model for the integration run.
- AnimateAnything is added to the runtime candidate registry; no production client is built
  for it unless the caller passes an `AnimateAnythingClient` (which requires the worker env +
  checkpoint paths).
- Real inference is remote-GPU work only; local tests use a fake AA client.
