# ADR 0024: Wan2.2-TI2V-5B as the Generative Animation Engine

- **Status**: accepted (implemented, integration run pending)
- **Date**: 2026-08-17 (revised 2026-08-17: replaced AnimateAnything with Wan2.2)
- **Context**: the user selected `Wan-AI/Wan2.2-TI2V-5B` as THE animation model for the
  project ("полностью замени AA на Wan2.2-TI2V-5B"), replacing the earlier AnimateAnything
  integration.

## Decision

Add a generative animation engine to the pipeline, selected by
`model_variants.animation = wan2.2-ti2v-5b` and activated by passing a `Wan2Client` to
`run_pages`/`run_page_panels`. When active, the animation stage consumes:

1. the ORIGINAL panel image (the scene crop, unchanged),
2. the accepted Qwen object descriptions (composed into one text prompt, PRIMARY first).

Wan2.2-TI2V-5B generates the panel's frame sequence directly from `(image, prompt)` using
I2V (Image-to-Video) mode. Unlike AnimateAnything, there is no motion mask input -- the
model generates the entire video conditioned on the input image and text prompt.

The deterministic plan/animate/reconstruct engine and its LaMa reconstruction / compositing
are NOT used on this path: the generative model produces the whole frame sequence, so there
are no per-object layers, no motion-revealed holes to reconstruct, and no CV compositing.

## Architecture

- `src/manga_animation/wan2/` — the engine:
  - `client.py` `Wan2Client` — the pipeline-facing model client (subprocess-backed).
  - `worker.py` — the isolated inference entrypoint that loads the checkpoint and generates
    frames; runs under Wan2.2's OWN Python environment.
  - `prompt.py` — deterministic prompt builder from `ObjectDescriptionResult`s.
  - `mask.py` — union merger of accepted SAM masks into one 0/255 mask (kept for provenance;
    Wan2.2 does not use it for generation).
  - `spec.py` — the serialized (image, prompt, hyper-parameters) contract between client
    and worker.
- Pipeline wiring in `pipeline/panels.py`: stage 3 becomes the generative engine when an
  `animation_client` is passed; stage 4 renders the generated `FrameSequence` directly.
  Fail-closed semantics are unchanged (`_build_plan` rejects a no-acceptance panel, unexpected
  failures isolate to their panel).
- Config: `model_variants.animation` + `animation_*` hyper-parameters in `default.yaml`;
  the candidate is registered in `orchestrator.py`.

## Dependency isolation (why a subprocess)

Wan2.2-TI2V-5B requires diffusers main branch (not the PyPI release) and specific
torch/transformers versions. The worker therefore runs in a dedicated Python environment on
the remote worker; the client shells out per panel. The heavy model and its pinned
environment are remote-GPU concerns (ADR 0003), not local dependencies.

## Output contract

The model's native output is a 121-frame @ 24 fps clip (~5s) at 720P, which the pipeline
renders as-is (H.264). This is significantly higher quality than the old AnimateAnything
output (16 frames @ 8 fps = 2s at 512x512).

## Key differences from AnimateAnything

| Aspect | AnimateAnything | Wan2.2-TI2V-5B |
|---|---|---|
| Parameters | ~1B | 5B |
| Architecture | UNet DDPM | Diffusion Transformer (DiT) |
| VAE | CogVideoX (8×8×4) | Wan2.2 (16×16×4, ratio=64) |
| Input | T2V only (text) | T2V + I2V (text + image) |
| Mask | Required (motion mask) | Not used |
| Resolution | 512×512 @ 8fps | 720P @ 24fps |
| Frames | 16 (2s) | 121 (5s) |
| Peak VRAM | ~15 GB | ~23 GB (4090) |
| Multi-GPU | Single instance | FSDP + Ulysses |
| Diffusers | vendored (0.24.0) | main branch |

## Consequences

- The deterministic animation engine remains available (regression path); the generative
  engine is the selected animation model for the integration run.
- Wan2.2-TI2V-5B is added to the runtime candidate registry; no production client is built
  for it unless the caller passes a `Wan2Client` (which requires the worker env +
  checkpoint paths).
- Real inference is remote-GPU work only; local tests use a fake Wan2 client.
- The old AnimateAnything package (`src/manga_animation/animation_anything/`) is retained
  for backward compatibility but is no longer the default engine.
