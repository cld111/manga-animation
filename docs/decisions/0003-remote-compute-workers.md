# 3. Remote Kaggle/Jupyter GPU is a disposable compute worker, config-selected

Status: Accepted

## Context

Environment inspection at project start found: macOS (arm64), Apple M1 Max, no NVIDIA GPU,
no CUDA, no `nvidia-smi`. Apple Silicon exposes GPU compute through Metal/MPS, not CUDA.
Most of the models this project will plausibly use (VLMs, SAM-family segmentation,
inpainting) are developed and best-supported against CUDA, and the larger variants may not
comfortably fit in this machine's memory even under MPS. Kaggle (or another Jupyter GPU
host) is therefore expected to be needed for real model work — T4 or L4, both CUDA, both
with materially less VRAM than convenient for large models.

That means the pipeline has to run correctly against at least three different hardware
profiles (local CPU/MPS, Kaggle T4, Kaggle L4) without becoming three different codebases.

## Decision

Remote GPU sessions are ephemeral compute workers, never the canonical project (see
[0002](0002-local-canonical-source.md)), and hardware differences between them are
resolved entirely through `PipelineConfig` (`src/manga_animation/core/config.py`) plus
layered YAML profiles in `configs/` (`default.yaml`, `local.yaml`, `kaggle.yaml`) —
never through code branching on "am I local or remote" or hardcoded device/dtype values
inside a pipeline stage.

Concretely:

- `device: auto` resolves to `cuda` > `mps` > `cpu` at runtime (`PipelineConfig.resolve_device()`),
  so the same default config is safe to check out on any of the three profiles.
- Anything that must actually differ per environment (`dtype: float16` on Kaggle vs.
  `float32` locally where MPS op support is inconsistent; `resolution`) lives in the
  environment's YAML profile, loaded via
  `load_config(env="kaggle")` / `load_config(env="local")`.
- If a task requires reaching an actual Kaggle/Jupyter server, the URL is requested from
  the user explicitly, every time — never invented, never assumed carried over from a prior
  session, since these sessions are expected to expire or rotate.

## Consequences

- Adding a new hardware profile (e.g. a different GPU tier, or a second local machine) is a
  new small YAML file, not a code change.
- Pipeline stage code must not import `torch.cuda` (or similar) directly to make decisions
  — it reads the resolved config. This is a real constraint on Phase 2+ implementation, not
  just a Phase 1 nicety.
- Losing access to a Kaggle session mid-task is an expected, handled situation (report
  unavailability, fall back to local work where possible, ask for a fresh URL when GPU work
  must resume) rather than an exceptional failure.

## Addendum (Phase 3.1): interactive browser access is not the intended transport

Phase 3.1's first real end-to-end run (see [`docs/phase3-results.md`](../phase3-results.md))
reached the remote Kaggle session via interactive browser automation (Claude Code's
`claude-in-chrome` integration driving the JupyterLab UI directly), because no programmatic
transport existed yet. This worked for a one-off run but is explicitly **not** how this
project's compute split is meant to work day to day: it depends on a live, manually-provided
URL and an interactive UI session, neither of which is scriptable, reproducible, or usable by
an unattended/background agent. Per explicit user direction, `claude-in-chrome` has been
disabled for this project (denied in `.claude/settings.local.json`'s `permissions.deny`) and
must not be relied on for the normal pipeline/benchmark execution path going forward. A
programmatic transport (Jupyter REST/kernel API, SSH, or another mechanism, to be determined
empirically against whatever the remote environment actually supports — not assumed) is
required before further real-GPU runs happen without manual, interactive access. This is
scoped, real follow-up work, deliberately not built as part of Phase 3.1.
