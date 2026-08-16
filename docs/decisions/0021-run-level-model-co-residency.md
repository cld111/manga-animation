# ADR 0021: Run-Level Model Co-Residency

Status: Accepted

Supersedes: the "one model family resident at a time" rule of
[ADR 0020](0020-stage-level-model-lifecycle.md) -- `ModelStage`'s deterministic release
mechanism itself is preserved unchanged.

## Context

ADR 0020 made model residency stage-scoped: each model-backed stage loaded its client,
processed every eligible panel, then deterministically released it before the next stage
loaded its own model. This was the right call when a single T4 (14.56 GiB usable) had to
hold each model alone -- Qwen2.5-VL's ~16 GiB of float16 weights could not co-exist with
anything else.

Phase 18.4's real GPU run (2xT4 Kaggle worker, docs/phase18.4-results.md) showed the
sequential scheme's cost: the object-description stage (Qwen) dominates wall-clock (838 s of
1229 s), and each stage boundary spends time loading and releasing models while the other
GPU sits idle. Meanwhile Qwen3-VL-8B (Phase 20 VLM swap) is the same ~16 GiB float16 class
as 2.5-VL-7B and shards across the session's 2xT4 via `device_map="auto"` (ADR 0005 path),
leaving per-GPU headroom: Qwen ~8.3 GiB per card + DINO 1.8 GiB + SAM 2.1 0.6 GiB + LaMa
~0.2 GiB still fits the 15 GiB budget of each T4. Keeping all four families resident for the
whole run is now feasible and removes every load/unload boundary (and its risk window) from
the hot path.

## Decision

- `run_pages` loads ALL model clients that have pending work TOGETHER at the start of the
  call and keeps them resident until the whole run finishes: grounding (DINO), object
  description (Qwen3-VL), segmentation (SAM 2.1), reconstruction (LaMa).
- Release happens exactly once, at the end, via the existing `ModelStage` context managers
  composed through `contextlib.ExitStack` (entered in stage order, unwound LIFO). Exception
  safety is unchanged: a failed run still deterministically releases every loaded model.
- Resume is preserved: a stage restored from its disk checkpoint (Phase 18.4 persistence)
  does not load its model at all -- the co-residency set is exactly the models with pending
  work, determined before anything loads.
- The VLM client (Qwen3VLClient) gains an explicit idempotent `load()`, so the run-level
  `ModelStage` brings it up together with DINO/SAM/LaMa instead of lazily inside the first
  `generate()` call.
- `ModelStage` (src/manga_animation/pipeline/lifecycle.py) is unchanged: its
  load-on-entry/unload-on-exit ownership, `gc.collect()` -> `empty_cache()` -> `ipc_collect()`
  release order, and re-entrancy guard all apply to the run-level scope as written.

## Consequences

- Positive: no load/unload cycle between stages (Qwen stays warm between panels); one
  explicit, deterministic release at the end instead of four mid-run ones; the panel runner
  no longer has four nested failure modes around stage boundaries; Qwen3-VL can prefill
  during earlier stages' CPU work.
- The 2xT4 memory budget is now the run's peak budget: every model must co-exist
  (verified feasible for 8B-class Qwen + DINO + SAM 2.1 + LaMa, see Context). A model
  larger than that (e.g. Qwen3-VL-32B) must either go back to a smaller VLM or to
  stage-level residency; the config-driven client factory keeps that a runtime choice.
- The per-stage `_release_memory_log` now logs once per run per model, not per stage.
- One model load per run per model instead of one per stage: a full-run Qwen load was
  already once; the win is removing the other three families' unload/load churn.

## Evidence

- docs/phase18.4-results.md: real 2xT4 timings and GPU samples showing stage-boundary
  churn and the per-model memory footprints the co-residency budget is derived from.
- tests/test_lifecycle.py `test_run_pages_co_residency_loads_all_models_up_front_and_unloads_at_the_end`:
  all models load before the first stage call, none unload until the last stage call,
  LIFO unwind, exact one load/unload per model.
