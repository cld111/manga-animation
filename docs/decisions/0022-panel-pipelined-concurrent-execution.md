# ADR 0022: Panel-Pipelined Concurrent Stage Execution

Status: Accepted

Depends on: [ADR 0021](0021-run-level-model-co-residency.md) -- the pipeline workers run on
top of the run-level residency, one client per worker, never shared across threads.

## Context

ADR 0021 made all models co-resident for the whole run, but the stages still executed
sequentially with a full barrier between them: grounding processed EVERY panel of EVERY page,
then object description processed every panel, then segmentation, then reconstruction.
Phase 18.4's real 2xT4 run (docs/phase18.4-results.md) showed the resulting idle time: DINO
finishes in ~14 s and SAM in ~17 s while Qwen takes ~838 s, yet the pipeline made each later
stage wait for the full earlier one. With all four models resident (ADR 0021) there is no
memory reason for a stage to wait -- only the code structure.

## Decision

Replace the stage barriers in `run_pages` with a concurrent panel pipeline:

- Five single-threaded workers, one per stage: grounding (DINO), object description (Qwen),
  segmentation (SAM), plan/animate/reconstruct (LaMa), render (CPU). Workers are Python
  threads connected by bounded `queue.Queue`s (maxsize 8), giving backpressure.
- A panel ("token") moves to the next stage as soon as the previous stage produced ITS
  result. DINO may still be grounding panel 3 while Qwen describes panel 1 and SAM
  segments panel 2 -- no stage waits for all panels of all pages.
- Determinism is preserved by construction: tokens enter the pipeline in fixed page/panel
  order, each worker is single-threaded and processes its queue FIFO, and each panel's
  per-stage outputs are computed by the same code paths as the sequential scheme -- so the
  per-panel results are byte-identical, only the scheduling differs.
- Shutdown is cascaded: a `None` sentinel is propagated downstream, and a worker that dies
  (unexpected exception) still forwards the sentinel in `finally`, so no worker can
  deadlock waiting on a queue whose producer died. Worker-level failures fail the run
  (fail closed); panel-level failures still isolate to the panel (REJECTED/ERROR), as in
  the sequential scheme.
- Checkpoints (Phase 18.4 persistence) are now written PER PANEL right after that panel
  completes a stage, and resume is per-panel: a panel absent from a stage's checkpoint
  re-runs exactly that stage (`_panel_start_stage`). Models are loaded only when at least
  one panel needs their stage.
- A panel whose page-manifest entry is already PASS/STATIC (video exists / static crop)
  is not re-entered into the pipeline at all -- completed panels are reused, not
  re-rendered.

## Consequences

- Positive: Qwen (the ~838 s bottleneck) starts on panel 1 while DINO finishes the rest of
  the batch; SAM/LaMa/render of panel N overlap Qwen of panel N+1. Wall-clock no longer
  contains per-stage barrier idle time. Resume granularity improves from page-level to
  panel-level. PASS/STATIC panels are skipped entirely (previously they were re-rendered
  on every invocation).
- The pipeline is single-process threaded; a model client is used by exactly one worker,
  so no cross-thread model access occurs. CUDA work of different models can overlap
  naturally where the cards allow.
- Deterministic order of panel completion is not guaranteed (renders can complete out of
  order), but the final manifest is written per-panel as renders finish and a final write
  happens at the end, so the artifact set is deterministic.
- Bounded queues cap the memory of in-flight panels (each holds its crop and masks).

## Evidence

- tests/test_lifecycle.py `test_run_pages_pipelines_panels_without_stage_barriers`: DINO's
  worker is gated so its second panel waits for Qwen's first `generate()`; the run still
  completes, proving Qwen started while DINO was still working (a barrier would deadlock).
- tests/test_lifecycle.py existing lifecycle/resume suite passes unchanged (modulo the
  per-panel resume semantics now being finer than the old per-page semantics).
