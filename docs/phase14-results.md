# Phase 14 Results: Stage-Level Model Lifecycle

Phase 14 hardened the panel-first pipeline's GPU memory lifecycle. The work is recorded in
ADR 0020; this file is the evidence record.

## Problem

Sequential panel processing OOM'd around panel 3/4 of a page on a Kaggle Tesla T4 (x2)
worker. The models (Qwen2.5-VL-7B-Instruct, Grounding DINO, SAM 2.1, LaMa) were loaded and
released once per panel, per stage, in the same Python process.

## Investigation

Forensic profiling (`scripts/run_phase14_gpu_mem_profiling.py`) drove the real clients
through the exact per-panel lifecycle (Qwen analysis -> DINO grounding -> Qwen validation ->
SAM segmentation -> Qwen mask_semantics -> LaMa reconstruction) for 4 panels on a real 2xT4
worker, recording `memory_allocated`/`memory_reserved`/peak plus a full-process live-CUDA-
tensor scan at every boundary.

### Root cause (deterministic)

`Qwen25VLClient.unload()` set `self._model = None` and called `torch.cuda.empty_cache()` --
but **not** `gc.collect()`. A `device_map="auto"` transformers model (ADR 0005's Qwen
sharding path) keeps cyclic Python references alive, so its tensors stay allocated and
`empty_cache()` has nothing to release. The ~16 GiB instance was only reclaimed when an
opportunistic cyclic GC happened to run during the NEXT model's `from_pretrained`, racing
that load into a CUDA OOM.

Direct measurement on the GPU worker:

```
after load:                              g0=13.23GiB g1=14.04GiB   (Qwen resident)
model = None; empty_cache()              g0=13.23GiB g1=14.04GiB   (NOTHING released)
gc.collect(); empty_cache()              g0= 0.00GiB g1= 0.00GiB   (fully released)
```

Grounding DINO (0.9 GiB) and SAM 2.1 (0.3 GiB) release without `gc.collect()` (they use
`.to(device)`, no cycles); only the `device_map="auto"` VLM leaks.

### Baseline reproduction (old per-panel lifecycle)

`--current-unload` profiling (the real old path: drop ref + `empty_cache()`, no gc):

```
panel1 Qwen unload      g0=7228MB g1=8605MB live=733 tensors / 15816MB   (STILL RESIDENT)
...
panel1 mask_semantics unload   g0=13540MB g1=2382MB live=735 / 15944MB    (STILL RESIDENT)
panel2 Qwen analysis infer -> torch.OutOfMemoryError: CUDA out of memory
```

Panel 1's Qwen instance (and each subsequent one) survived every `unload()`; panel 2's Qwen
load OOM'd. This matches the reported panel 3/4 failure (exact panel depends on timing).

A full-pipeline run of the old code on the same page did **not** OOM that particular time
(58 s): the cyclic GC happened to fire inside the next `from_pretrained` before the OOM. The
failure is a race, which is why it was intermittent -- and why a deterministic fix is required
rather than a retry.

## Fix

- `ModelStage` (`src/manga_animation/pipeline/lifecycle.py`): context manager owning one
  client's residency. Loads on entry; on exit -- success or exception -- drops references,
  `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`.
- `Qwen25VLClient.unload()` now collects garbage before flushing the allocator (the proven
  leak source, hardened regardless of caller).
- `run_page_panels` is stage-level: analysis -> grounding -> validation -> segmentation ->
  mask_semantics -> animation/reconstruction/compositing/rendering, one model load per stage
  per page. Panel failure isolation and manifest resumability preserved.
- Grounding/SAM `load()` idempotent; `reconstruct_hidden_region(managed_loaded=True)` for
  stage-owned LaMa residency.

## Result (fixed lifecycle, real GPU, 4-panel page)

Fixed-unload lifecycle profiling (same driver, 4 panels): every Qwen unload returns the
allocator to ~9-73 MiB/device, `live_tensors <= 2` at every boundary, and all four panels
process with no accumulation:

```
panel1 after Qwen unload  g0=9.1MB   g1=9.1MB   live=0
panel2 after Qwen unload  g0=73.1MB  g1=9.1MB   live=2
panel3 after Qwen unload  g0=73.1MB  g1=9.1MB   live=2
panel4 after Qwen unload  g0=73.1MB  g1=9.1MB   live=2
```

Full E2E (`run_page_panels`, real models) on `villainess_ending_scuffle.png` (720x3086, 4
detected panels), stage-level code:

- Panels varied run to run (VLM nondeterminism): one run `STATIC / STATIC / PASS /
  REJECTED(segmentation)`; a clean full run `STATIC / STATIC / REJECTED(mask_semantics) /
  REJECTED(segmentation)`. The safety gates are the same real gates as before this phase
  (segmentation overlap protection dropped two real secondary objects; the Phase 8.3
  asymmetric-edge check rejected a PRIMARY; the Phase 12 mask_semantics gate rejected a
  PRIMARY). Critically, the mask_semantics-rejected panel stayed REJECTED and was NOT
  rendered into a PASS -- the fail-closed PRIMARY behavior is preserved.
- Memory timeline (5 s samples): Qwen analysis stage holds ~7.3 GiB (g0) + ~8.7 GiB (g1)
  flat across all four panels' analysis, then drops to ~9 MiB on release; DINO (~1 GiB),
  validation Qwen (~7.3/8.7 GiB), SAM (~0.3 GiB), mask_semantics Qwen (~7.3/8.7 GiB), LaMa
  (~0.3 GiB) each follow the same load-once/release pattern. Peak allocated 8.7 GiB on one
  T4 (14.9 GiB usable).
- ModelStage release log: `analysis` released 15816-15817 MiB, `grounding` 892 MiB,
  `validation` 15816-15818 MiB, `segmentation` 281-284 MiB, `mask_semantics` 15816-15818 MiB,
  `reconstruction` 196-197 MiB.
- A later invocation resumed the completed page from its manifest (PASS/STATIC panels were
  reused; only the REJECTED panel was re-processed) -- real-GPU evidence of resumability.

## Performance impact

- Model loads per page: stage-level bounds each model to one load per stage. On a 4-panel
  page Qwen loads 3x total (analysis, validation, mask_semantics) instead of 3x per panel
  (12x). Measured cold Qwen load ~50-56 s; DINO load 7.2 s; SAM load 1.6 s; LaMa load 1.25 s.
- Per-call inference unchanged (the stage functions are the same per-crop code paths).
- Wall-clock on this page varied 58-176 s between identical runs because Qwen generation
  length is nondeterministic (46 s vs ~21 s per analysis call observed) and because different
  runs hit different panel outcomes; the lifecycle change adds no per-call overhead. Clean
  full run: 158.1 s, peak 8.7 GiB.

## Tests

- `tests/test_lifecycle.py` (9 tests): `ModelStage` loads/releases on success and on
  exception; clients without load()/unload(); re-entrancy guard; stage-level loads each model
  exactly once for a whole page; early-panel failure does not poison later panels or cleanup;
  PRIMARY mask_semantics rejection stays REJECTED (not rendered into PASS); unexpected stage
  exceptions isolate to their panel.
- Full suite: 592 passed (2 deselected, `-m slow`). `ruff check .` and `mypy src` clean.

## Adversarial review

An independent review agent found and the fix addressed two HIGH defects in the first
stage-level implementation: (1) a PRIMARY REJECT/ABSTAIN at mask_semantics left the panel's
animation data in the stage dict so the render stage overwrote REJECTED with PASS; (2) stages
only caught `PipelineStageError`, so a raw exception (CUDA OOM, model RuntimeError) aborted
the whole page instead of isolating to its panel. Both are fixed and regression-tested.

## Known limitations

- Wall-clock comparison between the old and new pipeline on this page is dominated by VLM
  nondeterminism; the structural model-load reduction (per-stage vs per-panel) is the certain
  benefit, not a specific wall-clock number.
- The 1xT4 configuration was not exercised (the worker exposed 2xT4); the design (one model
  resident at a time, peak 8.7 GiB) is compatible with a single T4 by construction.
- Panel 3's PRIMARY was PASS on one run and REJECTED (mask_semantics) on a clean re-run of the
  same page: the Phase 12 semantic mask gate's known VLM nondeterminism (recorded in
  current-status.md) is unchanged by this phase and can flip the same content's verdict across
  runs. This phase's job was the memory lifecycle, not the gate's calibration.
