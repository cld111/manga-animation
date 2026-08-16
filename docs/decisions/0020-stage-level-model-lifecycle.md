# ADR 0020: Stage-Level Model Lifecycle

Status: Superseded by [ADR 0021](0021-run-level-model-co-residency.md) for run residency
scope; the `ModelStage` deterministic release mechanism this ADR introduced remains the
production release path.

## Context

The panel-first pipeline processes several panels per page. Each model-backed stage
(analysis, target validation, semantic mask validation, grounding, segmentation,
reconstruction) had its own per-panel load/unload lifecycle, and `run_page_panels` invoked
`run_pipeline` once per panel, so every model was loaded and released once per panel.

On a real Kaggle 2xT4 run the pipeline OOM'd around panel 3/4 of a page. Phase 14's forensic
profiling (docs/phase14-results.md) established the mechanism: `Qwen25VLClient.unload()`
dropped the model reference but only called `torch.cuda.empty_cache()` without
`gc.collect()`. A `device_map="auto"` transformers model (ADR 0005's Qwen2.5-VL sharding path)
holds cyclic Python references, so its ~16 GiB of tensors stayed allocated. The memory was
only reclaimed when an opportunistic cyclic GC happened to run during the NEXT model's
`from_pretrained` -- racing that load into a CUDA OOM. Reproduced deterministically: on panel
1 the Qwen instance survives every `unload()`, and panel 2's Qwen load OOMs.

## Decision

Adopt a stage-level model lifecycle with explicit ownership:

- Each model-backed pipeline stage loads its client once, processes every eligible panel for
  that stage, then deterministically releases the client before the next stage loads its own
  model. The four model families never co-reside.
- Model residency is owned by a `ModelStage` context manager
  (`src/manga_animation/pipeline/lifecycle.py`): it loads the client on entry, and on exit --
  on normal completion AND on exception -- drops the client's references, runs
  `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`. A failed panel
  can no longer leave a model resident.
- `Qwen25VLClient.unload()` itself now collects garbage before flushing the caching
  allocator, so any caller path that releases the VLM is safe, not just `ModelStage`.
- `run_page_panels` restructures to stage loops: analysis -> grounding -> validation ->
  segmentation -> mask_semantics -> animation/reconstruction/compositing/rendering. LaMa is
  loaded once for the whole reconstruction stage (`reconstruct_hidden_region` gained
  `managed_loaded=True` for stage-owned residency). Panel failure isolation, per-stage
  manifest writes, and PASS/STATIC resumability are preserved.
- Grounding DINO and SAM 2.1 `load()` became idempotent for stage re-entry.

## Consequences

- Positive: deterministic memory release (proven on GPU: a Qwen stage now returns the allocator
  to ~9 MiB per device after release); models never co-resident, so a single T4 is sufficient
  for correctness; one model load per stage per page instead of once per panel (Qwen loaded 3x
  per page instead of 3x per panel -- a large wall-clock win on model loads); a failed panel
  cannot poison later panels' lifecycle.
- The stage loops in `run_page_panels` are a real restructure; the per-crop stage functions
  (`_ground_objects`, `_validate_objects`, `_segment_objects`, `_mask_semantics_objects`,
  `_animate_objects`, `_reconstruct_objects`, `_composite_and_render`) are shared with
  `run_pipeline`, keeping the safety gates behavior-identical by construction.
- The panel runner's `runtime_s` metric is now measured from a panel's first stage to when its
  status finalizes, spanning stage boundaries (slightly different semantics from the old
  per-panel wall-clock).
- No process-level cleanup is needed; stage cleanup proved sufficient on the real GPU runs.

## Evidence

- docs/phase14-results.md records the BEFORE (OOM at panel 2, ~16 GiB unreleased) and AFTER
  (4/4 panels process, allocator returns to baseline after every stage, peak 8.7 GiB on one
  T4) GPU measurements.
