# Current Project State

This is the single canonical answer to "what is true about the project right now?" It is
operational state, not a phase report. Historical implementation details and experiment
evidence remain in `docs/phase*-results.md`; decision rationale remains in `docs/decisions/`.

## Status

The deterministic pipeline and local test/evaluation infrastructure are implemented through
Phase 13's panel-first orchestration. Real model execution remains a remote-GPU operation. The
project is an engineering prototype with real end-to-end evidence and known real-world visual
limitations, not a production animation service.

## Current Pipeline

The implemented order is:

```text
page -> deterministic panel detection -> bounded scene crops
  -> independent analysis -> grounding -> validation -> segmentation -> mask_semantics
  -> animation -> reconstruction -> compositing -> rendering
```

- `run_page_panels` is the production page entry point: every detected panel gets a stable unit,
  its own scene crop, independent stages, output video or explicit status, and a page manifest.
- `panel_bbox` is logical geometry; `scene_crop_bbox` is the actual analysis/render canvas and
  is bounded by page edges and nearby panel geometry.
- Analysis is panel-aware by default; page analysis remains explicit. A panel's all-STATIC result
  is recorded as `STATIC` by the panel runner without inventing a video.
- Grounding uses a real panel crop when analysis provides one and returns page coordinates.
- `validation` checks grounded bbox plausibility, semantic agreement, and transform geometry
  before segmentation.
- `segmentation` produces a full-source-image `uint8` mask and applies coverage and asymmetric
  edge-touch safety checks.
- `mask_semantics` checks the real segmented mask's content with a VLM, independently of the
  pre-segmentation bbox check. It returns `ACCEPT`, `REJECT`, or `ABSTAIN` and is enabled by
  default.
- Animation uses deterministic OpenCV/NumPy transforms. Layers are composited in deterministic
  z-order with cross-object overlap protection; LaMa is used only for motion-revealed holes.
- Rendering produces H.264 and validates the decoded output, including frame count, timing,
  dimensions, and loop metrics.

The full stage ownership and lifecycle contract is in [`pipeline.md`](pipeline.md). The
resolution-independent plan contract is in [`animation-plan-schema.md`](animation-plan-schema.md).

## Runtime Defaults

The baseline in `configs/default.yaml` is:

| Setting | Current value |
|---|---|
| Analysis mode default | `run_pipeline(..., analysis_mode="panel")` |
| VLM | `qwen2.5-vl-7b-instruct` |
| Grounding | `grounding-dino-swin-l` |
| Segmentation | `sam2.1-hiera-base` |
| Inpainting | `lama-large` |
| VLM analysis resolution | 1536px long edge (`kaggle`: 2048, `local`: 1024) |
| VLM dtype | profile-dependent (`float32` default, `float16` on Kaggle) |
| Grounding/segmentation dtype | verified `float32` |
| Loop | 4.0s at 24 FPS |
| Codec | H.264 only |
| Semantic mask validation | enabled |

These are preliminary operational selections, not an exhaustive cross-candidate benchmark
conclusion. Candidates without implemented adapters remain research entries.

## Current Invariants

- Raw composited frames copy the original image outside transformed masks exactly, except for
  deliberately filled motion-revealed holes. Decoded H.264 frames may contain bounded codec
  noise; that is validated separately.
- A plan has at most one `PRIMARY`. A PRIMARY failure rejects the run; a SECONDARY/MICRO
  failure is isolated and drops only that object.
- Masks are full-source-image, 2D `uint8` arrays. Cross-object overlap can drop a secondary
  object rather than render a duplicate silhouette.
- Transform-aware validation is intentionally pre-segmentation because it validates a bbox;
  semantic mask validation is post-segmentation because it needs the real mask.
- `parent_id`/`children_ids` are structurally validated, but parent transforms are not
  inherited automatically; each animated object needs its own motion spec.
- Model-backed stages release their clients after the stage, including analysis, target
  validation, semantic mask validation, grounding, segmentation, and reconstruction.
- GPU model lifecycle is stage-level (Phase 14, ADR 0020): each model-backed stage loads its
  client once, processes every eligible panel, then deterministically releases it before the
  next stage loads its own model. Models never co-reside, and a failed panel can no longer
  leave a model resident to poison later panels. The VLM's `device_map="auto"` client is
  released with `gc.collect()` before the caching-allocator flush -- the phase-14 root cause
  of cross-panel CUDA OOM.
- `PipelineConfig.resolution` changes VLM analysis resizing only; downstream CV uses source
  geometry. `dtype` describes the VLM; verified grounding/segmentation clients use `float32`.
- Crossfade frames remain zero, and `h264` is the only supported output codec.
- A semantic all-STATIC result is valid analysis evidence, but the current render contract
  rejects an all-STATIC plan because it has no target to render.
- Ambiguous grounded objects that materially cross into another logical panel are rejected;
  panel processing and unrelated panel outputs continue independently.

## Validated Capabilities and Evidence

- In the Phase 9 10-sample, 8-series real-world baseline, before the Phase 12 semantic mask
  gate, panel mode reached 60% end-to-end completion, 100% grounding success among usable
  targets, 0 ERROR outcomes, and a 14.3% analysis-level semantic false-negative rate on the
  labeled subset. Page mode reached 20% completion and 5 ERROR outcomes. This is historical
  evidence for the panel default, not a current post-gate quality rate.
- Fully automatic real multi-object renders have been observed, including a dense six-object
  panel-mode render. Non-PRIMARY objects can be dropped while a valid PRIMARY render proceeds.
- Transform-aware validation catches bbox/transform mismatches before segmentation. The
  Phase 3.3.1 weapon/panel defect is now rejected rather than rendered.
- The Phase 8.3 protections are active: asymmetric mask validation catches the evidenced
  one-sided over-segmentation pattern, and overlap protection prevents the evidenced duplicate
  silhouette mechanism. Both are geometric, evidence-based, and not statistically calibrated.
- Phase 12's semantic mask gate catches the confirmed `wind_breaker_finish` PRIMARY defect in
  a live end-to-end run before rendering. A dense real page still rendered with the PRIMARY and
  accepted objects; the gate dropped one confirmed-defective `character_hair` secondary and
  also rejected the development benchmark's good-labeled `green_fluid`, exposing a real
  false-rejection trade-off rather than proving every drop correct.
- Real-world evaluation provenance is direct visual inspection of downloaded outputs by one
  evaluator. The Phase 12 semantic-mask benchmark has 13 real objects, but is development data
  used to design/calibrate the prompt, not held-out evidence.
- Phase 14's stage-level lifecycle was proven on a real 2xT4 Kaggle worker: the old per-panel
  Qwen unload (drop ref + `empty_cache()`, no `gc.collect()`) left ~16 GiB resident until an
  opportunistic GC raced the next load into a CUDA OOM (deterministically reproduced at panel
  2 in profiling); with `ModelStage` every stage returns the allocator to ~9 MiB per device
  after release and a full 4-panel page runs with peak 8.7 GiB on one T4 (docs/phase14-results.md).

## Known Limitations and Technical Debt

- `wind_breaker_finish` remains a confirmed, unfixed mid-cycle visual defect. Its original
  MESH_WARP hypothesis was disconfirmed; an oversized PRIMARY translate region is only a lead.
- `marika_love_meter` remains `UNKNOWN`; panel mode safely rejects its independent candidate,
  but this is mitigation, not root-cause repair.
- SAM 2.1 can produce semantically over-inclusive masks that pass all geometric checks. The
  semantic gate has a known real false negative (`cloth_5`, visibly including a speech bubble
  and hand), and its 13-sample development result is provisional: precision 0.75, recall 0.60,
  FPR 0.12, FNR 0.40. The VLM confidence values clustered at round numbers, and ABSTAIN was
  never observed in real calls, so its confidence band is not a calibrated safety valve.
- Same-category instance identity is unresolved: the gate can accept the correct category on
  the wrong physical instance. No real defect of this exact type has yet been observed.
- `MESH_WARP` has no calibrated upper bound relative to panel/page geometry.
- The seam-like artifact detector has approximately 50% real-world precision and is not a
  substitute for targeted visual QA. Whole-frame loop metrics do not detect mid-cycle defects.
- LaMa reconstruction was measurably softer than surrounding line art in all 11 measured real
  instances, although softness did not alone distinguish visible failures from clean outputs.
- Compositing is CPU and sequential across frames; a measured 9780px page with six rendered
  objects took 353.6s in compositing. Local ROI CV work reduced transform cost substantially,
  but full-page output-array contracts and frame assembly remain limits.
- The combined evaluation datasets still lack deliberate real coverage for partial occlusion
  and deformation/scale. The semantic-mask benchmark lacks enough independent labeled masks
  for a held-out calibration study.
- The full 10-sample real-world evaluation has not been rerun after enabling semantic mask
  validation, so the Phase 9 completion metrics are not a post-gate quality claim.
- Phase 13's panel-first implementation has local behavioral coverage and fake-client end-to-end
  coverage for multiple outputs, crop bounds, failure isolation, manifest fields, resumability,
  and cross-panel rejection. Phase 14 then ran representative real GPU validation of the new
  runner (4-panel page, real Qwen/Grounding DINO/SAM 2.1/LaMa on a 2xT4 Kaggle worker),
  producing varied real outcomes (STATIC/STATIC/PASS/REJECTED), with the stage-level lifecycle
  releasing every model deterministically after its stage (allocator back to ~9 MiB per device;
  peak 8.7 GiB on one T4) and manifest-based resumability reusing completed panels on a second
  invocation. See docs/phase14-results.md.

## Immediate Priorities

These are future work, not implemented capabilities:

1. Review the Phase 14 GPU-validated stage-level lifecycle outputs (the Phase 14 E2E run's
   rendered panel videos and crops) at native resolution.
2. Investigate the dense-mask semantic false negative and expand the real labeled-mask dataset
   before changing thresholds or claiming generalization.
3. Run a bounded context-size study for mask verification and establish a genuine development/
   held-out split when data volume permits.
4. Collect targeted same-category multi-instance evidence before designing instance-identity
   validation.
5. Gather evidence for a safe MESH_WARP bound and for mid-cycle artifact detection; do not add
   speculative geometry thresholds from one instance.
6. Treat articulated part-level animation and scene transitions as design-only future concepts,
   not current pipeline features.

## Verification and Workflow

Run locally:

```bash
uv run pytest
uv run pytest -m slow
uv run ruff check .
uv run mypy src
```

Actual model loading and inference must run on a remote GPU worker, never as a local pipeline
smoke test. Generated media and experiment JSON remain under git-ignored `outputs/`; source
changes move between local and remote only through git.

## Documentation Map

- [`../CLAUDE.md`](../CLAUDE.md): permanent operational rules and short reading path.
- [`architecture.md`](architecture.md): stable engineering principles and invariants.
- [`pipeline.md`](pipeline.md): current stage order, ownership, lifecycle, and safety contracts.
- [`animation-plan-schema.md`](animation-plan-schema.md): machine-readable plan contract.
- [`kaggle-jupyter.md`](kaggle-jupyter.md): verified remote-Kaggle connection/execution/watchdog procedure.
- [`decisions/`](decisions/): accepted decisions, supersession, and rationale.
- `phase*-results.md`: immutable historical evidence records; they do not override this file.
