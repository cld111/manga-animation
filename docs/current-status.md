# Current Project State

This is the single canonical answer to "what is true about the project right now?" It is
operational state, not a phase report. Historical implementation details and experiment
evidence remain in `docs/phase*-results.md`; decision rationale remains in `docs/decisions/`.

## Status

The deterministic pipeline and local test/evaluation infrastructure are implemented through
Phase 13's panel-first orchestration, hardened by Phase 14's stage-level model lifecycle,
reordered by Phase 18.3 (single VLM object-description stage) and Phase 18.4 (VLM before
segmentation: SAM segments only bboxes with an accepted action description), and validated
across multiple real pages and repeated GPU runs. Real model execution
remains a remote-GPU operation. The project is an engineering prototype with real end-to-end
evidence and known real-world visual limitations, not a production animation service.

## Current Pipeline

The implemented order is:

```text
page -> deterministic panel detection -> bounded scene crops
  -> grounding (DINO) -> object_description (Qwen, ONE call with ALL bboxes)
  -> segmentation (SAM, only for accepted bboxes) -> animation
  -> reconstruction -> compositing -> rendering
```

- `run_page_panels` is the production page entry point: every detected panel gets a stable unit,
  its own scene crop, independent stages, output video or explicit status, and a page manifest.
  It is the single-page wrapper over the Phase 18.4 batch entry point `run_pages`, which
  processes MANY pages with one model residency per stage ACROSS pages: each model loads ONCE,
  processes every eligible panel of every page (saving results per page), then releases --
  never a per-page load/unload cycle.
- `panel_bbox` is logical geometry; `scene_crop_bbox` is the actual analysis/render canvas and
  is bounded by page edges and nearby panel geometry.
- Analysis is panel-aware by default; page analysis remains explicit. A panel's all-STATIC result
  is recorded as `STATIC` by the panel runner without inventing a video.
- Grounding uses a real panel crop when analysis provides one and returns page coordinates.
- `validation` checks grounded bbox plausibility, semantic agreement, and transform geometry
  before segmentation.
- `segmentation` produces a full-source-image `uint8` mask and applies coverage and asymmetric
  edge-touch safety checks.
- `object_description` (Phase 18.3) is the pipeline's ONLY VLM stage. Qwen2.5-VL sees the FULL
  image plus ALL of its grounded candidates' bboxes as pixel coordinates in ONE call (never a
  crop, never the mask), reads the ACTION happening in the scene, judges each candidate
  (pass/ambiguous/partial/reject/not_animatable), and produces a structured animation
  description whose deterministically-mapped `MotionSpec` drives the animation stage (with the
  SAM mask). Fail-closed: candidates with a non-pass read, an identity-conflict (text/bubble/
  background), or a geometrically unsafe bbox+transform are dropped; a panel with no accepted
  candidate is REJECTED. The architecture has no analysis stage, no crop-based VLM validation
  and no mask-semantics stage; candidate labels come from the caller
  (`DEFAULT_ANIMATION_LABELS`). See docs/phase18.3-results.md.
- Segmentation runs AFTER object description (Phase 18.4 ordering: DINO -> Qwen -> SAM): SAM2
  segments ONLY the bboxes that earned an accepted action description, so it never spends an
  inference on a rejected candidate. A VLM-accepted bbox whose mask then fails the
  post-segmentation shape checks is still dropped (fail closed).
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
| Semantic mask validation stage | not part of the 18.3/18.4 runtime (flag retained for legacy evaluation) |
| Per-candidate VLM object description | enabled (Phase 18.3, runs before segmentation since Phase 18.4) |

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
- Transform-aware validation runs inside the object-description stage, before segmentation,
  because it validates a bbox against a transform kind; mask-level shape checks are
  post-segmentation because they need the real mask.
- `parent_id`/`children_ids` are structurally validated, but parent transforms are not
  inherited automatically; each animated object needs its own motion spec.
- Model-backed stages release their clients after the stage, including grounding, the VLM
  object-description stage, segmentation, and reconstruction.
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
- Phase 15 validated the Phase 14 lifecycle across 6 real pages (17 panels) in one 2xT4 session,
  plus repeated execution and a resume run: every model stage released exactly its own
  footprint (Qwen ~15.8 GiB per VLM stage, DINO ~892 MiB, SAM ~283 MiB, LaMa ~197 MiB) on every
  page, `allocated` returned to ~73/9 MiB per device after every page (17 timeline samples with
  both GPUs fully released), timeline peak was 8.7 GiB on one T4, and a repeated `villainess`
  run reproduced the same per-panel statuses (docs/phase15-results.md). An injected raw
  grounding `RuntimeError` isolated to its panel (ERROR) while later panels still processed and
  all models were released; a 1xT4 smoke run (CUDA_VISIBLE_DEVICES=0) completed with Qwen
  fitting on a single T4 (~12.1 GiB, CPU-offloaded) and the same stage lifecycle. Phase 15 also
   found and fixed two lifecycle teardown defects: a raising `client.unload()` could mask a
   stage exception and skip the deterministic release, and a failed `client.load()` in
   `__enter__` left the stage object permanently poisoned.
- Phase 16 added the drawn-effect animation track (`RADIAL_EXPAND` motion model plus
  effect-aware `_MOTION_HEURISTICS` entries and an analysis prompt that asks the VLM to list
  already-drawn effects as animation targets). A short real-GPU run on `wind_breaker_sprint`
  produced the first end-to-end drawn-effect render: a `speed_lines` PRIMARY (mesh_warp)
  passed grounding, geometric and semantic target validation, and mask-semantics, and
  rendered a PASS panel video with the seamless loop verified on the source frames; the
  page's `impact_burst` SECONDARY was correctly fail-closed by geometric validation instead
  of rendered. See `Known Limitations`/`Immediate Priorities` for what Phase 16 did and did
  not yet exercise.
- Phase 17 (diagnostic only, no production changes; see docs/phase17-results.md) measured the
  production DINO->SAM path against 64 human-annotated MangaSegmentation `body` instances
  (23 books) in three independent experiments. **Result: the bottleneck is Grounding DINO
  specific-instance localization, not SAM.** With a perfect GT box, SAM 2.1 segments manga
  characters well (median IoU 0.884, mean 0.853, recall 0.953). Grounding DINO's top
  "character body." detection on full pages is the wrong instance in 60/64 samples (median
  bbox IoU 0.000; detection scores 0.44-0.67, so no threshold fix), and the production path
  inherits this (median IoU 0.000; only 3/64 healthy end-to-end). When DINO's box is right
  (4/64), DINO->SAM->gates IoU ~= SAM-only IoU (0.86-0.96). Production gates rejected 20/64
  masks, 19 of already-bad masks + 1 false rejection (UltraEleven_111_695642). Safety track:
  text/balloon/onomatopoeia are not absorbed into object masks (<=1-3%); frame overlap is a
  background artifact of the localization failure, not a forbidden-target selection bug.
  Measured on full pages (no panel GT in the dataset) -- a conservative lower bound for
  panel-mode production. Comix Books v0 was excluded as GT (SAMv2-generated, aggregated
  masks -- circular for a SAM benchmark).
- Phase 18.1 (diagnostic only, docs/phase18.1-results.md) answered the follow-up: is the
  correct target present among ALL DINO detections, and at what rank? On the same 64 targets:
  **the correct candidate exists (R@All) in 89.1% at IoU >= 0.5 (78.1% at IoU >= 0.75), but
  only 6.2% are top-1** (R@3 23.4%, R@10 59.4%, R@20 78.1%; median best-correct rank 8).
  Category split at 0.5: A=4 (top-1), B=53 (below top-1), C=7 (absent). DINO's own confidence
  is not a usable ordering signal (correct candidates score ~0.74x the wrong top-1). This is
  Case A (candidate exists, ranking is the problem): next step is candidate selection /
  reranking (Phase 18.2), not candidate generation; the 7 category-C targets are the
  grounding-scale floor, not fixable by a selector.

## Known Limitations and Technical Debt

- `wind_breaker_finish` remains a confirmed, unfixed mid-cycle visual defect. Its original
  MESH_WARP hypothesis was disconfirmed; an oversized PRIMARY translate region is only a lead.
- `marika_love_meter` remains `UNKNOWN`; panel mode safely rejects its independent candidate,
  but this is mitigation, not root-cause repair.
- SAM 2.1 can produce semantically over-inclusive masks that pass all geometric checks. The
  semantic gate's 13-sample development result is provisional and has evolved: Phase 12's
  original prompt measured precision 0.75, recall 0.60, FPR 0.12, FNR 0.40, with a known
  real false negative (`cloth_5`, visibly including a speech bubble and hand). Phase 16
  strengthened the prompt to direct the VLM at absorbed adjacent content (text/bubbles/
  hands/faces); a real-VLM re-run measured precision 0.67, recall 0.80, FPR 0.25, FNR 0.20 --
  `cloth_5` is now caught. The two new "false rejects" (raised_sword_12, character_eyes_2,
  each flagged "includes a speech bubble with text") were forensically investigated with a
  neutral VLM probe: the VLM consistently reads multi-object scenes in both (raised_sword_12:
  "sword with a speech bubble '으악' and another character"; character_eyes_2: "a character
  holding a glowing orb, wearing armor, speaking"), while correctly reading the cloth_5
  control -- and raised_sword_12's tight bbox is 271x386 (aspect 0.7, 50% dark), not a thin
  sword. The provisional conclusion is that those two GOOD labels are more likely wrong than
  the prompt over-rejected (FPR 0.25 partly an artifact of stale labels); final adjudication
  still requires human visual inspection of `outputs/debug/phase16_forensics/*_vlm_crop.png`.
  The VLM confidence values clustered at round numbers, and ABSTAIN was never observed in
  real calls, so its confidence band is not a calibrated safety valve.
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

- `RADIAL_EXPAND` (Phase 16) is the drawn-effect motion model for the radial class of manga
  effects (impact bursts, energy fields, radiating focus lines, glow): a spatially-varying
  radial pulse about the object's own center where the center stays effectively fixed while
  the rim breathes outward/inward -- unlike uniform `SCALE`, which moves the whole footprint
  as one rigid block. The analysis prompt now asks the VLM to list ALREADY-DRAWN effects
  (speed lines, impact bursts, energy fields, smoke, water, glow, sparks) as first-class
  animation targets, and `_MOTION_HEURISTICS` maps effect labels to effect-specific motion:
  impact/burst/energy/glow -> `radial_expand`, smoke/steam/water/fluid -> `mesh_warp`, rain
  -> translate-down, sparks/particles -> opacity flicker, speed lines -> `mesh_warp`. Before
  Phase 16 every effect label (`rain`, `green_fluid`, `speed_lines`, `impact_effect`,
  `energy_effect`, `smoke`) collapsed to the SAME rigid `_DEFAULT_MOTION` (uniform translate,
  amplitude 0.02) -- exactly the "simple geometric displacement" the phase brief rejects.
- Phase 16 GPU evidence (real worker, 2xT4): on `wind_breaker_sprint` the analysis stage
  labeled `speed_lines` PRIMARY with `mesh_warp` (confidence 1.0) and `impact_burst`
  SECONDARY with `radial_expand`; the speed-lines PRIMARY passed grounding, geometric
  validation, semantic target validation, and mask-semantics (VLM confirmed "only speed
  lines"), and rendered a real PASS panel video with the seamless loop verified on the
  source frame sequence (wrap step <= 2x ordinary step) and 93.8% of pixels static across
  sampled frames -- the first end-to-end drawn-effect animation through the production
  pipeline. Ordinary objects (hair, cloth, bicycle) still map to their pre-Phase-16
  transform kinds; effect heuristics are ordered after the object heuristics and do not
  shadow them. The `impact_burst` SECONDARY on the same page was correctly fail-closed by
  geometric validation (bbox 86.7% of its reference region, and edge-touching) rather than
  rendered. On `eval_weapon_effects` the PRIMARY remained `weapon` (rotate) and was
  REJECTED by the known oversized-rotate-bbox geometric failure (the Phase 3.3 defect class,
  still correctly fail-closed), so that page did not exercise the effect track.
- Phase 16 extended evidence: after an effect-mask-density diagnostic showed real effect
  masks are sparse (6-17% of panel at density 0.28-0.50 while their grounding boxes reached
  98%), `radial_expand`'s pre-segmentation profile was relaxed to 60% area / 0 edge margin
  and a post-segmentation `max_mask_density` gate (0.70) was added. On `angels_of_war_fleet`
  the `space_ship_impact_burst` then passed grounding, geometry, semantic and mask-semantics
  ("only the space ship impact burst", confidence 1.0) and rendered -- the first end-to-end
  RADIAL_EXPAND impact-burst render (loop verified: wrap 2.13 <= 2x ordinary, 78.9% pixels
  static). A repeated sprint run and a `space_monster_creature` run ([PASS, PASS]) confirmed
  no regression to ordinary objects or to the already-working speed-lines path. A
  `wind_breaker_finish` run had one panel CUDA-OOM on the shared T4 (isolated as ERROR,
  remaining panels processed, all six model stages released) -- a documented resource-pressure
  class, not a lifecycle regression. The `impact_burst` -> `radial_expand` mapping was also
  confirmed on a real VLM after the label-keyed effect-classification fix (was `rotate` when
  the description mentioned a weapon).
- Phase 16 text-animation guard (goal 4, real finding on `sss_hunter_gladiator`): the VLM
  once labeled free-standing dedication text SECONDARY and the pipeline animated it (rotate)
  -- both the semantic target check and mask-semantics ACCEPTed it because text was the
  mask's only content. Fixed deterministically: `ANALYSIS_PROMPT` now forbids animating
  text-like elements, `_is_text_label` forces text-like semantic_labels to STATIC, and
  `_rank_candidates` excludes them from animated candidacy. Verified on a re-run: dedication
  text no longer animated, motion confined to the drawn speed-lines/objects.

## Immediate Priorities

These are future work, not implemented capabilities:

1. Investigate the dense-mask semantic false negative and expand the real labeled-mask dataset
   before changing thresholds or claiming generalization.
2. Run a bounded context-size study for mask verification and establish a genuine development/
   held-out split when data volume permits.
3. Collect targeted same-category multi-instance evidence before designing instance-identity
   validation.
4. Gather evidence for a safe MESH_WARP bound and for mid-cycle artifact detection; do not add
   speculative geometry thresholds from one instance.
5. Treat articulated part-level animation and scene transitions as design-only future concepts,
   not current pipeline features.
6. Visually review the Phase 16 RADIAL_EXPAND render (angels_of_war_fleet impact burst) and
   the speed-lines render (wind_breaker_sprint) -- they passed numeric loop/static checks but
   have not had human visual QA.
7. Add more effect-heavy pages to the real evaluation set: smoke/water/sparks/glow render
   paths remain unexercised end-to-end (only speed lines and impact burst confirmed).
8. Confirm the relaxed RADIAL_EXPAND bounds on real renders: Phase 16 evidence showed drawn
   effect masks are sparse (real speed_lines/impact_burst masks covered 6-17% of their panel
   at density 0.28-0.50 while their grounding boxes reached 98%), so `transform_geometry.py`
   now allows a 60% bbox area and a 0 edge margin for radial_expand and
   `segmentation/segment.py` rejects dense masks (density > 0.70, the confirmed "select
   everything" signature) post-segmentation. Neither threshold is statistically calibrated.
9. Investigate the wind_breaker_finish panel-1 CUDA OOM on a shared T4 (578 MiB request with
   406 MiB free while the sharded Qwen held 13.4 GiB): the panel is correctly isolated and
   all models release, but a worker-level retry/fallback for OOM on large panels would make
   dense pages more robust (same class as Phase 11's documented LaMa OOMs).
10. **Phase 17 follow-up (grounding is the measured bottleneck, docs/phase17-results.md):**
    before touching anything, decide a direction -- e.g. (a) confirm panel-crop grounding on a
    real panel-GT benchmark (the Phase 17 numbers are full-page and are a lower bound for
    panel mode), (b) evaluate prompt phrasing / a second grounding model (OWLv2 is an existing
    manifest candidate), or (c) accept instance-level ambiguity and redesign candidate
    selection. Do not change production grounding on a 64-sample full-page result alone.
11. **Phase 18.1 follow-up (docs/phase18.1-results.md):** the correct candidate exists among
    DINO detections in 89% of targets but is top-1 in only 6% -- so Phase 18.2 should build a
    candidate selector/reranker over top-K using an independent signal (the production
    `validate_target` VLM check is the existing candidate signal), measure reranked Recall@K
    and end-to-end DINO->SAM->gates IoU on the same 64 targets, and keep the 7 category-C
    targets (no candidate at all) out of the selector's success claim.

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
  Phase 17 (object-segmentation diagnostic benchmark) lives in
  [`phase17-results.md`](phase17-results.md); Phase 18.1 (DINO candidate recall) in
  [`phase18.1-results.md`](phase18.1-results.md). Phase 18.3 (per-candidate VLM object
  description) evidence in [`phase18.3-results.md`](phase18.3-results.md) and the full work
  report in [`phase18.3-report.md`](phase18.3-report.md).
