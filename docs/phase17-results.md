# Phase 17 Results: Object Segmentation Diagnostic Benchmark

**Goal.** Answer *where* production masks degrade, with human-annotated ground truth, by
measuring the three independent stages separately -- before changing anything. This phase is
diagnostic only: no production code, thresholds, or gates were modified.

**Date / environment.** 2026-08-15, remote 2× Tesla T4 worker. Grounding DINO `swin-l`
(`IDEA-Research/grounding-dino-base`, thresholds `0.25/0.2`), SAM 2.1 `hiera-base-plus`
(`facebook/sam2.1-hiera-base-plus`), both `float32` -- exactly the production clients.

**Reproducibility.** Canonical code/config in this branch (`configs/phase17_benchmark.yaml`,
`src/manga_animation/benchmarking/phase17/`); normalized artifacts (page images, GT masks,
per-sample masks, report JSON, visual packages) under the git-ignored
`outputs/experiments/phase17_object_segmentation/` (run `run_20260815_134104`). Tests:
`tests/test_phase17_*` (666 total pass, `ruff`, `mypy` clean). Metrics were independently
verified on synthetic masks before use (`tests/test_phase17_metrics.py`).

---

## 0. Dataset and method

- **GT source (mandatory):** MS92/MangaSegmentation human-annotated instance masks (COCO RLE),
  gated. Page images: non-gated `longle0702/manga109-segmentation` mirror of the Manga109 pages
  (the mirror covers ~1158 of ~10602 pages; the manifest is constrained to that set). MS92-vs-
  mirror annotation agreement is spot-verified (decoded masks match, IoU 1.0) and every page's
  image dimensions are checked against the MS92 record. Face masks are contained within body
  masks (median containment 1.000), so the body GT is the full character silhouette.
- **Main-object category:** `body` only. The other five MangaSegmentation categories (face,
  text, balloon, frame, onomatopoeia) are the brief's forbidden set and are excluded from the
  main score; face/text/balloon/frame/onomatopoeia are analyzed only in the safety track.
  Comix Books v0 was inspected and **excluded**: its masks are SAMv2-model-generated and
  aggregated per element type (not human-annotated per instance) -- circular as GT for a SAM
  benchmark (brief §4: "STOP investigating ... proceed using MangaSegmentation").
- **Benchmark:** 64 `body` instances across 52 pages and 23 books (size/context-stratified,
  seeded selection; GT bbox area 0.07%–23% of page). DINO prompt is the production prompt
  `"character body."` (via the real `_prompt_from_label`).
- **Three experiments** (brief §7):
  - **A:** GT bbox → SAM → mask vs GT (isolates SAM).
  - **B:** image → DINO → top bbox vs GT bbox (isolates localization).
  - **C:** image → DINO → production ranking/selection → SAM → production gates → mask vs GT
    (the real path). Reuses the actual `ground_object_candidates`, `segment_object`, the
    deterministic `_bbox_plausibility` half of `validate_target`, and the real `_validate_mask`
    / `_validate_mask_shape` gates. The VLM semantic gates (`validate_target`'s VLM check,
    `mask_semantics`) are **not** part of these experiments (brief §14: only DINO + SAM run).
- **Full-page grounding** (no panel crops, no panel GT available in the dataset). Production
  panel mode crops DINO's input to the panel, which removes distractor characters. This
  benchmark therefore measures the *harder, full-page* case; it is a conservative lower bound
  for panel-mode production localization (disclosed limitation, not a claim about panel mode).

## 1. Results

### Experiment A — pure SAM with a perfect box

| metric | mean | median | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| IoU | 0.853 | **0.884** | 0.815 | 0.926 | 0.524 | 0.962 |
| Dice | 0.917 | 0.939 | 0.898 | 0.961 | 0.687 | 0.981 |
| precision | 0.891 | 0.924 | 0.847 | 0.967 | 0.530 | 0.996 |
| recall | 0.953 | 0.961 | 0.944 | 0.972 | 0.737 | 0.990 |

**SAM 2.1 with a correct box segments manga characters well** (median IoU 0.88, 50% of samples
in 0.82–0.93; recall higher than precision, i.e. SAM slightly over-covers). Only 1/64 samples
drops below 0.6 IoU.

### Experiment B — Grounding DINO localization

| metric | mean | median | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| bbox IoU | 0.061 | **0.000** | 0.000 | 0.000 | 0.000 | 0.964 |
| GT coverage | 0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| area ratio | 6.56 | 2.03 | 0.93 | 5.83 | 0.24 | 86.8 |

- Detection rate 64/64 (something was always found above threshold), but **60/64 top detections
  have bbox IoU < 0.3 with the target instance** (wrong instance / wrong object). DINO's scores
  on these wrong boxes were 0.44–0.67 -- far above the 0.25 threshold, so no threshold tweak
  would have fixed this.
- Boxes are systematically too large (median area ratio 2.0; up to 87× the GT), consistent with
  detections grabbing a whole panel region instead of one character.

### Experiment C — real production DINO → selection → SAM → gates

| metric | mean | median | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| IoU (accepted, n=44) | 0.062 | **0.000** | 0.000 | 0.000 | 0.000 | 0.962 |

- Outcomes: 44/64 `accepted`, 20/64 `segment_gate_rejected`, 0 `candidate_selection_rejected`,
  0 `no_detection`.
- Of the 44 accepted masks, **41 have IoU < 0.3** (the segmented region is not the target
  character). Only **3/64 samples are healthy end-to-end** (C ≥ 0.5): MutekiBoukenSyakuma_086
  (0.96), GarakutayaManta_095 (0.92), YumeiroCooking_086 (0.86).

### The key pairwise evidence (brief §8)

| sample | GT-bbox→SAM (A) | DINO bbox IoU (B) | DINO→SAM (C) | outcome |
|---|---|---|---|---|
| MutekiBoukenSyakuma_086_430161 | 0.96 | 0.95 | 0.96 | accepted |
| UltraEleven_111_695642 | 0.95 | 0.95 | *(rejected)* | segment_gate_rejected |
| GarakutayaManta_095_160155 | 0.91 | 0.94 | 0.92 | accepted |
| YumeiroCooking_086_748655 | 0.86 | 0.96 | 0.86 | accepted |
| UltraEleven_110_695540 (typical failure) | 0.95 | 0.00 | 0.00 | accepted |

When DINO's box is right (4/64 samples), C ≈ A: **the whole DINO→SAM→gates chain is healthy
downstream of grounding.** When DINO's box is wrong (60/64), the production mask is wrong
regardless of how good SAM is. The bottleneck is localization, not segmentation.

## 2. Category-level analysis

The main benchmark has a single category (`body`), so cross-category variation within the main
score does not exist. The **safety track** (brief §10) covers the forbidden categories instead:

- **Forbidden-content absorption into production masks** (fraction of the final mask's pixels
  inside forbidden GT masks; 44 accepted samples): text **0.001** (max 0.026), balloon **0.010**
  (max 0.235), onomatopoeia **0.003** (max 0.035) -- negligible. The pipeline does **not**
  incorrectly absorb speech bubbles, dialogue text, or sound effects into object masks.
- **frame** overlap is high (mean 0.953), but this is NOT a text/safety finding: the MS92
  `frame` masks cover 60–97% of every page (the frame/background category), so any mask sitting
  in the page background -- which is what DINO's wrong detections produce -- overlaps frame by
  construction. The correct reading: wrong production masks are **background/frame content, not
  character content**, confirming the localization failure rather than a forbidden-target
  selection bug.
- The production **analysis-level gates** (VLM label selection, `mask_semantics`) were not
  exercised (brief §14) -- the "does the pipeline animate a forbidden region" question for the
  full production system remains out of scope for the GPU runs, unchanged.

## 3. Post-processing / gate analysis

- **20/64 masks were rejected** by the production `_validate_mask` / `_validate_mask_shape`
  gates. Of these, **19 had a raw SAM mask with IoU ≈ 0** (DINO's wrong box → SAM segmenting the
  wrong region) -- the gates rejected already-bad content, i.e. fail-closed correctly.
- **1/64 is a clear false rejection**: UltraEleven_111_695642 has A=0.95, B=0.95, a good raw
  mask, yet the production edge-asymmetry gate rejected it. The known `_MAX_BBOX_EDGE_TOUCH_
  FRACTION` heuristic can reject a genuinely good character mask (a real over-rejection, 1.6%).
- Net: candidate selection and post-processing are **not** the dominant quality bottleneck.
  They mostly reject what grounding already broke; they cost one good mask per 64.

## 4. Dominant visual failure modes

Ranked by observed frequency on the real montages
(`outputs/.../run_20260815_134104/visual_failures/*.png`, 12 packages for human review):

1. **Wrong-instance localization** (~60/64): DINO's top "character body." detection is a
   *different* character (or a whole panel region) than the target instance; SAM faithfully
   segments the wrong object and production accepts it.
2. **Panel-region / oversized detections**: DINO boxes up to 87× the GT box, grabbing whole
   panels; the resulting masks are background/frame content.
3. **One real SAM over-segmentation** (A min 0.524) and **one real gate false rejection**
   (UltraEleven_111_695642) -- both rare, neither is the primary problem.

## 5. Answers to the phase questions

1. **How good is SAM 2.1 with a perfect GT bbox?** Good: median IoU **0.884** (mean 0.853),
   precision 0.891 / recall 0.953. SAM is the healthy stage.
2. **How good is current Grounding DINO localization?** Poor for *specific-instance*
   localization on full pages: median bbox IoU **0.000**; 60/64 top detections are the wrong
   instance (bbox IoU < 0.3), with detection scores 0.44–0.67 that no threshold would fix.
3. **How good is the actual DINO → SAM production path?** Median IoU **0.000**; only 3/64
   healthy end-to-end. It faithfully inherits grounding's failure.
4. **Gap between GT-bbox→SAM and DINO-bbox→SAM?** Massive: A median 0.884 vs C median 0.000
   (accepted). When DINO is right, C ≈ A (0.86–0.96); when DINO is wrong, C ≈ 0.
5. **Which categories are difficult?** Not separable within one main category. Forbidden
   categories: text/balloon/onomatopoeia are *not* absorbed (good); frame overlap is a
   background artifact of the localization failure, not a category-specific defect.
6. **Dominant visual failure modes?** Wrong-instance localization and oversized panel-region
   detections (above).
7. **Primary bottleneck?** **Grounding DINO localization (specific instance, full-page
   context).** SAM is healthy; the gates reject mostly-bad masks (1/64 false rejection).
8. **Evidence?** The three-way per-sample diagnostics in §1 (64 samples), the C≈A-when-B-good
   subset (4/64), the 60/64 wrong-instance rate, and the 41/44 accepted-but-wrong production
   masks.

## 6. Observed facts vs hypotheses

**OBSERVED (measured this phase, 64 real human-annotated samples):**
- SAM median IoU 0.884 with the GT box; DINO median bbox IoU 0.000; production median IoU 0.000.
- 60/64 DINO top detections wrong instance; 41/44 accepted production masks wrong region; 3/64
  healthy end-to-end; 4/64 with a correct DINO box (all C≈A).
- 20/64 gate rejections, 19 of bad masks + 1 false rejection of a good mask.
- Text/balloon/onomatopoeia absorption ≈ 0.
- Full-page grounding (no panel crops) was measured; scores are a conservative lower bound for
  panel-mode production.

**HYPOTHESES (not established by this benchmark; listed for the next phase to test):**
- Panel-crop grounding (production panel mode) substantially raises the DINO-specific-instance
  rate -- plausible from this data (removing ~5–10 distractor characters) but **not measured**.
- DINO's "character body." phrase is unusually weak on manga line art; other promptings
  (e.g. "character." / a single phrase per detected character) may behave differently --
  **not measured** (one prompt only).
- The 20/64 gate-rejection rate would drop if grounding improved (most rejections were of
  already-wrong masks) -- a consequence, not separately measured.
- No claim of statistical calibration is made for a 64-sample, 23-book set.

## 7. Reproducibility and artifacts

- Metrics independently verified on synthetic masks (`tests/test_phase17_metrics.py`).
- Manifest: `configs/phase17_benchmark.yaml` (64 samples, seed 17).
- Data prep: `scripts/run_phase17_prepare_dataset.py` (needs `HF_TOKEN`; gated MS92).
- GPU run: `scripts/run_phase17_gpu_benchmark.py --manifest ... --dino ... --sam ...`.
- Results (git-ignored): `outputs/experiments/phase17_object_segmentation/run_20260815_134104/`
  (per-sample JSON + masks, `report.json`/`report.md`, `forbidden_overlap.json`, 12 visual
  failure packages). Saved locally from the remote worker.
- All source/tests pass: `uv run pytest` (666), `uv run ruff check .`, `uv run mypy src`.

## 8. What this phase did NOT do

Per the brief, Phase 17 did not modify: Grounding DINO, SAM 2.1, checkpoints, prompts,
thresholds, mask semantic gates, geometry gates, candidate ranking, post-processing,
architecture, effects, or rendering. No production change follows from this report without an
explicit next decision.
