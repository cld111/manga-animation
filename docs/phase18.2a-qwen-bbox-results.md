# Phase 18.2A Results — Qwen2.5-VL Direct Target Localization Benchmark

**Question.** For each of the 64 phase-17 human-annotated `body` targets: *can Qwen2.5-VL,
given the full page and the production target description, localize the SPECIFIC target
instance well enough to replace or spatially guide the DINO → candidate-selection path?*
This is the direct test of the simplest architecture (`Qwen → bbox → SAM`) before investing
further in a reranker.

Diagnostic only: no production code, prompts, thresholds, or gates were changed.

**Date / environment.** 2026-08-15, remote 2× Tesla T4 worker. Qwen2.5-VL-7B-Instruct
(float16, `device_map="auto"`), SAM 2.1 `hiera-base-plus` (float32) — the exact production
clients. Only the VLM and SAM ran (no DINO).

**Reproducibility.** Code: `src/manga_animation/benchmarking/phase18a/` +
`scripts/run_phase18_2a_gpu_benchmark.py`. Results (git-ignored):
`outputs/experiments/phase18_2a_qwen_bbox/run_20260815_164759/`
(`predictions_by_sample.json`, `sam_masks_by_sample.json`, `report.json`/`report.md`,
`per_target.json`, `visuals/`, `run_meta.json`). Reuses the phase-17 manifest/dataset
(64 targets, 52 pages, 23 books). Coordinate conversion is unit-tested
(`tests/test_phase18a_coords.py`); 720 tests pass, `ruff`/`mypy` clean. The VLM stage is
cached per sample, so the run is resumable.

**Method.** Qwen2.5-VL sees the FULL page at source resolution (its processor bounds vision
tokens internally) plus the production target description — the manifest's `"character body."`,
the same text Grounding DINO received in Phase 17/18.1. It returns
`{"found": bool, "bbox": [x1,y1,x2,y2]}`; GT is used only for scoring, never in the prompt.

---

## 0. Coordinate contract (measured, not assumed)

The brief required pinning down Qwen's coordinate space before trusting any bbox. A 3-sample
GPU smoke run established it empirically, and the full run then confirmed it cleanly:

- **Qwen2.5-VL reports coordinates in the PIXEL space of the source image, not a normalized
  0..1000 scale.** On 1654×1170 pages the model returned values > 1000 (x=1254, y=1174 ≈ page
  height 1170), which is only possible if the numbers are source-image pixels. The initial
  prompt requested the model's native 0..1000 grounding convention and was ignored; assuming
  0..1000 would have silently shrunk every box.
- The corrected prompt states the exact page size and requests pixel coordinates. Conversion is
  identity up to a 5% edge-tolerance clamp (a model overshoot of a few px past the page border
  is a normal spatial-estimate artifact), and any out-of-convention / non-integral / degenerate
  response is flagged as a conversion failure — never silently rescaled or swapped (`coords.py`,
  unit-tested).
- **Full-run result: 0 conversion failures in 64 responses.** The contract held cleanly; the
  "no silent coordinate mismatch" requirement is satisfied and reproducible.

## 1. Results

### Qwen direct bbox vs GT bbox

| IoU threshold | recall (all 64) | recall (found, n=55) |
|---|---|---|
| 0.25 | 7.8% | 9.1% |
| **0.50** | **4.7%** | **5.5%** |
| 0.75 | 1.6% | 1.8% |

- Found (usable source-pixel bbox): **55/64 (85.9%)**; not found: 9; conversion failures: 0.
- bbox IoU (found): mean **0.056**, median **0.000**, p25 0.000, p75 0.000, max **0.942**.
- bbox IoU (all 64, not-found = 0): mean 0.048, median 0.000.
- GT coverage (found): mean 0.100, median 0.000. Area ratio (found): mean 7.8×, median 2.9×,
  max 55× (boxes systematically far larger than the target).
- **Primary metric: Recall@IoU≥0.5 = 4.7% (3/64).**

### Error categories (phase-brief taxonomy)

| category | count |
|---|---|
| 9 coordinate conversion failure | 0 |
| 8 VLM not found | 9 |
| 7 target outside panel / page grab | 0 |
| 6 multiple similar objects (visual review) | 0 |
| 5 partially hidden object (visual review) | 0 |
| 4 bbox too small | 2 |
| 3 bbox too large | 27 |
| 2 wrong instance | 23 |
| 1 correct object, imprecise bbox | 2 |
| 0 good (IoU ≥ 0.75) | 1 |

### Downstream: Qwen bbox → SAM → mask vs GT mask

- Qwen bbox → SAM mask IoU (found, n=55): mean **0.047**, median **0.000**, max **0.947**.
- Qwen bbox → SAM mask IoU (all 64, not-found = 0): mean 0.040, median 0.000.

### Reference: GT bbox → SAM → mask vs GT mask

- GT bbox → SAM mask IoU (n=64): mean 0.853, **median 0.884**, p25 0.815, p75 0.926,
  min 0.524, max 0.962 — a byte-for-byte reproduction of the Phase 17 reference (median 0.884):
  same dataset, same SAM path, cross-validated.

## 2. Comparison with the existing pipeline

| signal | value |
|---|---|
| DINO top-1 bbox recall (Phase 18.1) | 6.2% |
| DINO candidate availability R@All (Phase 18.1) | 89.1% |
| **Qwen direct bbox recall@0.5** | **4.7%** |
| **Qwen bbox → SAM median mask IoU** | **0.000** |
| GT bbox → SAM median mask IoU | 0.884 |

## 3. OBSERVED FACTS

1. **Qwen localizes *a* character body 86% of the time, but the SPECIFIC instance only 4.7%**
   (3/64 at IoU ≥ 0.5; 1/64 at ≥ 0.75). Median bbox IoU is 0.000.
2. **The failure is instance selection, not mechanics.** 27 boxes are too large (median area
   ratio 2.9×, up to 55× — panel-scale), 23 are a different character, 9 not found, 2 too
   small. Zero coordinate-conversion failures; when Qwen commits to the right instance its box
   is tight (0.942 for UltraEleven_111_695642).
3. **The "wrong instance" cases are systematic, not random.** On pages with several GT targets,
   5/20 of Qwen's boxes overlap *another* GT target at IoU > 0.3; e.g. on
   MutekiBoukenSyakuma_086 Qwen returns the same dominant character (430162) both when that
   character IS the target (IoU 0.58, correct) and when it is NOT (target 430168 → IoU 0.00).
   With only `"character body."` Qwen picks a salient character, not the specified one.
4. **The Qwen bbox → SAM downstream is only as good as the bbox.** Median mask IoU 0.000
   overall; 0.947 on the one correct-tight-box case (UltraEleven_111_695642, which also carries
   a good GT→SAM mask 0.947). SAM is not the bottleneck (GT→SAM median 0.884 reproduces
   Phase 17 exactly).
5. **Qwen does not rescue the 7 category-C targets** (no DINO candidate in Phase 18.1): it
   finds a box in 6/7 but the best overlap is 0.38 (< 0.5).
6. **The 9 not-found targets overlap heavily with the small/ambiguous end** (GT box area
   0.07%–1% of page for most), consistent with the Phase 18.1 category-C floor.

## 4. INTERPRETATION

- **Qwen2.5-VL cannot resolve the specific target instance from the production-available
  description on full pages — and it is no better than DINO's top-1 (4.7% vs 6.2%).** Both
  models receive the same `"character body."`; neither can decide *which* character. The
  instance-resolution problem is therefore not a model-capability gap that a second VLM or a
  better ranking alone fixes: it is an **information gap in the description**.
- **Qwen's bbox is not usable directly (Option A) and not usable as a spatial hint (Option B).**
  50/64 found boxes are a wrong instance or a whole-panel/too-large region; feeding such a box
  to SAM (measured: median 0.000) or as a DINO prior would mislead more often than it helps.
  The 3 correct boxes are too rare to carry a production path.
- **The correct-next-signal remains DINO's candidate availability (R@All 89.1%).** Qwen adds
  nothing measured over DINO's top-1, so the Phase 18.1 conclusion stands: candidate
  *selection* over DINO's top-K is the only recovery path the data supports, and it needs a
  selector signal stronger than both DINO's score (Phase 18.1 fact 5) and Qwen's presence-style
  verification (Phase 18.2's 1/5 finding).
- **Production panel mode is the untested advantage.** These are full-page results; production
  analysis is per-panel with typically 1–3 characters, where `"character body."` becomes
  nearly specific. Phase 17's own framing (full-page = conservative lower bound) applies here
  too: the measured 4.7% is a floor, not a panel-mode claim.

## 5. HYPOTHESES (not established by this benchmark)

- **A richer, per-instance description (panel context, appearance, position) would lift Qwen's
  recall substantially.** Plausible — all 3 correct selections produced tight boxes (0.58–0.94)
  — but NOT measured: production currently stores no such description for these 64 targets.
- **Panel-cropped input would help both Qwen and DINO.** Consistent with Phase 17's framing but
  not measured (no panel GT in the dataset).
- **Qwen's box could still serve as a coarse *candidate* in a hybrid selector** (its
  wrong-instance boxes ARE real character proposals). Not tested; the evidence that a selector
  can use them is absent.
- No claim of statistical calibration is made for a 64-sample, 23-book set.

## 6. LIMITATIONS

- The benchmark measures **full-page localization with the single production-available
  description** (`"character body."`). It does not measure panel-mode production or richer
  descriptions, both of which are expected to be easier and are untested here.
- The primary metric is bbox IoU vs the **tight GT box**; a loose-but-correct box fails at
  IoU 0.5 even when SAM downstream is fine (e.g. MutekiBoukenSyakuma_086_430162: bbox IoU
  0.58, but qs mask 0.29 because the box overshoots). The downstream experiment partially
  compensates but inherits the overshoot.
- Error categories are mechanical thresholds (documented in `classify.py`); categories 5/6
  are flagged for human review, and the visual packages
  (`outputs/.../phase18_2a_qwen_bbox/run_*/visuals/`) are saved for that review — not
  adjudicated in-session.
- Same-day, same-worker measurement of Qwen/SAM; DINO numbers are frozen references from
  Phase 18.1, not re-measured (identical manifest/dataset, so the comparison is direct).

## 7. ARCHITECTURAL RECOMMENDATION

**Option C — do not use Qwen's direct bbox.** The evidence:

- **A (Qwen → SAM): rejected.** 4.7% recall@0.5; median downstream mask IoU 0.000.
- **B (Qwen → DINO refinement → SAM): rejected.** 50/64 found boxes are wrong-instance or
  panel-scale; refinement cannot fix a wrong instance.
- **C (Qwen → DINO candidates → improved selector → SAM): the only measured recovery path.** It
  is exactly the Phase 18.1/18.2 direction — DINO's candidate *availability* (R@All 89.1%) is
  the sole measured signal that beats chance at specific-instance recovery, and it needs a
  selector signal neither DINO's score nor Qwen's presence-check provides.

Additionally, and independently of the selector: **production should enrich target
descriptions** (per-panel context, appearance, position), since the measured bottleneck is the
instance-resolving information in the description, not the localization mechanics of either
model.

## 8. Final answer

> Может ли Qwen2.5-VL самостоятельно локализовать target достаточно хорошо, чтобы
> использовать его bbox напрямую или как spatial hint для DINO?

**No.** On the same 64 targets, full page, same production description, Qwen direct
localization reaches Recall@IoU≥0.5 = **4.7%** — statistically indistinguishable from DINO's
top-1 (6.2%) and far below DINO's candidate availability (89.1%). Its bbox is wrong-instance or
oversized in 50/64 found cases, so it is neither usable directly nor as a spatial prior
(Qwen bbox → SAM median mask IoU 0.000 vs GT bbox → SAM 0.884). The measured bottleneck is
instance-resolving *description* information, not VLM localization capability.

| signal | value |
|---|---|
| DINO top-1 | 6.2% |
| DINO candidate upper bound (R@All) | 89.1% |
| **Qwen direct bbox recall@0.5** | **4.7%** |
| **Qwen bbox → SAM median mask IoU** | **0.000** |
| GT bbox → SAM median mask IoU | 0.884 |

## 9. What this phase did NOT do

Per the brief: no VLM reranker was built, no VLM ran on DINO candidates, no DINO score added,
no production selector / Grounding DINO / SAM / animation pipeline changed. The coordinate
contract was pinned and unit-tested; the production description (`"character body."`) and
prompt-format output were the only inputs. Phase 18.2 (candidate selection/reranking) must not
start from this experiment's conclusion as if it had measured a reranker — it has not.
