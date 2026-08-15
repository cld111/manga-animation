# Phase 18.1 Results — DINO Candidate Recall Benchmark

**Question.** For each of the 64 phase-17 human-annotated `body` targets: *is the correct bbox
present among ALL Grounding DINO detections on its page, and how high is it ranked by DINO's
own confidence score?* This decides between **Case A** (candidate exists, ranking is the
problem -> next step is candidate selection/reranking) and **Case B** (candidate rarely exists
-> next step is candidate generation/grounding).

Diagnostic only: no production code, prompts, thresholds, or gates were changed.

**Date / environment.** 2026-08-15, remote 2× Tesla T4 worker. Grounding DINO `swin-l`
(`IDEA-Research/grounding-dino-base`, thresholds `0.25/0.2`), float32 -- the exact production
client and prompt (`"character body."`). Only DINO ran (no SAM).

**Reproducibility.** Code: `src/manga_animation/benchmarking/phase18/` +
`scripts/run_phase18_1_gpu_benchmark.py`. Results (git-ignored, also saved locally):
`outputs/experiments/phase18_1_candidate_recall/` (`detections_by_page.json`,
`per_target_recall.json`, `report.json`/`report.md`, `run_meta.json`). Reuses the phase-17
manifest/dataset (64 targets, 52 pages, 23 books). Pure recall logic is unit-tested
(`tests/test_phase18_candidates.py`, `tests/test_phase18_run.py`; 680 tests total pass,
`ruff`, `mypy` clean).

**Method.** DINO runs once per unique page (52 calls) with the production prompt; every
detection above DINO's threshold is kept (`max_candidates` is deliberately NOT applied --
recall measures what is *available to select from*, and the cap is a selection step). A
candidate is a "correct match" when its bbox IoU with the GT bbox is >= the threshold. The
"best" correct candidate is the highest-scored one (best rank by DINO confidence). GT is used
only for scoring, never to influence candidate generation or ranking.

---

## 1. Recall@K

| IoU threshold | R@1 | R@3 | R@5 | R@10 | R@20 | R@All | A | B | C |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 6.2% | 23.4% | 34.4% | 60.9% | 78.1% | 89.1% | 4 | 53 | 7 |
| **0.50** | **6.2%** | **23.4%** | **34.4%** | **59.4%** | **78.1%** | **89.1%** | **4** | **53** | **7** |
| 0.75 | 6.2% | 21.9% | 31.2% | 53.1% | 67.2% | 78.1% | 4 | 46 | 14 |

Categories: **A** = correct candidate exists and is top-1; **B** = exists but below top-1;
**C** = no candidate at this threshold.

## 2. Rank of the best correct candidate (IoU >= 0.5)

| rank | count |
|---|---|
| 1 | 4 |
| 2–3 | 11 |
| 4–5 | 7 |
| 6–10 | 16 |
| 11–20 | 12 |
| >20 | 7 |

Median best-correct rank: **8** (max 30).

## 3. Where the correct candidate is absent (category C, 7 targets)

| target | n candidates | best IoU |
|---|---|---|
| Joouari_091_265116 | 7 | 0.000 |
| UltraEleven_110_695541 | 34 | 0.235 |
| MutekiBoukenSyakuma_086_430168 | 9 | 0.074 |
| LoveHina_vol01_086_321521 | 33 | 0.000 |
| HeiseiJimen_101_210720 | 17 | 0.132 |
| MoeruOnisan_vol19_088_413704 | 12 | 0.000 |
| MeteoSanStrikeDesu_107_371490 | 4 | 0.058 |

Four of the seven have no overlapping detection at all (best IoU 0.0); the other three only
partial overlaps below 0.5. These 7 are where candidate generation genuinely fails on this
benchmark.

## 4. OBSERVED FACTS

1. **The correct candidate exists among all DINO detections in 89.1% of targets** (57/64) at
   IoU >= 0.5 (R@All), and in 78.1% even at the strict IoU >= 0.75.
2. **Only 6.2% (4/64) have the correct candidate at top-1.** R@3 = 23.4%, R@5 = 34.4%,
   R@10 = 59.4%, R@20 = 78.1% (IoU >= 0.5).
3. **Category B dominates: 53/64 targets (82.8%) have a correct candidate below top-1.**
   Median best-correct rank is 8; the best correct candidate is usually buried in the
   top-10..top-30 band.
4. **11% of targets (7/64) have no correct candidate at all** (category C); 4 of these have
   zero overlapping detection.
5. **DINO's own confidence does NOT rank the correct candidate first**: for the 53 B-cases,
   the correct candidate's score averages 0.74x the top-1's (mean score gap 0.15; correct
   candidates cluster at score ~0.42, well below the 0.25 threshold but below the wrong
   top-1s). R@1 is driven down by DINO's confidence, not by detection absence.

## 5. INTERPRETATION

- **This is Case A (candidate exists, ranking is the problem), not Case B.** Detection covers
  89% of targets; the failure is that the correct candidate sits below top-1 in 83% of cases.
  A candidate *selector/reranker* operating on the top-K detections is therefore the correct
  direction: top-10 holds a correct candidate for 59% of targets, top-20 for 78%.
- The 11% category-C share is the floor that candidate *generation* must fix independently of
  any reranker (e.g. the small/ambiguous characters in §3).

## 6. HYPOTHESES (not measured in this phase)

- **A reranker with an independent signal can recover most of the 53 B-cases.** Plausible
  because the candidates exist; NOT established, because recovery needs a signal that
  distinguishes the correct candidate from ~20 wrong ones, and DINO's raw score cannot be that
  signal (fact 5). The existing production `validate_target` VLM check is the obvious signal to
  test in Phase 18.2 -- it is currently applied only in rank order and rejected top-1 in most
  of these cases, so measuring "VLM-rerank over top-K" is the direct next experiment.
- **The 89% R@All ceiling depends on the full detection set** (~21 detections/page). If a
  reranker is limited to top-10, the recoverable ceiling is R@10 = 59% unless the signal also
  helps reorder beyond rank 10.
- **Category-C is partly a DINO-small-object/ambiguity issue, not fixable by selection.**
  4/7 C-cases have zero overlap; their small GT boxes (see phase-17 data) may be below DINO's
  effective detection scale.

## 7. NEXT RECOMMENDATION

> **Phase 18.2 — candidate selection / reranking, not candidate generation.** The correct
> target is already present among DINO's detections in 89% of cases but is top-1 in only 6%.
> A reranker is the right next step.

Specifically, Phase 18.2 should:
1. **Rerank the top-K (e.g. 10–20) DINO candidates per page** with an independent signal
   (start with the production `validate_target` VLM check -- the only existing independent
   signal, already applied in production), scoring each candidate and selecting the best,
   then measuring the resulting DINO->SAM->gates IoU against the phase-17 GT.
2. **Report the reranked Recall@K and the end-to-end IoU** on the same 64 targets, so
   Phase 18.1's "recoverable ceiling" is checked against the real selector.
3. **Keep the category-C targets (7/64) out of the reranker success claim** -- they need
   grounding-scale changes, and mixing them in would hide the selector's true performance.

Do NOT start Phase 18.2 automatically; this report must be reviewed first.

## 8. What this phase did NOT do

No production change: Grounding DINO, SAM, thresholds, prompts, candidate cap, validation
gates, VLM planning, animation, or compositing are untouched. No ML reranker was trained.
GT was used only for evaluation.
