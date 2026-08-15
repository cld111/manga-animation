# Phase 18.2 Results — VLM-Guided Candidate Reranking (5-page diagnostic)

**Question.** If we give the existing production VLM ALL DINO candidates instead of taking
DINO top-1, how much better does it select the correct instance?

**Scope of this run.** Per an explicit scoping decision, this benchmark ran on a **5-page
subset** of the phase-17/18.1 benchmark (ARMS_072, AkkeraKanjinchou_084/090, Arisa_087/088) =
**5 GT targets**, not the full 64. The 5-page results are directional evidence and a mechanism
check; the full 64-target run is the remaining step for a statistically meaningful answer. This
is disclosed here rather than hidden.

**Mechanism.** Reuses the production-compatible VLM verification end to end: the exact
`validate_target` prompt (`_build_verification_prompt`), crop (`_crop_with_margin`, 15%
margin), and parser (`_parse_verification`) applied to EVERY DINO candidate on a page (not just
top-1) with the production Qwen VLM (`qwen2.5-vl-7b-instruct`, float16, 2×T4). GT is used only
after selection, never to influence it. DINO detections are reused from Phase 18.1
(`detections_by_page.json`), so 18.1/18.2 are directly comparable. No production code changed.

**Cost measured (the brief's performance experiment):** 84 unique candidate crops scored;
`total_elapsed` 268.6 s including ~2 min model load → **~4.6 s per VLM call**. Extrapolated to
the full 64-target benchmark (~1013 candidates) this is ~90 minutes of VLM inference -- the
main cost of reranking, as the brief anticipated.

---

## 1. OBSERVED

**Selection Accuracy @1 (selected candidate IoU >= 0.5 with GT), same 5 targets:**

| selector | sel@1 |
|---|---|
| DINO top-1 (Phase 18.1 baseline, same targets) | **0 / 5 (0%)** |
| VLM rerank — A (semantic only: matches, then confidence) | **1 / 5 (20%)** |
| VLM rerank — B (semantic + DINO score, 1:1 blend) | **0 / 5 (0%)** |
| VLM rerank — C (production plausibility gate, then semantic) | **1 / 5 (20%)** |

**Per target:**

| target | DINO top-1 correct | VLM A correct | best available IoU |
|---|---|---|---|
| AkkeraKanjinchou_084_17245 | no | no | 0.52 |
| Arisa_087_43395 | no | no | 0.84 |
| Arisa_088_43463 | no | **yes** | 0.59 |
| ARMS_072_3763 | no | no | 0.58 |
| AkkeraKanjinchou_090_17643 | no | no | 0.92 |

**Recall@K (correct candidate's rank in the VLM-ordered list, eligible targets):** R@1 = 20%,
R@3 = 20%, R@5 = 20%, R@10 = 80% for strategies A and C (R@10 80% for B).

**Error classes (strategy A, provisional):** 4/5 "semantic confusion between multiple
characters" (the VLM answers `matches=True` for several candidates on the page and picks the
wrong one); 1 correct. Category C (correct candidate absent) = 0/5 on this subset.

## 2. INTERPRETATION

- **Directionally positive, sample far too small to conclude.** VLM-A raised sel@1 from 0/5
  (DINO) to 1/5 on the same targets. All 5 targets had a correct candidate available
  (best-available IoU 0.52–0.92), so 5/5 were in principle recoverable -- VLM recovered 1.
- **Adding DINO score cancels the gain (A 20% vs B 0%).** Consistent with Phase 18.1's finding
  that DINO confidence misranks the correct candidate lower; blending it back in reintroduces
  the bad ordering. DINO score should NOT be a ranking input.
- **The dominant failure is semantic confusion:** the production verification prompt
  ("Does the crop show character body?") is satisfied by every character on a page, so the VLM
  picks among several `matches=True` candidates largely by confidence noise. The production
  prompt is a *presence* check, not a *specific-instance* discriminator -- this is the
  measured reason reranking is weak.

## 3. HYPOTHESES (not established by this run)

- On the full 64-target benchmark, VLM-A lands somewhere between 6.2% (DINO baseline) and
  89.1% (availability ceiling); the 5-page signal (20%) does not predict where.
- A *specific-instance* prompt (e.g. naming a distinguishing visual property, or a contrastive
  "which of these crops is THE target character" formulation) would reduce the semantic-
  confusion class more than tuning confidence thresholds. Not measured.
- The `matches` boolean adds little as a ranking signal when almost every candidate matches;
  the `confidence` float carries the (weak) ordering. Not separately measured.

## 4. LIMITATIONS

- **5 targets only** (5 pages) -- the numbers are not statistically meaningful; per the brief,
  no calibration claim is made.
- **Cost:** ~4.6 s/candidate; the full 64-target run is ~90 min of VLM inference. Reranking
  every page's ~20 candidates per object is expensive if applied naively in production.
- The production prompt is a presence check; specific-instance discrimination is exactly what
  it is NOT designed for (see INTERPRETATION).
- Category C was absent on this subset; the reranker's behavior on absent-candidate targets is
  not exercised here.

## 5. NEXT RECOMMENDATION

1. **Run the full 64-target benchmark** (the 5-page run is the mechanism check; the decision
   needs the real numbers). Expected wall time ~90 min on the same worker.
2. Before integrating, **change the selection signal, not the selector plumbing**: the
   production presence prompt cannot discriminate instances. Test a specific-instance /
   contrastive verification prompt (benchmark-only) against the same 64 targets, and drop
   DINO score from the ranking (B already shows it hurts).
3. If a full run still shows sel@1 far below the 89.1% ceiling, the fallback is a
   **multi-stage grounding strategy** (panel-crop grounding + top-K candidate pass) rather than
   betting on VLM reranking alone.

Per the brief: **do not proceed to production integration** based on a 5-target diagnostic.
Run the full benchmark and review first.

## 6. Artifacts

- Results (git-ignored, also saved locally): `outputs/experiments/phase18_2_candidate_reranking/`
  (`report.json`/`report.md`, `per_target_rerank.json`, `vlm_scores_by_page.json`, `run_meta.json`,
  5 visual packages under `visuals/rerank_<sample>.png`).
- Code: `src/manga_animation/benchmarking/phase18/{rerank,run_rerank,report_rerank,visuals}.py`,
  `scripts/run_phase18_2_gpu_benchmark.py`. Tests: `tests/test_phase18_rerank*.py` (11 tests,
  691 total pass, ruff/mypy clean).
- VLM scores are cached per (page, box) so the full run resumes without re-inference.
