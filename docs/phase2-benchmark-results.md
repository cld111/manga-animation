# Phase 2 benchmark results: grounding stage

Live results from the remote Kaggle GPU worker, superseding the desk-research-only claims in
[`docs/decisions/0004-phase2-model-candidates.md`](decisions/0004-phase2-model-candidates.md)
where they conflict. This is a first pass (2 sample pages, 1 object class) meant to validate
the benchmarking harness end-to-end and surface early signal — not a final selection. See
"Next steps" below for what's still needed before the `grounding` stage in `configs/default.yaml`
gets a `model_variants` entry.

## Environment

- Remote: Kaggle Jupyter session, 2x Tesla T4 (15360 MiB each), driver 580.159.04
- `torch` 2.10.0+cu128, `transformers` 5.0.0, CUDA available, `device=cuda`, `dtype=float32`
- Sample pages: 2 colored interior pages from *Omniscient Reader's Viewpoint* (MangaDex id
  `9a414441-bbad-43f1-a3a7-dc262ca790a3`, chapter 255, English translation), fetched via
  `scripts/fetch_sample_pages.py` (MangaDex "Full Color" tag; not committed to git, see
  `.gitignore` — copyrighted third-party content, re-fetchable on demand).
- Prompt classes: `hair. face. hand. speech bubble. eye.`

## Results

| Candidate | Device | Dtype | Load (s) | Latency mean (ms) | Latency p95 (ms) | Peak mem (MB) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grounding-dino-swin-l | cuda | float32 | 2.98 | 551.9 | 583.0 | 1780 | ok |
| owlv2-vit-l14 | cuda | float32 | 2.27 | 3091.3 | 3096.8 | 5150 | ok |

(Table generated via `manga_animation.benchmarking.report.render_markdown` from
`BenchmarkResult` records built from the run above — the harness written for Phase 2 works
end-to-end against real model output, not just the fake adapter in `tests/test_benchmarking.py`.)

## Qualitative notes

- **Grounding DINO** (`threshold=0.25`, `text_threshold=0.2`): one clean, tightly-localized
  "hand" detection per sample page (scores 0.53, 0.55). `face`/`hair`/`speech bubble` returned
  nothing above threshold on either page.
- **OWLv2** (`threshold=0.1`): five overlapping "hand" boxes across the two pages (scores
  0.13-0.30) — noisier and more redundant than Grounding DINO's single box; would need
  higher threshold and/or NMS tuning before it's usable as-is. Same missing classes
  (`face`/`hair`/`speech bubble`) at this threshold.
- Annotated crops saved to `outputs/debug/phase2-grounding/` (git-ignored, local only).

**Correction to ADR 0004's desk research:** the literature comparison cited there claims
OWLv2 is faster than Grounding DINO (8.5 vs 3.2 FPS). On this actual workload (tall manga
pages, `owlv2-large-patch14-ensemble` specifically) it measured ~5.6x *slower* and ~2.9x
more peak memory than Grounding DINO — a concrete instance of exactly the gap between
aggregator benchmarks and our own workload the ADR flagged as a risk. Published numbers for
one OWLv2 size/config don't transfer to another.

## Open questions this pass does not answer

- Only `hand` triggered detections at these thresholds — `face`/`hair`/`speech bubble` need
  either lower thresholds, different phrasing, or may genuinely be harder for these models
  on manga/manhwa line art vs. photographic training data. Needs more samples and threshold
  sweeps before concluding either way.
- n=2 pages, one manhwa series (full-color, digital-native art) — no traditional
  black-and-white-with-colored-chapter manga tested yet, and no panel with the kind of
  clearly "PRIMARY motion" action (falling object, flag, hair) the project actually cares
  about; this run's pages happened to be a dialogue-heavy phone-screen panel.
- `sam3-concept-grounding` (the third `grounding` candidate in ADR 0004) not yet benchmarked.

## Next steps

- Broaden the sample set (`scripts/fetch_sample_pages.py --count N`, across more series/tags)
  before drawing a selection conclusion.
- Sweep detection thresholds per class rather than one fixed threshold across `hair`/`face`/
  `hand`/`speech bubble`/`eye`.
- Benchmark `segmentation` and `inpainting` candidates from ADR 0004 the same way.
- Benchmark `sam3-concept-grounding` and revisit the "does SAM 3 collapse grounding +
  segmentation" architectural question from ADR 0004 once numbers exist.
