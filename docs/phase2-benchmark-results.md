# Phase 2 benchmark results: grounding stage

Live results from the remote Kaggle GPU worker, superseding the desk-research-only claims in
[`docs/decisions/0004-phase2-model-candidates.md`](decisions/0004-phase2-model-candidates.md)
where they conflict. This is a first pass (2 sample pages, 1 object class) meant to validate
the benchmarking harness end-to-end and surface early signal — not a final selection. See
"Next steps" below for what's still needed before the `grounding` stage in `configs/default.yaml`
gets a `model_variants` entry.

> Current cross-stage status (this result's standing as PRIMARY/FALLBACK, plus every other
> stage's status) is tracked in
> [ADR 0005](decisions/0005-phase2-model-selection.md), not here — this file stays a
> point-in-time results record for the grounding stage specifically.

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

## Next steps (partly answered — see the second pass below)

- Broaden the sample set (`scripts/fetch_sample_pages.py --count N`, across more series/tags)
  before drawing a selection conclusion.
- Sweep detection thresholds per class rather than one fixed threshold across `hair`/`face`/
  `hand`/`speech bubble`/`eye`.
- Benchmark `segmentation` and `inpainting` candidates from ADR 0004 the same way.
- Benchmark `sam3-concept-grounding` and revisit the "does SAM 3 collapse grounding +
  segmentation" architectural question from ADR 0004 once numbers exist.

## Second pass (2026-08-12): broader sample set, corrected API, reproducible harness

Run via the now-committed `scripts/phase2_kaggle_benchmark.py` (the first pass above was ad
hoc notebook code; this pass fixed that reproducibility gap — see ADR 0005). Same environment
as above (2x Tesla T4 15360 MiB, `torch` 2.10.0+cu128, `transformers` 5.0.0), same MangaDex
series (*Jeonjijeok Dokja Sijeom* / *Omniscient Reader's Viewpoint*, ch.255, "Full Color" tag),
broadened to **n=6 pages** (up from n=2) via `scripts/fetch_sample_pages.py --count 6`.

**API correction found and fixed:** `GroundingDinoProcessor.post_process_grounded_object_detection()`
no longer accepts `box_threshold` on `transformers` 5.0.0 (confirmed via
`inspect.signature` on the real environment) — it was renamed to `threshold`, matching
OWLv2's processor convention. The first pass's `box_threshold` kwarg (matching the API at the
time) now raises `TypeError`; `scripts/phase2_kaggle_benchmark.py` is fixed and this is the
call signature going forward.

| Candidate | Device | Dtype | Load (s) | Latency mean (ms) | Latency p95 (ms) | Peak mem (MB) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grounding-dino-swin-l | cuda | float32 | — (not re-measured this pass) | 497.6 | 635.9 | — (not re-measured this pass) | ok |
| owlv2-vit-l14 | cuda | float32 | 14.22 (incl. fresh download) | 2936.6 | — | 5994.4 | ok |

Grounding DINO per-image latencies (ms): `685.6, 446.0, 481.5, 487.0, 442.9, 442.9`.

**The `face`/`hair`/`speech bubble` gap from the first pass is resolved** — at n=6 with the
corrected call, Grounding DINO (`threshold=0.25`, `text_threshold=0.2`) detects every prompted
class across the sample set:

| Page | Detected labels |
| --- | --- |
| 1 | face, eye, eye, speech bubble, hair, hair, hand, hair, hair |
| 2 | face, eye, hair, eye, speech bubble |
| 3 | face, hand, hand, eye, eye, hair, face, hair, eye, eye |
| 4 | face, eye, hair, hair, hair, eye, speech bubble, eye |
| 5 | face, hand, hand, hand, hand, hair, hair |
| 6 | hand, eye, face, face, face hand, hair, hand, eye, hair, speech bubble, hair, hair |

This does **not** mean the first pass's "may genuinely be harder on manga/manhwa line art"
open question is fully closed — same series as before (n=6 pages, still 1 series), and mask
*quality* (not just detection) is still untested — but it does resolve the specific "did we
just need more samples/a fixed threshold" question in favor of "yes."

**VLM stage: first real attempt, real failure.** `qwen2.5-vl-7b-instruct` loaded fully to CPU
(729/729 weight shards) but **OOM'd moving to a single T4 at float16**:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 0 has a total
capacity of 14.56 GiB of which 62.81 MiB is free. Including non-PyTorch memory, this
process has 14.50 GiB memory in use. Of the allocated memory 14.15 GiB is allocated by
PyTorch...
```

This directly contradicts [ADR 0004](decisions/0004-phase2-model-candidates.md)'s desk-research
claim that a 7B model "fits comfortably on a T4/L4" — the weights alone (14.15 GiB) leave
essentially no headroom on a 14.56 GiB-usable T4 for activations or KV cache, let alone a
comfortable margin. A `device_map="auto"` attempt (sharding across this environment's 2x T4s)
was started but not completed/confirmed before this pass ended — see ADR 0005's open
questions. Not yet run: `qwen3-vl-small`, `internvl3-8b`, `sam3-concept-grounding`, and the
`segmentation`/`inpainting` stages entirely.

## Next steps

- Confirm whether `device_map="auto"` (2xT4 sharding) or int8/int4 quantization gets
  `qwen2.5-vl-7b-instruct` running at all before treating it as a real VLM candidate.
- Try `qwen3-vl-small` and `internvl3-8b` — plausibly smaller/more T4-friendly.
- Sweep detection thresholds per class (still not done) and test on a second, distinct
  manga series/art style (still only one series tested across both passes).
- Benchmark `segmentation` and `inpainting` candidates from ADR 0004 — zero data so far.
- Benchmark `sam3-concept-grounding` and revisit the "does SAM 3 collapse grounding +
  segmentation" architectural question from ADR 0004 once numbers exist.
