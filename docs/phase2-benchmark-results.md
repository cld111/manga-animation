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

## Third pass (2026-08-12): VLM fixed, segmentation and inpainting first real runs

Same Kaggle session/environment as the second pass, continued (2x Tesla T4 15360 MiB,
`torch` 2.10.0+cu128, `transformers` 5.0.0). This pass answered the second pass's top open
question and, per the Phase 2.1 continuation brief's priority order, went on to get first
real results for `segmentation` and `inpainting` — the two stages that had zero data before
this pass.

### `vlm`: `device_map="auto"` fixes the OOM, and real inference succeeds

`qwen2.5-vl-7b-instruct` loaded via `Qwen2_5_VLForConditionalGeneration.from_pretrained(...,
device_map="auto")` instead of `.to("cuda")`:

- **Load:** 100.9s (from a warm Hugging Face cache — the weights were already downloaded
  from the second pass's OOM attempt). Sharded automatically: `model.visual` and
  `language_model.layers.0-10` on GPU0, `language_model.layers.11-27` + `lm_head` on GPU1.
- **Peak VRAM:** 9.34 GB on GPU0, 10.74 GB on GPU1 (`torch.cuda.max_memory_allocated`) —
  comfortably within each T4's 15.6 GB, unlike the single-GPU attempt.
- **Real inference, not just loading:** generated a full text response (200 new tokens,
  24.6s) to a free-form "describe motion cues" prompt on a real sample page. Output was
  coherent and on-topic — it correctly identified the page as static (no speed lines, no
  wind-blown hair/cloth), described the background and character accurately, and noted the
  dialogue/chapter-title text.
- **Structured-output test** (the actual acceptance bar — "a structured animation plan
  compatible with the existing project schema"): prompted for one JSON object per candidate
  object with `semantic_label`/`motion_type`/`confidence`/`reason` fields matching
  `ObjectPlan`'s naming and `MotionType`'s enum values exactly. Result, page 1 (24.9s):

  ```json
  {"semantic_label": "eyes", "motion_type": "static", "confidence": 0.9, "reason": "The eyes appear to be looking straight ahead without any indication of movement."}
  {"semantic_label": "mouth", "motion_type": "static", "confidence": 0.9, "reason": "The mouth is closed and there are no visible signs of movement or expression changes."}
  {"semantic_label": "hair", "motion_type": "static", "confidence": 0.8, "reason": "The hair appears to be in place with no visible wind or movement effects."}
  {"semantic_label": "clothing", "motion_type": "static", "confidence": 0.9, "reason": "The clothing does not show any folds or movements that suggest it is animated."}
  {"semantic_label": "background", "motion_type": "static", "confidence": 0.9, "reason": "The background is static with no elements suggesting motion."}
  ```

  Valid JSON, correct field names, correct lowercase enum values, and (per "Static Is a
  Valid Result") an appropriately conservative all-STATIC read of a genuinely static page.
  Repeated on page 6 (28.7s) — also all-STATIC, including a "smoke cloud" object still
  called static despite this page having the richest grounding detections of the sample set
  (12 boxes across face/hand/hair/eye/speech-bubble, see the second pass's table above).
- **Honest limitation, not yet resolved:** every object across both tested pages came back
  STATIC. This is *consistent* with this sample set's already-known gap (no page with an
  unambiguous PRIMARY-motion cue — see the first pass's "Open questions"), so it does not
  yet confirm the model correctly assigns PRIMARY/SECONDARY/MICRO when a real motion cue is
  present. That specific claim needs a page with a genuine drawn motion cue, still untested.
- Fields the schema needs that this prompt didn't ask for (`object_id`, `panel_id`,
  `parent_id`/`children_ids`) were not tested — plausible to add with more prompt
  engineering, not attempted this pass.

**Conclusion:** `qwen2.5-vl-7b-instruct` is a working VLM candidate on this project's 2xT4
Kaggle profile, moving it out of PENDING. It requires 2 GPUs at float16 — a single-T4/L4
profile would need quantization or a smaller model, still untested.

### `segmentation`: first real SAM2.1 result

`sam2.1-hiera-base-plus` via `Sam2Model`/`Sam2Processor` (confirmed to exist in
`transformers` 5.0.0, resolving the adapter's `# VERIFY` note). One real API correction
found: `post_process_masks(masks, original_sizes)` takes no `reshaped_input_sizes` argument
on this version (the adapter's original guess raised `KeyError`) — fixed in
`scripts/phase2_kaggle_benchmark.py`.

Two real box prompts, both taken from this session's actual Grounding DINO output on page 1
(not synthetic placeholders):

| Prompt box | Load (s) | Latency (ms) | IoU scores (3 candidates) | Mask coverage of image | Peak VRAM (GB) |
| --- | --- | --- | --- | --- | --- |
| face (score 0.83) | 10.39 | 535.2 (incl. first-call warmup) | 0.822, 0.895, **0.921** | 1.6% | 9.66 (not isolated — see note) |
| hair (score 0.32) | — (reused loaded model) | 100.6 | 0.888, **0.948**, 0.946 | 11.0% | 0.78 (isolated via `reset_peak_memory_stats`) |

The face row's peak-VRAM figure includes leftover allocations from prior models in the same
session (not reset first) — the hair row's 0.78 GB, measured after an explicit
`torch.cuda.reset_peak_memory_stats()`, is the trustworthy per-call figure. Both prompts
produced a plausible top-scoring mask (IoU ≥0.89) with sensible area — the hair mask covers
a smaller fraction of its (larger) box than a solid face silhouette would, consistent with
hair being a thin/irregular structure rather than a filled region. No pixel-level visual QA
was done this pass (no way to view images from this session) — the "is the mask good enough
not to visibly damage the artwork" question ultimately needs a human/visual check, which
this numeric pass does not substitute for.

**Conclusion:** `sam2.1-hiera-base` is a working segmentation candidate, moving it out of
PENDING. Not yet tested: `sam3`, thin structures under closer scrutiny (single-pixel hair
strands), overlapping/occluded objects, or a full 6-page sweep.

### `inpainting`: first real LaMa result, with an important compositing finding

`lama-large` via the `simple-lama-inpainting` PyPI package (per the candidate's notes in
`configs/benchmark_candidates.yaml`). One real environment issue found and worked around:
importing it *after* `cv2`/`numpy` were already loaded in the same kernel raised
`RuntimeError: empty_like method already has a different docstring` (a numpy/cv2 ABI
conflict from the mid-session `pip install`) — importing it first, in a fresh kernel, avoided
this. Documented in the adapter as an environment note, not a package defect.

Real test: a synthetic rectangular hole (30-55% width, 55-68% height of page 1 — standing in
for a region an object's motion would reveal, per this stage's actual purpose) inpainted
against the real sample page:

- **Load:** 2.13s (downloads the ~196MB `big-lama.pt` checkpoint from GitHub releases).
- **Latency:** 2863.5ms for one hole on one page.
- **Peak VRAM:** 1.16 GB — small, matching ADR 0004's "~50M params" sizing.
- **Important finding:** the raw output is **not pixel-aligned with the input** — a
  1778x1000 source came back 1784x1000 (the model pads to an internal stride). Naively
  substituting the full raw output for the original page would silently violate "Original
  Image Is the Source of Truth" (`docs/architecture.md`): a max pixel diff of 255 (mean 9.0)
  was measured *outside* the intended hole when the raw output was simply resized back to
  the source resolution and compared directly. This is not a LaMa quality problem — it's
  confirmation that `cv-agent`'s compositing step (alpha-blend only the masked hole onto an
  untouched copy of the source, per its ownership section in
  `.claude/agents/cv-agent.md`) is a hard requirement, not a nice-to-have, for this
  candidate specifically.

**Conclusion:** `lama-large` is a working inpainting candidate, moving it out of PENDING.
`aot-inpainting-manga` remains genuinely not runnable (no standard pipeline, adapter still
raises `NotImplementedError`) — correctly PENDING/NOT RUNNABLE, not REJECTED.
`sdxl-inpainting` untested this pass.

## Next steps

- Get a genuine PRIMARY/SECONDARY/MICRO read from the VLM on a page with an actual drawn
  motion cue — every test so far has been on pages that are genuinely (or plausibly) static.
- Visual/qualitative review of the SAM2.1 masks and LaMa fill (this pass only had numeric
  access, no way to render/view images from the session) — the project's own acceptance bar
  ("good enough to not visibly damage the artwork") is fundamentally a visual judgment.
- Try `qwen3-vl-small` and `internvl3-8b` for comparison against Qwen2.5-VL now that at
  least one VLM candidate works end-to-end.
- Benchmark `sam3` (segmentation) and `sdxl-inpainting`.
- Sweep detection thresholds per class (still not done) and test on a second, distinct
  manga series/art style (still only one series tested across all three passes).
- Benchmark `sam3-concept-grounding` and revisit the "does SAM 3 collapse grounding +
  segmentation" architectural question from ADR 0004 once numbers exist.
