# 4. Phase 2 model candidate shortlist and benchmark methodology

Status: Superseded by ADR 0005. This document remains the historical shortlist and benchmark
methodology; it is not the active runtime model selection.

## Context

The initial Phase 2 brief deliberately left `model_variants: {}` empty in
`configs/default.yaml`. Later real GPU work established preliminary operational candidates;
the current baseline and remaining selection uncertainty are summarized in
[`docs/current-status.md`](../current-status.md) and ADR 0005.

Per [0003](0003-remote-compute-workers.md), this machine has no CUDA GPU, and per standing
project policy the pipeline is never run locally even as a smoke test — all actual model
loading, inference timing, and quality evaluation happens on a remote Kaggle/Jupyter T4 or
L4 session. This ADR is the work that can be done *before* that session is available:
narrowing the field from "every model that exists" to a short, justified list worth actually
spending GPU time on, plus the criteria that will decide between them.

## Candidate shortlist

Sourced from current (August 2026) model landscape research; see citations at the end.
Sizes/licenses are as reported by each model's own card/repo and should be re-verified at
integration time, not treated as final.

### Stage: `vlm` (panel/scene semantic understanding — `src/manga_animation/analysis`)

| Candidate | Size | License | Why shortlisted |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 7B | Qwen license (Apache-2.0-like, commercial use allowed) | Strong OCR (OCRBench ~888) — manga pages are text-dense (dialogue, SFX); 7B fits comfortably on a T4/L4 |
| Qwen3-VL (smaller variants) | varies, sub-10B options | Qwen license | Newer generation than 2.5; reported to target exactly the "small VLM you can actually run" niche — worth a head-to-head against 2.5-VL-7B rather than assuming newer is better |
| InternVL3-8B/14B | 8B/14B | MIT | MIT is the least restrictive license in this list; InternVL3-78B is close to the open-weight ceiling on MMMU (~72%) so the small variants are worth checking for a quality/license tradeoff against Qwen |

Not shortlisted: 70B+ variants (Qwen2.5-VL-72B, InternVL3-78B, Llama 4 Maverick) — likely
won't fit a single T4/L4 at usable batch size/latency, and per-panel semantic understanding
does not obviously need frontier-scale reasoning. Revisit only if the smaller candidates
fail qualitatively on real manga pages.

### Stage: `grounding` (map Animation Plan objects to image regions — `src/manga_animation/grounding`)

| Candidate | Size | License | Why shortlisted |
|---|---|---|---|
| Grounding DINO (Swin-L) | 218M | Apache-2.0 | Best published accuracy (zero-shot COCO mAP 52.5); grounding correctness matters more than speed here since it runs once per object, not per frame |
| OWLv2 (ViT-L/14) | 428M | Apache-2.0 | ~2.7x faster than Grounding DINO; a reasonable fallback if Grounding DINO's latency is a problem at Kaggle session time limits |
| SAM 3 (concept-prompted mode) | — | SAM license (check commercial terms at integration time) | See architectural note below — may collapse `grounding` + `segmentation` into one model/one stage |

### Stage: `segmentation` (pixel-accurate per-object masks — `src/manga_animation/segmentation`)

| Candidate | Size | License | Why shortlisted |
|---|---|---|---|
| SAM 2.1 | Hiera-B/L variants | Apache-2.0 | Mature, well-documented, box/point-promptable — the safe default `docs/pipeline.md` already names illustratively |
| SAM 3 | — | SAM license | Newer (Nov 2025), text/concept-promptable, reported 2x gain over SAM 2 on concept segmentation. Also reported much slower than lightweight detectors in some configurations — that claim needs verifying on our own actual workload, not taken at face value from an aggregator benchmark |

**Architectural note (not a decision — flagging for the orchestrator/user):** SAM 3 can take
a text concept prompt and directly output detect+segment+track, which overlaps both the
`grounding` and `segmentation` stages `docs/pipeline.md` currently lists separately. If SAM 3
benchmarks well, the pipeline could plausibly merge those two stages behind one model. That's
a pipeline-shape change, not just a model swap, so it should come back as an explicit proposal
after real benchmark numbers exist, not be decided from a literature review.

Anime/manga line art differs enough from the photographic data these models are trained on
(hard black outlines, screentone shading, no photographic texture) that a qualitative check
against real manga pages matters more here than for the other stages — general segmentation
benchmarks say little about line-art performance. No off-the-shelf manga-native segmentation
model was found; if SAM 2.1/SAM 3 underperform on line art specifically, the fallback is
investigating line-art-specific heuristics (e.g. trapped-ball flood-fill segmentation) as a
later, separate decision — not something to build preemptively now.

### Stage: `inpainting` (optional hidden-region reconstruction — `src/manga_animation/reconstruction`)

| Candidate | Size | License | Why shortlisted |
|---|---|---|---|
| LaMa (`lama_large`) | ~50M | Apache-2.0 | Proven at exactly this kind of task in production (manga-image-translator uses it for text-region removal/fill at scale); fast, small, no prompting needed |
| AOT inpainting (manga-tuned) | small | check at integration (SafeTensors community conversion) | Manga-domain-tuned rather than generic-photo-tuned, direct prior art from `manga-image-translator` |
| Stable Diffusion XL inpainting | ~2.6B UNet | CreativeML/SDXL license (check commercial terms) | Higher quality, prompt-guided style matching, for if LaMa/AOT visibly fail on complex screentone/texture regions |

Not shortlisted: FLUX.1 Fill — best published quality, but 12GB+ VRAM and license terms that
need scrutiny for commercial use, for a task (filling small regions a small motion reveals,
per `docs/pipeline.md`) that doesn't need frontier generative quality. Matches the project's
"don't over-engineer" principle ([0001](0001-hybrid-vlm-cv-architecture.md)) — pull it off
the shelf only if LaMa/AOT/SDXL all visibly fail.

## Benchmark methodology

No labeled manga evaluation dataset exists for this project yet, so benchmarking is
deliberately two-tier rather than a single automated score:

1. **Quantitative, automated (per candidate, per stage), run on the remote GPU:**
   latency (ms/panel, mean and p95), peak VRAM/memory, and load time, on both a T4 and an L4
   profile (`configs/kaggle.yaml` dtype/batch settings). Captured as `BenchmarkResult`
   records (`src/manga_animation/benchmarking/schemas.py`).
2. **Qualitative, manual spot-check:** each candidate run against a small set of real manga
   pages (not yet collected — see Open questions) and reviewed for domain fit (OCR accuracy
   on manga text, grounding/segmentation correctness on line art, inpainting artifact-freeness
   on screentone). This is a judgment call recorded as notes on the `BenchmarkResult`, not a
   number — automating manga-domain quality scoring is itself a research problem out of scope
   for Phase 2.

Selection weighs, in this order: (a) fits comfortably in a T4's VRAM at the project's target
`dtype`/`resolution` — a candidate that only works on L4 is a fallback, not a default; (b)
qualitative domain fit on real manga pages; (c) latency; (d) license permits the project's
use. A candidate that wins (a)–(c) but fails (d) is disqualified, not "selected with a
license caveat."

## Open questions (need the remote session and/or user input, not answerable now)

- No sample manga pages exist yet under `examples/` to benchmark against — needed before any
  qualitative pass can run.
- Actual T4/L4 timing and VRAM numbers — everything in this ADR is desk research; the
  shortlist narrows the field, it doesn't pick a winner.
- SAM 3's exact license terms for this project's use case.

## Consequences

- `configs/benchmark_candidates.yaml` encodes this shortlist in a form
  `src/manga_animation/benchmarking` can load — adding/removing a candidate is a config edit,
  not a code change.
- This ADR will be superseded by a follow-up ADR once real benchmark results pick a winner
  per stage and `configs/default.yaml`'s `model_variants` gets populated.
- Nothing here commits to code yet — no model-loading dependencies (`torch`, `transformers`,
  etc.) are added to `pyproject.toml` in this step; that happens once a candidate is actually
  selected, per stage, to avoid installing dependencies for models that lose the comparison.

## Progress

First real benchmark pass (grounding stage, on the remote Kaggle GPU) is in
[`docs/phase2-benchmark-results.md`](../phase2-benchmark-results.md) — it already found one
of this ADR's desk-research claims (OWLv2 being faster than Grounding DINO) doesn't hold on
our actual workload, which is exactly why this ADR treats the shortlist as a starting point,
not a conclusion.

## Sources

- [Best Open-Weight Vision-Language Models 2026 — Presenc AI](https://presenc.ai/research/best-open-weight-vision-language-models-2026)
- [The Best Local Vision Language Models in 2026 — TinyWeights.dev](https://tinyweights.dev/posts/best-local-vision-language-models-2026/)
- [Grounding DINO vs. OWLv2: Compared and Contrasted — Roboflow](https://roboflow.com/compare/grounding-dino-vs-owlv2)
- [SAM 2 vs SAM 3: What Changed — Scematics](https://scematics.io/resource/blogs/sam-2-vs-sam-3-what-changed-data-annotation-2026)
- [SAM 3: Segment Anything with Concepts — Meta AI](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Fast Leak-Resistant Segmentation for Anime Line Art — SIGGRAPH Asia 2024](https://dl.acm.org/doi/10.1145/3681758.3698003)
- [manga-image-translator — GitHub (zyddnys)](https://github.com/zyddnys/manga-image-translator)
- [mayocream/aot-inpainting — Hugging Face](https://huggingface.co/mayocream/aot-inpainting)
