# 5. Phase 2 model selection — status and findings

Status: Proposed — **partial**. `deterministic-animation` and `video-rendering` have local,
executed evidence and a recommended approach. `grounding` has one real (if limited) remote
GPU pass with a leading candidate, not yet finalized. `vlm`, `segmentation`, and `inpainting`
have **no executed benchmark evidence** — adapter code is written and committed, but nothing
has run, because no Kaggle/Jupyter GPU session was available during this pass (see "Open
questions"). This ADR will be superseded once those three stages have real results.

## Context

Per the Phase 2 brief and [ADR 0004](0004-phase2-model-candidates.md), Phase 2's job is to
benchmark and select one model per stage (`vlm`, `grounding`, `segmentation`, `inpainting`),
plus confirm technical feasibility for the two non-model stages
(`deterministic-animation`, `video-rendering`). Per standing project policy (ADR 0004; see
[CLAUDE.md](../../CLAUDE.md)), model benchmarking runs on a remote Kaggle/Jupyter GPU worker,
never locally, and the assistant must ask the user for the server URL rather than guess or
reuse a stale one. No URL was available during this pass, so the four model-benchmarking
stages are gated on that; see "Open questions" below.

What this pass *did* produce, all committed to git (the local canonical copy, per ADR 0002):

- `scripts/phase2_kaggle_benchmark.py` — a reproducible adapter implementation for all 11
  shortlisted candidates across `vlm`/`grounding`/`segmentation`/`inpainting`, built on the
  existing model-agnostic harness (`manga_animation.benchmarking`, from Phase 2's first
  commit). This fixes a real reproducibility gap: the first grounding-stage benchmark
  (`docs/phase2-benchmark-results.md`) was run ad hoc in a Kaggle notebook, and only its
  numeric results were committed — the adapter code that produced them was not. Nothing in
  this script has been executed; several candidates (Qwen3-VL, SAM 3, InternVL3, AOT
  inpainting) carry explicit `# VERIFY:` comments where the exact library API could not be
  confirmed against this assistant's knowledge (see "Open questions").
- `scripts/phase2_cv_feasibility.py` — executed locally (CPU, real sample manga page).
  Confirms all six `TransformKind`s are implementable via OpenCV/NumPy and that the
  project's two hard invariants (static-region pixel preservation, seamless-loop math) hold
  numerically. See "deterministic-animation" below.
- `scripts/phase2_video_feasibility.py` — executed locally (CPU, no GPU/ML involved).
  Confirms H.264 encoding of a real frame sequence produces a valid, correctly-timed,
  loop-continuous file, and surfaced one real implementation requirement (even
  width/height for `yuv420p`) not previously documented. See "video-rendering" below.

## Stage-by-stage status

For each stage: PRIMARY / FALLBACK / REJECTED per the Phase 2 brief's definitions, or
**PENDING** where no candidate has actually been run yet — a candidate that hasn't been
benchmarked is not "rejected," it's simply unevaluated, and this ADR does not pretend
otherwise.

### `vlm` — semantic manga/page analysis

**PENDING — no candidate benchmarked.** Adapters committed for all three shortlisted
candidates (`qwen2.5-vl-7b-instruct`, `qwen3-vl-small`, `internvl3-8b`); none executed.
Run with: `uv run python scripts/phase2_kaggle_benchmark.py --stage vlm` on the remote
worker. `qwen3-vl-small`'s adapter reuses Qwen2.5-VL's chat-template shape as a starting
point (`# VERIFY:` on the model class) and `internvl3-8b`'s adapter's exact `.chat()` call
is a placeholder pending its `trust_remote_code` model card — both need first-run
correction, not just first-run timing.

### `grounding` — object grounding

**PRIMARY (preliminary): `grounding-dino-swin-l`.** One real pass exists
(`docs/phase2-benchmark-results.md`, Kaggle T4x2, 2 sample pages): markedly faster
(551.9ms vs. 3091.3ms mean latency) and lighter (1780MB vs. 5150MB peak) than
`owlv2-vit-l14` on this actual workload — notable because it contradicts the aggregator
literature ADR 0004 cited (OWLv2 reported *faster*). Explicitly **not final**:
`docs/phase2-benchmark-results.md`'s own "Next steps" call for a broader sample set,
per-class threshold sweeps, and a `sam3-concept-grounding` pass before concluding — none of
which happened this pass (no GPU access). Treat this PRIMARY as "best evidence so far,"
re-run `scripts/phase2_kaggle_benchmark.py --stage grounding` before treating it as settled.

- **FALLBACK: `owlv2-vit-l14`.** Works, but 5.6x slower and 2.9x more memory on this
  workload; also noisier at the tested threshold (5 overlapping boxes vs. Grounding DINO's
  1 clean box) — usable if Grounding DINO fails outright on a page, not a first choice.
- **PENDING: `sam3-concept-grounding`.** Adapter committed (`# VERIFY:` on exact SAM 3
  class), not yet run — this is also the candidate that could collapse `grounding` +
  `segmentation` into one stage (ADR 0004's architectural note), so it matters beyond just
  filling out the comparison table.

### `segmentation` — pixel-accurate masks

**PENDING — no candidate benchmarked.** Adapters committed for `sam2.1-hiera-base` and
`sam3` (`# VERIFY:` on exact SAM2/SAM2.1 class name). Both adapters currently prompt with a
placeholder synthetic box (no real grounding output is wired in yet) — timing-only until a
real grounding candidate is selected and its output can be piped in.

### `inpainting` — hidden-region reconstruction (owned by `cv-agent`)

**PENDING — no candidate benchmarked.** `lama-large` and `sdxl-inpainting` adapters are
committed and should run as-is. `aot-inpainting-manga`'s adapter deliberately raises
`NotImplementedError` — `mayocream/aot-inpainting` has no standard `transformers`/`diffusers`
pipeline, and writing a guessed generator architecture would be worse than an honest
placeholder (see ADR 0004: license/checkpoint format both still TBD for this candidate).

### `deterministic-animation` — CV implementation feasibility

**PRIMARY: OpenCV/NumPy, executed and verified locally (CPU).**
`cv2.warpAffine`/`cv2.getRotationMatrix2D` for `translate`/`rotate`/`scale`/`shear`,
`cv2.remap` with a smooth displacement field for `mesh_warp`, alpha blend for `opacity`.
Real run against `examples/sample_page_01.png` (800x2305):

- All six `TransformKind`s produce bit-exact static-region preservation (0 pixel
  difference) — **once "the static region" is defined as exactly `alpha == 0`, not an
  arbitrary mask-value threshold**. An earlier version of this check used `mask < 8` as a
  proxy for "outside" and reported false failures (max diff up to 7/255) purely from
  interpolated mask edge values in the 1-7 range being wrongly excluded from the
  intentionally-blended region. This is a real, worth-recording implementation note for
  Phase 5: the compositing code's "untouched" check must use the transformed alpha's actual
  zero/nonzero boundary, not a heuristic cutoff.
- Seamless-loop math (integer-speed sinusoid) is bit-exact: frame 0 vs. a freshly-rendered
  frame at `t = duration_s` diff to zero.
- CPU cost: ~20-33ms/frame at full source resolution (800x2305, above the pipeline's
  default 1536px working resolution) on this dev machine (Apple M1 Max) — a first real
  baseline, not yet measured at the actual configured working resolution or with real
  (non-elliptical-placeholder) masks.

No FALLBACK/REJECTED framing applies here — this is a classical-CV feasibility check, not a
model comparison; the "selection" is which OpenCV primitive maps to which `TransformKind`,
which is what the check confirms.

### `video-rendering` — encoding/output

**PRIMARY: ffmpeg, `libx264`, `yuv420p`, `crf 18`, `preset medium`, `+faststart` —
executed and verified locally (CPU, no ML/GPU involved).** Real encode of the 96-frame
sequence `scripts/phase2_cv_feasibility.py` produced:

- Output demuxable, fps and frame count match the source exactly.
- Loop continuity at the wrap point (last decoded frame -> first) has the same order of
  magnitude as an ordinary adjacent-frame step (mean abs diff 0.54 vs. 0.50) — no
  discontinuity introduced by H.264 encoding.
- **Finding, now recorded in the `video-rendering` skill:** `yuv420p` requires even
  width/height; this run's real sample page (800x2305) has an odd height and the encode
  failed outright (`height not divisible by 2`) until a `pad=ceil(iw/2)*2:ceil(ih/2)*2`
  filter was added. Manga pages are not guaranteed to have even dimensions — the rendering
  stage must pad (not crop, to avoid losing source content) before encoding.
- `ffmpeg` itself is confirmed **absent** on this local dev machine — validated instead via
  `imageio-ffmpeg`'s sandboxed vendored binary, a validation-only convenience (see the
  script's docstring). Production `video-agent` code still must depend on and check for a
  real system `ffmpeg`, per its existing documented constraint.

## Reproducibility

Every executed result above records, per the Phase 2 brief's reproducibility requirements:

- **CV/video feasibility:** OpenCV version, device (`cpu`), source page path/checksum
  (implicit via git-ignored `examples/`, re-fetchable via `scripts/fetch_sample_pages.py`),
  fps/duration/resolution, and full command — see `outputs/experiments/phase2_cv_feasibility.json`
  and `outputs/experiments/phase2_video_feasibility.json` (git-ignored, regenerable by
  re-running the two scripts; see ADR 0002's "Remote Compute Is Disposable" — the same
  reasoning applies to any generated artifact).
- **Grounding (existing pass):** environment recorded in `docs/phase2-benchmark-results.md`
  (GPU, driver, `torch`/`transformers` versions, dtype, sample provenance).
- **Not-yet-run stages:** `scripts/phase2_kaggle_benchmark.py` records environment metadata
  (git commit, `torch`/`transformers` versions, GPU name/memory, timestamp) automatically
  into its output JSON on every run, so the *next* real run is reproducible from the moment
  it happens — this was the gap the first grounding pass left open.

## Consequences

- No `configs/default.yaml` `model_variants` entries are populated by this ADR — nothing
  here is final enough to select, per its own "PENDING" markers above.
- Phase 3 (Animation Plan generation from real VLM output) cannot start until `vlm` moves
  out of PENDING — it has zero executed evidence.
- The `video-rendering` skill (`.claude/skills/video-rendering/SKILL.md`) gets a short
  addition recording the even-dimensions padding requirement found here.
- This ADR should be superseded (not edited into a false "Accepted") once `vlm`,
  `segmentation`, and `inpainting` have real remote-GPU results and `grounding` has the
  broader validation pass `docs/phase2-benchmark-results.md` already called for.

## Open questions (need the remote GPU session and/or user input)

- **A Kaggle/Jupyter server URL** — not guessed or reused per ADR 0003/CLAUDE.md; needed to
  run `scripts/phase2_kaggle_benchmark.py` for `vlm`, `segmentation`, `inpainting`, and to
  extend `grounding` per `docs/phase2-benchmark-results.md`'s "Next steps."
- Exact `transformers`/`diffusers` API for candidates released after this assistant's
  knowledge cutoff (Qwen3-VL, SAM 3) — marked `# VERIFY:` in
  `scripts/phase2_kaggle_benchmark.py`; first real run on each will need small corrections,
  not a rewrite.
- `mayocream/aot-inpainting`'s actual generator architecture and checkpoint format — its
  adapter currently raises `NotImplementedError` rather than guess.
- Real segmentation/inpainting timing is currently only meaningful against a placeholder
  box/mask (no grounding candidate is selected yet to supply a real one) — re-run once
  `grounding` is finalized.
