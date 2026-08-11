# 5. Phase 2 model selection — status and findings

Status: Proposed — **every required stage now has at least one real, working result**.
`deterministic-animation` and `video-rendering` have local, executed evidence and a
recommended approach. `grounding` has two real remote-GPU passes (n=2, then n=6) with a
leading candidate. `vlm`, `segmentation`, and `inpainting` each now have at least one
confirmed-working real candidate (`qwen2.5-vl-7b-instruct` via `device_map="auto"`,
`sam2.1-hiera-base`, `lama-large` respectively) — see the third pass in
`docs/phase2-benchmark-results.md`. None of the six stages is "finalized" in the sense of
exhaustive candidate comparison, broader sample coverage, or visual QA — see "Open
questions" — but the Phase 2 acceptance bar (at least one working candidate / real benchmark
per required stage) is met for all six.

## Context

Per the Phase 2 brief and [ADR 0004](0004-phase2-model-candidates.md), Phase 2's job is to
benchmark and select one model per stage (`vlm`, `grounding`, `segmentation`, `inpainting`),
plus confirm technical feasibility for the two non-model stages
(`deterministic-animation`, `video-rendering`). Per standing project policy (ADR 0004; see
[CLAUDE.md](../../CLAUDE.md)), model benchmarking runs on a remote Kaggle/Jupyter GPU worker,
never locally, and the assistant must ask the user for the server URL rather than guess or
reuse a stale one — no URL was available for this ADR's first draft, but one was provided
partway through this work and used for the `grounding`/`vlm` runs described below (see
`docs/phase2-benchmark-results.md`'s "Second pass"). `segmentation` and `inpainting` are
still untested — that GPU time ran out before reaching them; see "Open questions".

What this pass produced, all committed to git (the local canonical copy, per ADR 0002):

- `scripts/phase2_kaggle_benchmark.py` — a reproducible adapter implementation for all 11
  shortlisted candidates across `vlm`/`grounding`/`segmentation`/`inpainting`, built on the
  existing model-agnostic harness (`manga_animation.benchmarking`, from Phase 2's first
  commit). This fixes a real reproducibility gap: the first grounding-stage benchmark
  (`docs/phase2-benchmark-results.md`) was run ad hoc in a Kaggle notebook, and only its
  numeric results were committed — the adapter code that produced them was not. The
  `grounding` and `vlm` adapters were exercised for real on a live Kaggle T4x2 session this
  pass (one real bug found and fixed in `GroundingDinoAdapter`, one real OOM found and
  documented in `Qwen25VLAdapter` — see "Stage-by-stage status"); `segmentation`/`inpainting`
  adapters and the `qwen3-vl-small`/`internvl3-8b`/`sam3-concept-grounding` candidates remain
  unexecuted, several still carrying `# VERIFY:` comments where the exact library API could
  not be confirmed against this assistant's knowledge (see "Open questions").
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

**PRIMARY (preliminary): `qwen2.5-vl-7b-instruct`, with `device_map="auto"`.** The first
attempt (`.to("cuda")` onto a single T4) OOM'd — float16 weights alone (14.15 GiB) leave
essentially no headroom on a 14.56 GiB-usable T4, directly contradicting ADR 0004's
desk-research "fits comfortably on a T4/L4" claim. Sharding across this environment's 2x T4s
via `device_map="auto"` fixed it: loads (~101s warm-cache), runs real inference (not just
loading), and produces valid structured output — a JSON-per-object list with
`semantic_label`/`motion_type`/`confidence`/`reason` fields matching `ObjectPlan`'s naming
and `MotionType`'s enum values, correctly defaulting to `static` on genuinely static pages
(see `docs/phase2-benchmark-results.md`'s third pass for the full output and both peak-VRAM
figures, ~9.3 GB / ~10.7 GB across the two GPUs).

**Explicitly not final:** requires 2 GPUs at float16 — untested whether it fits a single
T4/L4 with quantization (int8/int4), which matters for any single-GPU deployment profile.
More importantly, every object across both tested pages came back `static` — this pass has
**not yet confirmed the model correctly assigns PRIMARY/SECONDARY/MICRO** when a real drawn
motion cue is present, because neither tested page had one (a known gap in this sample set
since the first grounding pass). `qwen3-vl-small` and `internvl3-8b` remain untried; either
could be more single-GPU-friendly, but that's a hypothesis, not a result. Run with:
`uv run python scripts/phase2_kaggle_benchmark.py --stage vlm` on the remote worker.

### `grounding` — object grounding

**PRIMARY (preliminary, strengthened by a second pass): `grounding-dino-swin-l`.** Two real
passes now exist (`docs/phase2-benchmark-results.md`):

- **Pass 1** (n=2 pages): markedly faster (551.9ms vs. 3091.3ms mean latency) and lighter
  (1780MB vs. 5150MB peak) than `owlv2-vit-l14` — contradicts the aggregator literature ADR
  0004 cited (OWLv2 reported *faster*).
- **Pass 2** (n=6 pages, broader sample, and a real API fix — `transformers` 5.0.0 renamed
  `post_process_grounded_object_detection`'s `box_threshold` kwarg to `threshold`): mean
  latency 497.6ms, still clearly faster than OWLv2's 2936.6ms at n=6. More importantly, this
  pass **resolved pass 1's biggest open question** — at n=2, only `hand` was ever detected;
  at n=6 with the threshold fix, every prompted class (`face`, `eye`, `hair`, `hand`,
  `speech bubble`) is detected across the sample set. This was a real methodology bug
  (wrong kwarg name silently no-op'ing thresholds in a way that suppressed most detections),
  not a genuine manga-domain weakness — an important correction to how pass 1's qualitative
  finding should be read.

Still **not final**: both passes are the same one MangaDex series (full-color manhwa);
per-class threshold sweeps still haven't happened; `sam3-concept-grounding` still untested.
Treat this PRIMARY as "best evidence so far, now on firmer ground," not settled.

- **FALLBACK: `owlv2-vit-l14`.** Works, consistently ~6x slower and ~1.2x more memory than
  Grounding DINO across both passes; usable if Grounding DINO fails outright on a page, not
  a first choice.
- **PENDING: `sam3-concept-grounding`.** Adapter committed (`# VERIFY:` on exact SAM 3
  class), not yet run — this is also the candidate that could collapse `grounding` +
  `segmentation` into one stage (ADR 0004's architectural note), so it matters beyond just
  filling out the comparison table.

### `segmentation` — pixel-accurate masks

**PRIMARY (preliminary): `sam2.1-hiera-base`.** `Sam2Model`/`Sam2Processor` confirmed to
exist in `transformers` 5.0.0 (resolving the adapter's `# VERIFY` note); one real API
correction made (`post_process_masks` takes no `reshaped_input_sizes` argument on this
version — fixed in `scripts/phase2_kaggle_benchmark.py`). Two real box prompts (taken from
this session's own Grounding DINO output, not synthetic placeholders — "face" and "hair" on
the same page) both produced a plausible top mask (IoU ≥0.89) at sensible latency
(100.6-535.2ms) and low VRAM (0.78 GB, isolated measurement) — see
`docs/phase2-benchmark-results.md`'s third pass for the full table.

**Explicitly not final:** only 2 box prompts on 1 page tested; no pixel-level visual review
was possible this pass (no way to render/view images from the session), so the project's
actual acceptance bar — "good enough to not visibly damage the artwork" — is not yet
verified, only approximated by IoU scores and coverage-fraction sanity checks. `sam3`
untested.

- **PENDING: `sam3`.** Adapter committed, not yet run.

### `inpainting` — hidden-region reconstruction (owned by `cv-agent`)

**PRIMARY (preliminary): `lama-large`.** Real test: a synthetic rectangular hole (standing
in for a region an object's motion would reveal) inpainted on a real sample page via
`simple-lama-inpainting`. Works: 2.13s load (196MB checkpoint), 2863.5ms latency, 1.16 GB
peak VRAM — small, matching ADR 0004's "~50M params" sizing. One real environment issue
found and documented (a numpy/cv2 ABI conflict from installing the package mid-session,
avoided by importing it before `cv2`/`numpy` — not a package defect).

**Important finding, not a disqualifier:** the raw model output is not pixel-aligned with
the input (a 1778x1000 page came back 1784x1000 — internal stride padding). Naively
substituting the full raw output measured a max pixel diff of 255 (mean 9.0) *outside* the
intended hole — this confirms `cv-agent`'s compositing step (alpha-blend only the masked
hole onto an untouched source copy) is a hard requirement for this candidate, exactly as its
ownership section in `.claude/agents/cv-agent.md` already specifies. Not a reason to
downgrade the candidate; a reason the compositing step must not skip mask-based blending.

- **PENDING / NOT RUNNABLE: `aot-inpainting-manga`.** `mayocream/aot-inpainting` has no
  standard `transformers`/`diffusers` pipeline; the adapter deliberately raises
  `NotImplementedError` rather than guess at an architecture (see ADR 0004: license/checkpoint
  format both still TBD). This is a real "cannot currently execute" finding, not a rejection
  after testing — per the Phase 2.1 continuation brief, recorded as PENDING/NOT RUNNABLE.
- **PENDING: `sdxl-inpainting`.** Adapter committed (standard `diffusers` pipeline, high
  confidence it would work), not yet run this pass.

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
- **Grounding, `vlm`, `segmentation`, and `inpainting` (all real runs):** environment
  recorded in `docs/phase2-benchmark-results.md` (GPU, driver, `torch`/`transformers`
  versions, dtype, sample provenance) — including exact error text where something failed
  (the `vlm` OOM, the SAM2 `KeyError`, the LaMa numpy/cv2 `RuntimeError`), so every finding
  is independently verifiable rather than a paraphrased summary.
- `scripts/phase2_kaggle_benchmark.py` records environment metadata (git commit,
  `torch`/`transformers` versions, GPU name/memory, timestamp) automatically into its output
  JSON on every run, so any future run is reproducible from the moment it happens — this was
  the gap the very first grounding pass (pre-ADR-0005) left open, now fixed for every stage.
- The repo now has a real git remote (`origin`, GitHub, public) specifically so the remote
  worker can `git clone`/`git pull` rather than relying on ad hoc code transfer — closing a
  gap ADR 0002 had assumed was already true but wasn't (no remote existed before this ADR).

## Consequences

- No `configs/default.yaml` `model_variants` entries are populated by this ADR yet — every
  stage's PRIMARY is still marked preliminary (single-page or few-page evidence, one
  candidate per stage tried), not a finalized cross-candidate selection.
- Phase 3 (Animation Plan generation from real VLM output) can now start prototyping against
  `qwen2.5-vl-7b-instruct` + `device_map="auto"`, since it has confirmed working structured
  output — but should treat the PRIMARY/SECONDARY/MICRO distinction as unverified until
  tested on a page with a real motion cue.
- The `video-rendering` skill (`.claude/skills/video-rendering/SKILL.md`) gets a short
  addition recording the even-dimensions padding requirement found here.
- `scripts/phase2_kaggle_benchmark.py` is corrected in three places based on this pass's real
  errors: `GroundingDinoAdapter` (`threshold`, not `box_threshold`), `Sam21Adapter`
  (`post_process_masks` takes no `reshaped_input_sizes`), `Qwen25VLAdapter`
  (`device_map="auto"`, not `.to(device)`) — future runs use the fixed calls.
- Any future `cv-agent` compositing implementation must alpha-blend `LamaAdapter`'s (or any
  inpainting candidate's) output through the mask onto an untouched source copy — never use
  it as a full-frame replacement, per the real pixel-alignment finding above.
- This ADR should be superseded once every stage has moved past "one preliminary candidate"
  to a real cross-candidate comparison with broader sample coverage and visual QA.

## Open questions (need further remote GPU time and/or user input)

- Does `qwen2.5-vl-7b-instruct` correctly assign PRIMARY/SECONDARY/MICRO on a page with an
  actual drawn motion cue, or does it default to `static` regardless of what's on the page?
  Neither tested page had an unambiguous motion cue, so this is still unknown.
- Does `qwen2.5-vl-7b-instruct` fit a single T4/L4 with int8/int4 quantization, for
  deployment profiles without 2 GPUs? Untested.
- Exact `transformers`/`diffusers` API for candidates released after this assistant's
  knowledge cutoff (Qwen3-VL, SAM 3) — marked `# VERIFY:` in
  `scripts/phase2_kaggle_benchmark.py`; this pass's three real API corrections (grounding,
  segmentation, vlm) are a preview of the kind of fix these will likely also need.
- `mayocream/aot-inpainting`'s actual generator architecture and checkpoint format — its
  adapter currently raises `NotImplementedError` rather than guess; recorded as
  PENDING/NOT RUNNABLE.
- Visual/qualitative review of SAM2.1 masks and LaMa fills — this pass only had numeric
  access (IoU scores, pixel-diff statistics, coverage fractions) with no way to render or
  view images from the Kaggle session. The project's actual acceptance bar for both stages
  is fundamentally a visual judgment call that numeric proxies only approximate.
  `qwen3-vl-small`, `internvl3-8b`, `sam3`, `sdxl-inpainting`, `sam3-concept-grounding` — all
  committed, none run.
- All real evidence so far is one MangaDex series (full-color manhwa) — a second, visually
  distinct series (e.g. traditional black-and-white manga) has not been tested for any stage.
