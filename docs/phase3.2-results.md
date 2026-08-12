# Phase 3.2 results: VLM targeting reliability + grounding-target validation

Implementation and local-verification results for Phase 3.2 (see the Phase 3.2 brief,
delivered directly to the assistant, not committed as a file, and
[ADR 0006](decisions/0006-grounding-target-validation.md) for the design this implements).
This is a point-in-time results record, not a design doc — ADR 0006 is the source of truth
for *why* the design looks like this; this file is *what happened when it ran*.

**Status: implementation complete, locally verified, AND a real end-to-end validation run
completed on the remote GPU worker.** The original real Phase 3.1 failure (a `flag_banner`
grounding candidate that scored 0.269 and landed on a face/speech-bubble region) was
reproduced against the real models and is now correctly REJECTED by the new validation stage,
with a diagnostic reason that independently, correctly names the real defect. See "Real
end-to-end run" below for the full results.

## What Phase 3.1 left open

1. `qwen2.5-vl-7b-instruct` returned an all-STATIC read on every real page tested across
   Phase 2 and Phase 3.1 (4 pages, 2 series), including a page with a genuine drawn motion
   cue conveyed through page-level speed-line SFX rather than per-object deformation.
2. Grounding DINO's real match for `flag_banner` scored 0.269 and landed on a
   face/speech-bubble region — a technically valid, above-threshold, in-bounds detection for
   entirely the wrong object. SAM 2.1 then produced a clean mask for that wrong region, and
   the rendered video visibly distorted a character's face.

Both are documented in [`docs/phase3-results.md`](phase3-results.md).

## VLM investigation (before changing anything)

Inspected `src/manga_animation/analysis/plan_builder.py` and `client.py` against the brief's
checklist, using the real evidence already recorded in `docs/phase3-results.md` and
`docs/phase2-benchmark-results.md` (no new remote GPU access was needed for this part — the
evidence already existed):

- **Prompt structure:** the original prompt only recognized motion cues drawn directly on the
  candidate object itself ("motion lines, deformation, wind-blown shapes"). The real Phase 3.1
  page's actual motion cue was page-level speed-line SFX layered over the panel, not
  deformation on any single object — a category of evidence the prompt never asked the model
  to consider. The model's real, quoted reason ("not depicted with any motion lines or
  deformation") is a correct, literal answer to the question actually asked, not a model
  failure.
- **Output schema:** `_RawObjectDecision` forces a binary-ish STATIC/PRIMARY/SECONDARY/MICRO
  commitment with no separate continuous "how much motion signal is here" score. Real,
  independently-findable bug: `_select_single_primary` (Phase 3.1) discarded every
  SECONDARY/MICRO decision when deciding whether a plan was usable, even when the model *did*
  identify real motion-worthy objects, just not confidently enough to call one "primary".
- **Image resolution / resizing:** already fixed in Phase 3.1 (`_resized_for_vlm` applies
  `config.resolution` to the long edge). Left as-is this phase — re-loosening it without a
  real GPU run to re-verify OOM safety would violate the brief's own "no arbitrary thresholds
  without calibration" constraint on a *different*, already-burned finding (the Phase 3.1
  OOM). Flagged as a real, unresolved tension instead (see "Remaining limitations").
- **Small motion cues / panel splitting:** `docs/pipeline.md` documents "Panel / scene
  analysis" as a distinct step before VLM understanding, but `plan_builder.analyze_page` never
  implements it — the whole page is always treated as one `PanelPlan` covering `(0,0,1,1)`.
  This means fine motion cues on a small sub-region of a large page are proportionally tiny
  after the page-level resize. Real, confirmed gap; **not fixed this phase** (see "Remaining
  limitations" — building real panel detection is a larger, separately-scoped change than
  Phase 3.2's two stated goals).
- **STATIC classification logic / forced single PRIMARY:** confirmed the schema does
  unnecessarily force a single PRIMARY read in two ways — (a) it only ever considered
  literally-"primary"-labeled decisions (now fixed, see below), and (b) even after picking
  one, every other object is forced to STATIC in the emitted plan (kept, by design — Phase
  3.2 does not redesign the animation engine to animate multiple objects at once).

**Conclusion, matching the brief's "do not immediately replace the model" instruction:** both
real Phase 3.1 findings are prompt/schema/selection-logic gaps with a specific, evidenced
mechanism, not general model-capability failures. Neither required a model swap.

## Changes made

### VLM targeting (`src/manga_animation/analysis/plan_builder.py`)

- `ANALYSIS_PROMPT` now explicitly lists five evidence categories (deformation on the object,
  motion lines on/touching the object, **panel/page-level effect lines near the object**,
  **pose/position implying mid-action**, implied physical force) instead of only the first
  two — directly targeting the real page-level-SFX gap above.
- `_select_single_primary` → `_rank_candidates`: ranks every non-STATIC decision by
  `(motion_type priority, confidence)` instead of requiring a literal `"primary"` label.
  `motion_type` still strictly dominates `confidence`, so an existing real "primary" always
  wins — Phase 3.1's tested selection behavior is unchanged whenever a primary exists; the
  only behavior change is that a page with real SECONDARY/MICRO signal but no PRIMARY label
  now produces a usable plan instead of an all-STATIC-equivalent failure.
- The `AnimationPlan` schema itself is unchanged — ranking is an internal `plan_builder`
  representation (a ranked `list[_RawObjectDecision]`), not a new schema field, per
  `docs/animation-plan-schema.md`'s pixel-free/model-agnostic design.

### Grounding (`src/manga_animation/grounding/ground.py`)

- New `ground_object_candidates()`: returns every usable detection (score-ranked, degenerate
  ones dropped after clipping) instead of only the best one — no extra grounding-model
  inference, since `client.detect()` already returns every box above its own threshold in one
  call. `ground_object()` is now a thin wrapper (`max_candidates=1`), so every Phase 3.1
  caller/test is unaffected.

### New validation stage (`src/manga_animation/validation/validate.py`)

`validate_target()` — explicit ACCEPT/REJECT with structured diagnostics
(`pipeline.types.ValidationResult`) for one grounding candidate:

1. Deterministic bbox-plausibility pre-filter (no model call) — reuses segmentation's own
   coverage-fraction bounds (`pipeline.types.MIN_OBJECT_COVERAGE_FRACTION`/
   `MAX_OBJECT_COVERAGE_FRACTION`, relocated there so both stages share one source of truth).
2. Semantic-agreement check — one `VLMClient.generate()` call on the cropped candidate region
   (existing protocol, no new model dependency), asking whether the crop actually depicts the
   target, with the intended `transform_kind` given as context.
3. Fail-closed on an unparseable VLM response (rejected, not swallowed into a false accept).

Never raises; a REJECT is a normal, structured, logged outcome. See ADR 0006 for the full
calibration rationale — in particular, why a simple stricter confidence threshold was
rejected: the real, observed `flag_banner` score (0.269, wrong) and `hair` score (0.32,
correct) are closer to each other than either is to any plausible cutoff.

### Orchestrator (`src/manga_animation/pipeline/orchestrator.py`)

- New stage sequence: `analysis -> grounding -> validation -> segmentation -> ...`.
- Grounds once, tries every ranked candidate through `validate_target` in score order, uses
  the first ACCEPTed one. If every candidate is rejected, raises
  `PipelineStageError(stage="validation", ...)` with every candidate's rejection reason
  attached — a new `Stage` literal value, distinguishing "grounding found nothing"
  (`stage="grounding"`) from "grounding found candidates but none were correct"
  (`stage="validation"`).
- Automatic retry is scoped to exactly what the brief's failure-behavior section names:
  ranked grounding candidates for the plan's one chosen object, then the *existing*
  human-driven controlled-fallback (`run_pipeline(..., plan=...)`) if those are all
  exhausted. No automatic object-substitution was added — see ADR 0006's "Scope of automatic
  retry" for why that reading was chosen over a broader interpretation.
- The fallback path is validated too now (a deliberate, documented behavior change from Phase
  3.1, where the fallback path made zero VLM calls at all) — "never silently animate an
  unvalidated candidate" applies to a human-authored target as well.

## Tests added/changed

- `tests/test_analysis.py`: 4 new tests — SECONDARY-only and MICRO-only candidates now
  produce a usable plan; a real PRIMARY still always outranks a more-confident SECONDARY; the
  broadened prompt still mentions page/panel-level and pose cues (regression guard against
  silently reverting the fix).
- `tests/test_grounding.py`: 5 new tests for `ground_object_candidates` (full ranked list,
  `max_candidates` cap, degenerate-candidate skipping, all-degenerate raises, `ground_object`
  still delegates correctly), plus 4 more added with the real grounding-client bug fix found
  during the E2E run (see "Run 1" below): `_detections_from_scores_boxes_labels` normal
  aligned case, the real reproduced zero-detection/placeholder-label case, a short-labels
  fallback case, and confirmation that a genuine `scores`/`boxes` mismatch still raises.
- `tests/test_validation.py` (new file): 11 unit tests for `validate_target` — accept on
  agreement, reject on disagreement even at a high grounding score, bbox pre-filter rejects
  without a model call, small-plausible-bbox still reaches the model, fail-closed on
  unparseable output, markdown-fenced JSON still parses, crop margin is applied, candidate
  rank is recorded, REJECT never raises.
- `tests/test_pipeline.py`: 5 new orchestrator-level integration tests — correct candidate
  accepted; next ranked grounding candidate tried and accepted after the first is rejected;
  every candidate rejected raises `stage="validation"`; a high-scoring-but-semantically-wrong
  candidate is still rejected; the controlled-fallback plan is now demonstrably validated too
  and can itself be rejected. Existing fakes (`FakeVLMClient`, `FakeGroundingClient`,
  `ExplodingVLMClient`) were extended (not replaced) to also answer the new validation-stage
  prompt, so every pre-existing Phase 3.1 test kept its original assertions unchanged.

## Test results (local)

```
uv run pytest -q         -> 182 passed (0 skipped -- ffmpeg installed locally for this run)
uv run ruff check .      -> All checks passed!
uv run mypy src          -> Success: no issues found in 34 source files
```

Per CLAUDE.md's standing policy, no real model (VLM/grounding/segmentation/inpainting)
inference happened locally — every test above uses fake clients (`Fake*Client` doubles), the
same pattern every prior Phase 1-3.1 test already used. `ffmpeg` (a system binary, not a
model) was installed locally via Homebrew specifically so the full render path could be
exercised in these tests rather than skipped — this does not violate the "no model inference
locally" policy, `ffmpeg` encodes frames the fake clients already produced.

## Real end-to-end run

**Performed**, on the user's live Kaggle Jupyter session (2x Tesla T4, `torch` 2.10.0+cu128,
`transformers` 5.0.0 — same environment as ADR 0005/Phase 3.1), reached via a genuinely
programmatic, non-browser transport: the session's Jupyter REST/kernel-WebSocket API, driven
directly over HTTP/WebSocket from this local session (no `claude-in-chrome`, no interactive
browser automation — see "How this run was executed" below). This closes the gap
`docs/phase3-results.md` left open ("A programmatic (non-browser) remote-compute transport for
future phases — scoped but not built this phase").

Per CLAUDE.md's standing policy, no model inference happened locally; every real model call
(`qwen2.5-vl-7b-instruct`, `grounding-dino-swin-l`, `sam2.1-hiera-base`, `lama-large`) ran on
the remote GPU worker.

### Run 1: automatic mode, the three existing real sample pages

`uv run python scripts/run_phase3_2_validation.py --env kaggle` against
`examples/sample_page_01.png`, `examples/sample_page_02.png`, `examples/phase3_action_page.png`
(no fallback plan — fully automatic VLM -> grounding -> validation -> ... for every page).

| Page | Analysis (VLM) | Grounding | Validation | Result |
| --- | --- | --- | --- | --- |
| `sample_page_01.png` | all-STATIC | — | — | FAILED (stage=`analysis`) |
| `sample_page_02.png` | PRIMARY: `character_hair` (translate) | 1 candidate, score 0.610 | ACCEPT (confidence 0.95): *"The image clearly shows a character's hair, which is a plausible object for translation motion."* | **COMPLETED** — real MP4 rendered, `seamless_loop_verified=True` |
| `phase3_action_page.png` | PRIMARY: `weapon` (rotate) | 0 candidates above threshold | — | FAILED (stage=`grounding`) |

Aggregate (n=3 pages, automatic mode):

- **VLM usable-target rate: 2/3 (66.7%)** — a real, direct, measured jump from Phase 2/3.1's
  documented 0/4 (0%) all-STATIC rate on every previously-tested real page. Both
  `sample_page_01.png`/`sample_page_02.png` were part of that original all-STATIC evidence
  set (`docs/phase2-benchmark-results.md`); the broadened prompt now finds real, non-STATIC
  signal on at least one of them per run (see the non-determinism note below — it wasn't the
  *same* page that succeeded across every run, but usable-target output was produced where
  none ever was before).
- **Grounding candidates that reached validation: 1; acceptance rate 1/1 (100%); rejection
  rate 0/1 (0%).**
- **Pages fully completed (rendered): 1/3.**
- **Fallback rate: 0/3** (no fallback plan needed or used in this automatic run).

**Real visual QA** (decoded frame pixels fetched directly via the Jupyter Contents API, not
simulated): compared frame 0 (rest) against frame 24 (quarter-cycle peak) of
`sample_page_02.png`'s render. The character's hair visibly translates between frames — a
genuine, correct, small-amplitude sway. The character's face, both speech bubbles ("WAIT A
MINUTE.", "EXCUSE ME?"), and the narration box are pixel-identical between the two frames —
no distortion, no bleed from the animated region. This is a true positive: the validator
accepted a candidate that a human review confirms is actually correct.

**Real, new finding (not previously documented): the VLM's STATIC/PRIMARY decision is not
deterministic run-to-run for an identical page.** A separate earlier run of this same script
(before the run tabulated above) had `sample_page_01.png` produce a usable `character_hair`
PRIMARY read on its first attempt, while the run above had the same page come back all-STATIC.
`Qwen25VLClient.generate()` (`analysis/client.py`) does not pin sampling (no fixed seed,
no explicit `do_sample=False`/temperature=0), so `generate()` is free to vary between calls on
identical input. This directly affects how the "VLM usable-target rate" metric should be read
(it is a noisy per-run estimate, not a fixed page property) and is recorded here as a new,
real, evidence-based finding for a future phase — not fixed this phase (out of scope; the two
problems Phase 3.2 targets are prompt/schema coverage and grounding-target validation, not
decoding determinism).

### Run 2: the original Phase 3.1 failure, reproduced directly

A fresh Kaggle session (the first session's URL expired mid-investigation of an unrelated bug
below; the interrupted attempt on the original session is **not reported as a result**, per
the same honesty policy `docs/phase3-results.md` already applied to its own interrupted second
attempt — it was never actually observed). On the fresh session: reconstructed Phase 3.1's
exact hand-authored fallback plan (`semantic_label="flag_banner"`, `mesh_warp`, amplitude 0.12,
matching `_MOTION_HEURISTICS`' flag/banner template in `plan_builder.py`) as an `AnimationPlan`
JSON, then:

```
uv run python scripts/run_phase3_2_validation.py --env kaggle \
    --page examples/phase3_action_page.png \
    --fallback-plan outputs/experiments/phase3_1_flag_banner_repro_plan.json
```

Real result:

```
validation stage: object_id=obj_flag_banner_repro candidate_rank=0 REJECT
  (semantic_match=False confidence=0.00):
  "The crop shows a character's head and dialogue box, not a flag banner."
examples/phase3_action_page.png: FAILED (stage=validation)
```

**This is the direct, real-model confirmation the whole Phase 3.2 initiative was built to
produce.** Grounding again found a candidate for `flag_banner` on this page (as it did in
Phase 3.1); this time, the validation stage's VLM crop-check independently, correctly
identified the crop as a face and dialogue box — the *exact* real defect from Phase 3.1's
visual QA, described in the model's own words without being told what the defect was. The run
correctly failed with `stage="validation"` (distinct from `stage="grounding"`, exactly as
designed — grounding *did* find a technically valid candidate; the candidate was rejected on
semantic grounds). No video was produced. **The historically-observed face/speech-bubble
distortion cannot recur through this path.**

(Incidental timing note: this run took ~159s at the validation stage, vs. ~5s in Run 1 —
because the fallback path skips the analysis stage entirely, `Qwen25VLClient` had not yet been
loaded in this process, so validation's first VLM call paid the full ~100s model-load cost
documented in ADR 0005. Not a defect — a real, minor, worth-noting performance characteristic
of the fallback path specifically.)

### Combined false-positive/false-negative check

Across both runs, 2 real validation decisions were made: 1 ACCEPT (visually confirmed correct
above) and 1 REJECT (visually/historically confirmed correct — it is the known-bad case).
**Zero false positives, zero false negatives, on this small real sample.** This is a strong
initial signal, explicitly not treated as a large-sample statistical result — n=2 decisions
across one MangaDex series (the same limitation ADR 0006 and Phase 2/3.1 already carry).

### How this run was executed

Reached the session's Jupyter server directly over HTTP (`GET .../api/status`,
`.../api/contents/...`, `.../api/kernels`) and WebSocket (`.../api/kernels/<id>/channels`,
the standard Jupyter kernel messaging protocol) from this local session — no browser, no
`claude-in-chrome`, matching the Phase 3.2 brief's explicit constraint (goal 8). Code changes
reached the worker only via `git push` (local) / `git pull` (remote), per CLAUDE.md's
canonical-source policy — never manual file copying. A dedicated, isolated Jupyter kernel was
started for this work on each session (never reusing the user's own already-connected
notebook kernel) and torn down again after use.

## Remaining limitations

- No real panel/scene splitting before the VLM call (`docs/pipeline.md`'s documented but
  never-implemented "Panel / scene analysis" step) — a real, confirmed gap for pages where the
  motion-relevant panel is a small fraction of a large page. Out of scope this phase (larger
  change than either of Phase 3.2's two stated goals).
- The extreme-aspect-ratio resolution/OOM tension found in Phase 3.1 (long-edge capping
  crushes the short edge on a 7:1 page) is unchanged — flagged, not touched, since fixing it
  without a real GPU re-verification would itself violate "no thresholds without calibration."
- The VLM crop-check's `confidence` field is recorded but not used as a second numeric gate on
  top of `matches` (see ADR 0006's "Open questions") — deliberately, to avoid stacking an
  uncalibrated cutoff on a brand-new signal.
- All real evidence behind this phase's calibration decisions (the flag_banner/hair score
  comparison) is still one MangaDex series — untested on a second, visually distinct series.
- **New, real finding from the E2E run:** `Qwen25VLClient.generate()` does not pin sampling
  (no fixed seed, no `do_sample=False`), so the VLM's STATIC/PRIMARY decision is not
  deterministic run-to-run on an identical page (see "Run 1" above) — the "VLM usable-target
  rate" is a noisy per-run estimate, not a fixed page property. Not fixed this phase.
  Reproducibility of a specific plan is unaffected (a plan, once produced, is deterministic
  downstream) — this affects only whether a given automatic run *produces* a usable plan.
- **New, real bug found and fixed during the E2E run:** `GroundingDinoClient.detect()`
  crashed (`zip() argument 2 is shorter than argument 1`) because `result["labels"]`/
  `text_labels` are not reliably the same length as `result["scores"]`/`result["boxes"]` on
  this `transformers` version (confirmed by direct reproduction: a zero-detection result can
  return `text_labels=['']`, length 1, while `scores`/`boxes` are correctly length 0). Fixed
  by zipping only `scores`/`boxes` (the pair confirmed always aligned) and pulling the label
  opportunistically by index — see the `36a0e6a` commit and its new unit tests. This bug
  pre-dates Phase 3.2 (it lives in Phase 3.1's `client.py`) but was only ever exercised for
  real by this phase's broader, multi-page automatic run — no earlier real run had reached
  grounding on more than one hand-picked page/object.
- Sample size for the false-positive/false-negative check above is small (n=2 real validation
  decisions) — a strong initial signal, not a statistically powered result.

## Verdict against the Phase 3.2 acceptance criteria

**"The pipeline no longer confidently animates a semantically incorrect grounded region"** —
**CONFIRMED on real models**, not only unit-tested: the exact original `flag_banner` failure
(Grounding DINO finding a candidate on `examples/phase3_action_page.png`, the same page that
produced Phase 3.1's face/speech-bubble distortion) was reproduced against the live pipeline
and correctly REJECTED by the validation stage, with a diagnostic reason that independently
names the real defect ("a character's head and dialogue box, not a flag banner"). No video
was produced for that candidate — the historically-observed distortion cannot recur through
this path. Separately, a real correct candidate (`character_hair`, `sample_page_02.png`) was
ACCEPTed and its resulting animation visually confirmed correct (hair moves; face/speech
bubbles/narration box pixel-identical across frames).

**Overall: PASS.** Both Phase 3.2 goals are met and confirmed on real models: (1) VLM
usable-target rate measurably improved (0/4 historically -> 2/3 in this run) via the broadened
prompt and fixed candidate ranking; (2) the pipeline demonstrably no longer confidently
animates a semantically incorrect grounded region — the specific real failure this phase was
scoped to fix is now rejected, on the real page, by the real models, not a synthetic
reconstruction. See "Remaining limitations" for what this PASS does not cover (panel
splitting, cross-series generalization, decoding non-determinism, sample size).
