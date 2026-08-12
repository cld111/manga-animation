# Phase 3.2 results: VLM targeting reliability + grounding-target validation

Implementation and local-verification results for Phase 3.2 (see the Phase 3.2 brief,
delivered directly to the assistant, not committed as a file, and
[ADR 0006](decisions/0006-grounding-target-validation.md) for the design this implements).
This is a point-in-time results record, not a design doc — ADR 0006 is the source of truth
for *why* the design looks like this; this file is *what happened when it ran*.

**Status: implementation complete, locally verified. The real end-to-end validation run on
the remote GPU worker has not happened yet** — see "Real end-to-end run" below for why, and
what's needed to complete it.

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
  still delegates correctly).
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
uv run pytest -q         -> 178 passed (0 skipped -- ffmpeg installed locally for this run)
uv run ruff check .      -> All checks passed!
uv run mypy src          -> Success: no issues found in 34 source files
```

Per CLAUDE.md's standing policy, no real model (VLM/grounding/segmentation/inpainting)
inference happened locally — every test above uses fake clients (`Fake*Client` doubles), the
same pattern every prior Phase 1-3.1 test already used. `ffmpeg` (a system binary, not a
model) was installed locally via Homebrew specifically so the full render path could be
exercised in these tests rather than skipped — this does not violate the "no model inference
locally" policy, `ffmpeg` encodes frames the fake clients already produced.

## Real end-to-end run — PENDING

**Not yet performed.** Two things are required that this session cannot supply on its own,
per standing project policy:

1. **A Kaggle/Jupyter server URL.** CLAUDE.md: "Never guess a Jupyter/Kaggle server URL... If
   a task needs to reach an actual remote server and no URL has been given, ask the user for
   it explicitly." None has been given this session.
2. **A non-browser transport.** The Phase 3.2 brief explicitly disallows `claude-in-chrome`/
   interactive browser automation for compute access, and `docs/phase3-results.md` already
   recorded that the interactive-browser transport used for Phase 3.1's real run was removed
   from this project's available tools afterward (`.claude/settings.local.json`'s
   `permissions.deny`) as not the pipeline's intended normal execution path. No programmatic
   replacement (Jupyter REST/kernel API, SSH, or similar) exists yet — this was already a
   known, explicitly out-of-scope gap carried over from Phase 3.1.

`scripts/run_phase3_2_validation.py` (this phase's new entry point) is ready to run on the
remote worker: `uv run python scripts/run_phase3_2_validation.py` against the three existing
real sample pages (`examples/sample_page_01.png`, `examples/sample_page_02.png`,
`examples/phase3_action_page.png`), reporting exactly the metrics the Phase 3.2 brief asks
for (VLM usable-target rate, grounding candidate acceptance/rejection rate, fallback rate,
and a `needs_visual_review` list for manual false-positive checking) into
`outputs/experiments/phase3_2_validation_<timestamp>.json`.

This section will be filled in with real numbers once that run happens — via the standard
`local: push -> remote: pull, run, commit/push if source changed -> local: pull` workflow
(CLAUDE.md), not by fabricating results here.

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
- The real end-to-end validation run itself, see above.

## Verdict against the Phase 3.2 acceptance criteria

**"The pipeline no longer confidently animates a semantically incorrect grounded region"** —
architecturally demonstrated and unit/integration-tested against a reconstruction of the exact
real failure (a high-scoring, in-bounds, semantically-wrong candidate is rejected regardless
of its grounding score — see `tests/test_pipeline.py::test_run_pipeline_rejects_semantically_wrong_candidate_even_at_high_grounding_score`),
but **not yet confirmed against the real, original `flag_banner` failure case on real models**
— that confirmation is exactly what the pending real end-to-end run (above) would provide.

**Overall: implementation PASS, full acceptance PENDING** on the real end-to-end run.
