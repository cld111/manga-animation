# 9. Evaluation ground-truth immutability and provenance (Phase 3.3.2)

Status: Accepted

## Context

A real, observed incident motivated this ADR: `sample_page_02`, this evaluation dataset's one
confident positive-control sample (`animation_possible: "yes"`, `ground_truth_uncertain:
false`), had its `animation_possible` originally set on the strength of a single real Phase 3.2
session in which the VLM read `PRIMARY: character_hair`. That specific run was later checked by
a direct, non-VLM pixel-diff of the decoded render (frame 0 vs. frame 24): the hair visibly
translated, and the face/speech-bubbles/narration box stayed pixel-identical outside the
animated region. That diff is real evidence that the *rendering* pipeline correctly isolated
and animated whatever region the VLM had chosen — it is not independent evidence that hair is
*the* visually-justified animation target on this page, because no one separately re-examined
the source artwork itself for a real drawn motion cue on the hair, the way e.g.
`eval_static_dialogue`'s STATIC label is grounded in a direct check of the source page against
this project's own STATIC/ANIMATED evidence categories (`analysis/plan_builder.py`'s
`ANALYSIS_PROMPT`).

Two further real, independent sessions — the main Phase 3.3 end-to-end run (commit `7d05bd2`)
and the separate Phase 3.3.1 re-check — had the same VLM read the same page as all-STATIC, in
every call both times (see `docs/phase3.3-results.md`'s "VLM nondeterminism" section). This is
not a hypothetical concern: it is a real, twice-independently-reproduced reversal of a label
this dataset had marked confident and non-uncertain.

Investigating this surfaced the actual architectural gap. `evaluation/metrics.py::
compute_metrics` was already, and remains, structurally correct: it only ever reads ground
truth from `EvalSample` (`evaluation/dataset.py`) and compares it against a `PageRunOutcome`/
`RepeatedRunRecord` (`schemas.py`/`nondeterminism.py`) — separate pydantic models representing
what one real pipeline run actually produced. No code path anywhere in this project writes to
`configs/phase3_3_eval_dataset.yaml`, or constructs an `EvalSample` from live VLM/pipeline
output. So the failure was never "the evaluation code lets a VLM overwrite ground truth" — it
was two narrower, still-real gaps:

1. `EvalSample` was an ordinary mutable pydantic model. Nothing at the type level stopped a
   future function — an evaluation script, a caching layer, a well-intentioned "helper" — from
   mutating a loaded `EvalSample` in place based on a fresh VLM read. The guarantee held only by
   convention (no code happened to do this yet), not by construction.
2. There was no explicit, machine-checkable signal that a specific sample's ground truth had
   been intentionally revised, as opposed to simply reflecting whatever was first written down.
   Git history is a real audit trail, but requires `git log -p` archaeology per sample to use as
   one; nothing in the data itself said "this label changed, and here is why."

`sample_page_02` specifically was a case where ground truth had insufficiently independent
provenance in the first place — it was seeded by a VLM's own semantic classification, and only
mechanically (not semantically) re-confirmed — which is precisely the "VLM as its own oracle"
failure this phase's brief warns against, just one step removed from the obvious form (the code
never literally copied a live VLM answer into the manifest; the manifest's original author did,
once, by hand, treating a lucky single VLM read as if it were independent evidence).

## Decision

1. **`EvalSample` is now `model_config = ConfigDict(frozen=True)`**
   (`src/manga_animation/evaluation/dataset.py`). Any attempted mutation of a loaded instance —
   by a VLM-driven code path or anything else — raises `pydantic.ValidationError` immediately,
   rather than silently succeeding. This is the direct, minimal fix for gap 1 above: the
   immutability the architecture always implicitly relied on is now enforced by construction,
   not by nobody having written the offending code yet.

2. **New `annotation_version: int` field** (default `1`, `ge=1`) on `EvalSample`. Bumped by hand
   only when a sample's ground-truth fields are intentionally revised, alongside a normal git
   commit — it is not a full changelog (git remains the record of *why*), just a fast,
   in-repo, machine-checkable signal that a given sample's annotation is not at its original
   value. `configs/phase3_3_eval_dataset.yaml`'s header comment documents the convention
   directly next to the data it governs, per this project's "prefer versioned data over hidden
   code constants" instinct.

3. **`sample_page_02`'s `animation_possible` revised from `"yes"` to `"uncertain"`**
   (`ground_truth_uncertain` from `false` to `true`, `annotation_version` bumped to `2`). This is
   not a new invented category — it is the exact convention this dataset already established for
   `sample_page_01`'s own cross-session nondeterminism. The sample's `notes` field records the
   full evidence trail (the original single-session basis, the pixel-diff's real but narrower
   meaning, the two later contradicting sessions) so a future human reviewer has everything
   needed to adjudicate it without re-deriving this investigation. `expected_target_category`/
   `expected_motion_category`/`expected_region_note` are nulled (matching `sample_page_01`'s
   existing pattern for uncertain samples), and `regression_reference` is nulled — the previous
   text asserted an all-STATIC read would itself be "a real quality regression," which the
   evaluation harness's actual `_check_regression` logic never enforced (it only flags a
   *completed* outcome as a violation) and which the new evidence no longer supports as a
   confident claim either way.

4. **No change to `compute_metrics`'s logic.** It already treats `EvalSample` as the fixed
   comparison target and `PageRunOutcome`/`RepeatedRunRecord` as the thing being compared
   against it; this ADR strengthens that guarantee at the type level and in documentation, not
   by rewriting metric computation.

## Consequences

- Any future code that tries to "helpfully" cache or refresh ground truth from a live VLM run
  now fails loudly (`ValidationError`) instead of silently corrupting an in-memory `EvalSample`.
  The on-disk YAML was already safe from anything but an explicit file write + commit; this
  closes the narrower but real in-process gap.
- `sample_page_02` is no longer this dataset's confident positive-control case. This is a real,
  disclosed loss — the dataset now has zero samples with both `animation_possible == "yes"` and
  `ground_truth_uncertain == False` that also specify a confident `expected_target_category`.
  Establishing a new one requires actual human adjudication (of `sample_page_02` itself, via
  direct inspection of `examples/sample_page_02.png` for a real drawn motion cue on the hair, or
  of a different page entirely) — explicitly out of this phase's scope, not silently deferred.
- Phase 3.3.1's transform-aware geometric validation (ADR 0008) is untouched and remains
  orthogonal by design: a sample's `animation_possible` (semantic ground truth, this ADR's
  concern) and a specific grounding candidate's `transform_compatible` (geometric safety for one
  transform, ADR 0008's concern) are independent axes. A semantically true-positive sample can
  still fail evaluation because its one real candidate was geometrically unsafe for its assigned
  transform, without that failure implying anything about, or being permitted to change, the
  sample's stored semantic ground truth (`tests/test_evaluation.py::
  test_transform_geometry_failure_does_not_alter_semantic_ground_truth`).
- `EvalSample.model_copy(update={...})` (pydantic's standard frozen-model pattern) is now the
  only in-Python way to derive a modified copy — no call site in this project currently does
  this, and none should for evaluation purposes; a real ground-truth change belongs in
  `configs/phase3_3_eval_dataset.yaml`, reviewed and committed, not constructed at runtime.

## Open questions

- Whether a stronger provenance model is needed — e.g., an explicit
  `annotation_source: Literal["human_visual_inspection", "pixel_diff_of_render", ...]` field per
  sample — is left open. This phase fixes the one real incident at the level its evidence
  actually supports (immutability, versioning, and honestly re-labeling the one contaminated
  sample); generalizing to a provenance taxonomy from a single real data point would be
  speculative schema growth this project's own conventions caution against.
- No live remote GPU worker was available during this investigation (per ADR 0002/0003: remote
  compute is disposable and never guessed at; CLAUDE.md requires an explicit URL before reaching
  one). The brief's Experiment 3 (testing deterministic decoding — `temperature=0`,
  `do_sample=False`, a fixed seed) was therefore not run this phase. The existing real evidence
  already in this repository — three independent real sessions, `Qwen25VLClient.generate()`
  pinning no seed/temperature (`docs/phase3.2-results.md`'s original finding, unchanged) — is
  sufficient to explain the observed instability's mechanism without a new live run, but does
  not rule out other contributing factors (prompt-adjacent sensitivity, a `transformers` version
  change to `generate()`'s default sampling config). Confirming the exact decoding-parameter
  boundary remains future work, not attempted here without a live worker to run it on.
- Whether `sample_page_02` should be re-adjudicated by an actual human reviewer, or retired in
  favor of a different, unambiguous positive-control sample, is left to the user — flagged, per
  this phase's explicit operational rule, not resolved unilaterally.
