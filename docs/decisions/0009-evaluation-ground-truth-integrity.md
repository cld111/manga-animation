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
- **Resolved, same phase, after a live worker became available**: Experiment 3 (deterministic
  decoding) was run for real on the user's live Kaggle session, in a dedicated kernel, against
  commit `a2296bc`. Result: **`do_sample=False` (true greedy decoding) changed nothing.** The
  checkpoint's own default `generation_config` already carries `temperature=1e-06` — sampling
  noise was never a plausible mechanism, since the default is already numerically indistinguishable
  from greedy. 9/9 calls to `sample_page_02` this session (3 baseline, 3 forced-greedy, 3 more
  after a full model unload/reload) came back all-STATIC; 6/6 calls to `sample_page_01` came back
  identically `character_hair`/PRIMARY/0.9 — both conditions were internally 100% self-consistent
  *within* this session, matching this project's existing finding that within-session repeated
  calls are stable. `torch`/`transformers` versions matched every other documented session
  exactly (`2.10.0+cu128`/`5.0.0`), ruling out a library-version explanation too. **The
  decoding-parameter hypothesis this ADR originally proposed as the leading mechanism is
  therefore experimentally disconfirmed, not just untested.** The real cross-session flip (one
  historical `hair` read vs. now four independent real sessions — Phase 3.3, Phase 3.3.1, and
  two conditions of this run — all reading all-STATIC) remains real and unexplained at the
  decoding-parameter level; the most plausible remaining mechanism is GPU floating-point/kernel
  nondeterminism across physically different hardware allocations or sharding layouts between
  separate Kaggle sessions (a non-reproducible reduction order shifting logits enough to flip a
  near-boundary STATIC/non-STATIC decision on this specific page) — not confirmed, since testing
  it would need the same image run across several *separate fresh Kaggle sessions* in one
  investigation, which this run's single session couldn't provide. Left genuinely open, not
  guessed at further.
- Whether `sample_page_02` should be re-adjudicated by an actual human reviewer, or retired in
  favor of a different, unambiguous positive-control sample, is left to the user — flagged, per
  this phase's explicit operational rule, not resolved unilaterally.

## Revision (Pre-Phase-3.4): `annotation_provenance` field + verified-action samples

This ADR's own "Open questions" originally left a stronger provenance model deliberately
unbuilt: "generalizing to a provenance taxonomy from a single real data point would be
speculative schema growth." A second, real, independent case has since arrived: the project
owner manually placed and personally verified two real images
(`examples/verified_action/action_sample_1.png`, `.../anction_sample_2.png`) as genuinely
containing action/animation, explicitly *not* via VLM inference or by inspecting a pipeline's
own rendered output — the exact failure mode `sample_page_02`'s original annotation fell into.
Two distinct, real provenance stories (one contaminated, one clean) is enough evidence to justify
a minimal structural field, not a taxonomy built ahead of need.

**Decision**: `EvalSample` gains `annotation_provenance: Literal["independent_human_verification"]
| None = None`. `None` (the default, and the value every pre-existing sample keeps) means "no
structured provenance recorded" — it is deliberately **not** backfilled onto
`sample_page_01`/`sample_page_02`/`phase3_action_page`/`eval_static_dialogue`/
`eval_weapon_effects`, since asserting a specific provenance category for them now would be a new
historical claim this project cannot actually verify after the fact; their real provenance
remains exactly where Phase 3.3.2 already put it — free text in `notes`. Only the two new samples,
whose provenance is genuinely known with this level of confidence right now, get the field set.

`EvalSample.fetch_script` also becomes `str | None` (was required, `min_length=1`): the verified
samples have no reproducible fetch mechanism at all — a different, permanent kind of gap from
`sample_page_01`/`sample_page_02`'s "has a fetch script, but the underlying query isn't
reproducible" gap — and `None` represents that honestly instead of inventing a placeholder script
path.

**`load_eval_dataset` gains one new invariant**: two samples may never declare the same
`image_path` (raises `ValueError`) — a cheap, filesystem-independent check that directly guards
the "do not duplicate the same image under multiple semantic identities" requirement this
integration was built under. It does not hash image bytes (that would require every referenced
image to exist on disk just to load the manifest schema, breaking the existing image-free
manifest-validation path), so a byte-identical file under a different path and a different
`image_path` string would not be caught — a real, narrower scope than a full content-duplicate
check, chosen deliberately over adding a filesystem dependency to schema loading.

**What independent verification does and does not establish**: exactly `animation_possible:
"yes"` — nothing else. `expected_target_category`/`expected_motion_category`/
`expected_region_note` stay `null` on both new samples, the same "yes, but target genuinely
open" pattern `phase3_action_page` already established — not a new kind of gap. No new field was
added to represent a target/bbox/transform expectation for these samples; none of that was
independently verified, so none of it is recorded.

**Evaluation reporting**: no new `EvaluationReport` field. The 3-way split Pre-Phase-3.4 asks for
(verified positive controls / verified negative controls / unresolved) is already fully derivable
from existing fields — `semantic_false_negative_rate.denominator` (resolved "yes" samples),
`semantic_false_positive_rate.denominator` (resolved "no" samples), and
`unresolved_ground_truth_count` — so `scripts/run_phase3_3_evaluation.py`'s report output was
extended to print this split explicitly, reusing those three numbers rather than building a
second parallel metric system.
