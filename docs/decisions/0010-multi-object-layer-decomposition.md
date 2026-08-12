# 10. Multi-object layer decomposition (Phase 4/5)

Status: Accepted

## Context

Every phase through 3.3.x deliberately animated exactly one object per page. This was never a
technical ceiling — `analysis/plan_builder.py`'s `_rank_candidates` has ranked every non-STATIC
VLM decision since Phase 3.2 — it was an explicit, documented scope limit:
`build_plan`/`_build_plan_from_panels` forced every decision except the single chosen PRIMARY to
`STATIC` with `motion=None`, even when the VLM itself had proposed real SECONDARY/MICRO motion
for it. `docs/phase3.2-results.md` records this directly: "even after picking one, every other
object is forced to STATIC in the emitted plan (**kept, by design** — Phase 3.2 does not
redesign the animation engine to animate multiple objects at once)."

`README.md`'s own "Planned phases" table names the next phase after 3.3: "**4** | Layer
decomposition refinement, hidden-region reconstruction hardening." `src/manga_animation/
layers/__init__.py` has been a one-line stub since Phase 1 ("Implemented starting Phase 4").
`docs/pipeline.md`'s stage diagram already names "Layer decomposition ... separate each
animated object into an independently transformable layer" as a real stage. This ADR is that
deferred work: the deliberate single-object limit above is the concrete thing "layer
decomposition" unlocks.

No other document in this repository specifies a more detailed design than the one line above —
this ADR is that design, not a transcription of a pre-existing one.

**Honest scope note**: `README.md`'s own phase table lists "Phase 5 | Secondary/micro motion,
multi-object plans" as a *separate*, later phase from "Phase 4 | Layer decomposition
refinement, hidden-region reconstruction hardening." This decision does not cleanly stay inside
only the Phase 4 half. A "layer decomposition" that only ever handles exactly one object isn't
meaningfully decomposing anything — the `Layer` type and multi-layer compositing described
below only become real once more than one object can actually be animated, which is what Phase
5 names. In practice the two were not separable: building a real `Layer` abstraction and
`composite_frame_stack` without also letting `plan_builder`/`orchestrator` produce and process
more than one animated object would have been type/API work with no real capability behind it.
This ADR is therefore Phase 4's "layer decomposition" *and* most of Phase 5's "secondary/micro
motion, multi-object plans" together, not Phase 4 alone — recorded explicitly rather than
silently relabeling either phase's scope in `README.md`'s table. What this ADR does **not**
cover: Phase 4's other named half, "hidden-region reconstruction hardening," beyond the
multi-object *integration* inside `compositing.composite_frame_stack` (see "Decision" point 6)
— `reconstruction.reconstruct_hidden_region`'s own per-object logic is unchanged, and further
robustness work there (mask-touches-edge cases, multiple disjoint holes, inpainting-failure
handling) remains real, undone future work. Real-page evidence that the VLM actually proposes a
genuine multi-object (PRIMARY + SECONDARY/MICRO) plan in practice is also still outstanding —
this capability is built and deterministically tested, not yet observed exercising itself on a
real page in this project's own evidence base (see "Open questions" below).

## Decision

**1. `analysis/plan_builder.py` no longer forces every non-chosen decision to STATIC.** A
decision the VLM itself marked SECONDARY or MICRO now keeps that real `motion_type` and gets a
real `MotionSpec` from the same heuristic table (`_MOTION_HEURISTICS`) the PRIMARY object
already used. Two cases are deliberately unchanged: a decision the VLM marked STATIC stays
STATIC, and an extra PRIMARY-labeled decision that lost to a higher-confidence PRIMARY still
becomes STATIC — `_rank_candidates`' existing "keep the best PRIMARY, defer the rest" policy for
*that* specific case is not being generalized into a new "demote to SECONDARY" rule with no
evidence behind it.

**2. New `Layer` type** (`pipeline/types.py`): `object_id`, `frames: tuple[(ImageArray,
MaskArray), ...]` (one transformed `(image, mask)` pair per frame index, i.e.
`generate_transformed_layer`'s existing per-object output collected across the whole loop), and
`z_order: int`. This formalizes what was already an implicit raw tuple in the single-object
path into a real, named, shared contract between `pipeline/orchestrator.py` and
`compositing`.

**3. Z-order policy: `MotionType` rank, not inferred depth.** PRIMARY composites on top (highest
`z_order`), then SECONDARY, then MICRO; ties within the same `MotionType` break on `object_id`
(lexicographic) for full determinism. Rationale: the analysis prompt's own definition of
"primary" is "the one thing a reader's eye should read as moving on purpose" — keeping it
un-occluded by other animated layers preserves that intended reading. This project has no real
evidence of actual depth/occlusion relationships between objects on a page (no per-object depth
annotation exists anywhere in the schema or dataset), so inferring one would be speculative;
`MotionType` rank is the one ordering signal that's already real, already VLM-produced, and
already meaningful.

**4. `pipeline/orchestrator.py` loops grounding → validation → segmentation → animation over
every non-STATIC `ObjectPlan`, not just the PRIMARY.** The PRIMARY object keeps Phase 3.1's
exact failure policy: if every one of its ranked grounding candidates fails validation, the
whole run fails (`PipelineStageError`, stage="validation"), unchanged. A SECONDARY/MICRO
object's grounding/validation failure does **not** fail the run — it is dropped from rendering
(logged, not silently discarded) and the run proceeds with whatever objects did succeed,
including a PRIMARY-only render if every secondary/micro object failed. Rationale: PRIMARY is
the object the whole plan exists to justify; a secondary enhancement failing to ground is a
real, expected, lower-stakes outcome that this project's own architecture already treats as
optional ("motion that follows from a primary mover... adds life without carrying narrative
weight" — the schema's own SECONDARY/MICRO descriptions), not grounds to fail an otherwise-good
render.

**5. `compositing.composite_frame_stack`** (new, alongside the existing single-layer
`composite_frame`, which is unchanged and still used by nothing new) composites N layers for one
frame index, sorted by `(z_order, object_id)`. The static-region-preservation invariant
generalizes exactly: every pixel not covered by *any* layer's current-frame mask must equal the
source image exactly, at every frame — still a fresh copy of the original per frame, never a
patched running buffer.

**6. Reconstruction stays per-object** (`reconstruct_hidden_region`'s signature is unchanged,
called once per animated object exactly as before — see the "Revision" section below for a real
bug found and fixed in `_compute_hole_mask`'s own formula, independent of this multi-object
change). The multi-object awareness lives entirely in `composite_frame_stack`: an object's
reconstructed hole
pixels are only used at a given frame where (a) that object's own current mask doesn't cover the
pixel, (b) its `hole_mask` says its original position did cover it, and (c) *no other layer's
current-frame mask covers it either* — if a different animated layer is already sitting on top
of what would be the first object's revealed background, that layer paints it correctly and the
reconstruction fill must not fight with it.

## Consequences

- A plan can now legitimately render more than one moving object. Every existing single-object
  test and every real page analyzed so far (7 dataset samples, 4 phases of real runs) still
  produces exactly the same single-PRIMARY-only render it always did, unless the VLM actually
  proposes a real SECONDARY/MICRO candidate alongside a PRIMARY — this is additive, not a
  behavior change for the common case.
- `PipelineRunResult` gains fields for the additional grounded/validated/segmented secondary
  objects (see `pipeline/orchestrator.py`) — the PRIMARY-specific fields it already had are
  unchanged in meaning, so no existing consumer of that type breaks.
- No new schema field was added to `AnimationPlan`/`ObjectPlan` — `motion_type` already
  distinguished PRIMARY/SECONDARY/MICRO/STATIC; this ADR changes what the *pipeline* does with
  that existing distinction, not what the plan itself records.
- `evaluation/` is untouched by this ADR. `PageRunOutcome.primary_semantic_label`/
  `primary_motion_type` still describe only the PRIMARY object, same as before — extending
  evaluation to report on secondary/micro objects too is real future work, not attempted here
  (out of this phase's scope; the repository's own plan places broader evaluation work at Phase
  7, not here).
- This has not been validated against real models on a live GPU worker (no Kaggle/Jupyter URL
  was available during this phase) — see `docs/phase3.3-results.md`'s Phase 4 section for what
  was and wasn't actually run.

## Open questions

- Real depth/occlusion between two drawn objects on a page is not something this project
  measures or has evidence for. The `MotionType`-rank z-order rule is a defensible default, not
  a validated one — if a real page ever shows a visually wrong z-order (e.g. a secondary object
  that should visually pass in front of the primary one), that would be new, real evidence to
  revisit this rule, not something to guess at preemptively.
- Whether a SECONDARY/MICRO object's hole-filling should ever be skipped entirely (not just
  deferred to whichever layer currently covers it) when its own transform never meaningfully
  reveals new background is already handled by `reconstruct_hidden_region` returning `None` in
  that case (unchanged, pre-existing behavior) — not a new question this ADR introduces.
- ~~No real evaluation dataset sample has yet produced a genuine multi-object (PRIMARY +
  SECONDARY/MICRO) plan in a real, observed VLM run~~ — resolved, see the "Revision (Phase 5
  audit)" section below: real multi-object VLM plans are now observed. A related, still-open
  question that section leaves open: no real page has yet produced a *successfully rendered*
  multi-object output (both real multi-object plans found so far share a PRIMARY object whose
  grounding fails for an unrelated reason, before any SECONDARY/MICRO object is reached).

## Revision (reconstruction hardening): `_compute_hole_mask` formula was wrong

Auditing `src/manga_animation/reconstruction/__init__.py` as the second, explicitly-named half
of Phase 4 ("hidden-region reconstruction hardening") found a real, serious, previously-silent
bug in `_compute_hole_mask`, confirmed on real data (not just synthetic fixtures) before being
fixed.

**The bug**: the formula was `original_mask & ~UNION(transformed_masks)` — "the region never
covered by any frame across the whole loop." This is mathematically guaranteed to return an
empty (or near-empty) hole whenever any single sampled frame fully reproduces `original_mask`.
In practice, that condition is *always* true for this project's actual motion model: frame
index 0 (`t_frac=0`) is every `loop_mode="cycle"` motion's rest pose (`analysis/curves.py`,
the seamless-loop convention with the default `phase=0`), and every `TransformKind`'s affine
matrix reduces to an exact identity at `value=0` — confirmed by direct inspection of
`animation/transforms.py::_affine_matrix`/`_mesh_warp_frame`. Frame 0 is always part of the
`transformed_masks` list `pipeline/orchestrator.py` passes in, so the union it contributes to
already reproduces the full original mask almost exactly, making the hole computation vacuous
for every rigid/warp transform kind (confirmed empirically across TRANSLATE, ROTATE, SCALE,
SHEAR, MESH_WARP — only `OPACITY`, which legitimately never moves the mask, correctly has no
hole either way).

**Real-data confirmation**: run against `examples/sample_page_01.png`'s actual hair region (a
~112,000px mask, the project's own real `hair`/`TRANSLATE` heuristic motion) — the buggy
formula computed **zero** hole pixels; the corrected formula computed **70,343** (62% of the
mask). Rendering a mid-swing frame without any reconstruction fill (i.e. exactly what the
buggy `None` result produced in every real run to date) shows a severe, real visual defect: a
hard duplicate-looking seam in the sky where the hair vacated, and corrupted line-art fragments
near the eye/eyebrow where loose hair strands used to be — the exact "ghosting" hidden-region
reconstruction exists to prevent. Filling the corrected hole region (even with an
obvious placeholder color, since no live GPU worker was available to run real LaMa inference)
exactly covers the defect and nothing else — background stars, the face, and the ear outside the
hole are untouched, confirming the hole's shape and the compositing boundary are both correct.
Debug crops are saved locally under `outputs/debug/reconstruction_validation/` (git-ignored,
not canonical, regenerable by re-running the same script against the same real example image).

**Fix**: `_compute_hole_mask` now computes `UNION over frames of (original_mask &
~transformed_masks[i])` (equivalently, `original_mask & ~INTERSECTION(transformed_masks)`) —
a pixel needs a real fill value if *any* frame leaves it uncovered, regardless of whether a
*different* frame re-covers it, because `compositing.composite_frame`/`composite_frame_stack`
blend each frame from its own mask independently. See
`tests/test_reconstruction.py`'s new "Phase 4 reconstruction-hardening" section for the
regression coverage, including the exact case (a region covered by one frame but not another)
that the old formula got wrong.

**What was investigated and found NOT to be a real, separate bug** (per this ADR's own
"prefer the smallest change that closes real correctness gaps" — not fixing what isn't
broken):
- Empty/degenerate (down to 1px) masks: already handled correctly, `np.any(hole_mask)` gates
  the `None` return regardless of mask size; regression-tested.
- Masks touching the image boundary: the hole mask is always exactly the source image's own
  `(H, W)` array — there is no separate geometry that could exceed it; regression-tested.
- Multiple disconnected hole regions: boolean array operations don't assume connectivity, and
  a single `client.inpaint()` call naturally receives and handles a multi-region mask (LaMa and
  similar inpainting models operate on the whole masked image, not per-connected-component);
  regression-tested.
- Thin (down to 1px-wide) hole regions: computed exactly, with no morphological
  erosion/dilation cleanup applied — deliberately; this project has no calibration evidence a
  cleanup step would help more than it would risk shrinking or growing a real hole boundary.
- Reconstruction of one object erasing another visible object's pixels: already prevented by
  `compositing.composite_frame_stack`'s per-pixel "does another layer currently cover this"
  check (cv-agent's original Phase 4 work) — extended with one additional regression test for
  the *partial*-coverage case (only part of a hole covered by another layer), which the
  existing full-coverage-only test didn't exercise.
- Panel-boundary coordinate contracts: not applicable — `reconstruct_hidden_region` operates
  entirely in full-page pixel space and never receives or converts panel-relative coordinates.

**What remains a real, documented, NOT-fixed limitation**: whether the actual inpainting
*content* respects text/line art/speech bubbles it wasn't meant to touch is a model-quality
question this project's deterministic code cannot guarantee or test — the hole is scoped
correctly (see above), but what a real inpainting model paints inside that hole, when the hole
happens to abut fine line art, is not something a unit test can verify without running the real
model. Not observed as broken in this pass's real-data check (the one artifact resembling
damaged line art there was traced to this validation script's own rough color-threshold test
mask bleeding slightly into the face region, not a pipeline defect), but not claimed as
verified either — real LaMa output quality against real, tightly-segmented masks remains
future, live-model validation work.

Corrects the previous "Open questions" bullet above about `reconstruct_hidden_region` returning
`None`: that gating logic (`if not np.any(hole_mask): return None`) is unchanged, but what
feeds it (`_compute_hole_mask`) was wrong until this revision.

## Revision (Phase 5 audit): real multi-object VLM evidence obtained, full render still not observed

A repository-level Phase 5 scope audit (this ADR plus `README.md`'s phase table are the
canonical, agreeing definition — no discrepancy found against `docs/pipeline.md` or the
implementation) confirmed the software described above was already complete and added
regression coverage the existing tests didn't have: `tests/test_pipeline.py` gained three
tests proving `object_id` identity survives grounding → segmentation → animation →
reconstruction without ever cross-associating two objects (`mask(A) -> animation(B)`,
`reconstruction(A) -> layer(B)`) — including one call-argument-level test verified against a
deliberately introduced swap bug (reverted) to confirm it actually fails when that bug is
present, not just when everything already works.

That left exactly one genuinely open item from this ADR's own "Open questions": real evidence
that the VLM proposes a genuine simultaneous PRIMARY + SECONDARY/MICRO plan on an actual page,
not only in deterministic fake-client tests. With the project owner's live Kaggle T4 session
(reached the same way as `docs/phase3.2-results.md`'s "How this run was executed" — Jupyter
REST/kernel-WebSocket API, no browser, no `claude-in-chrome`), `analyze_page` was run 3 times
each against 5 real pages (`examples/sample_page_01.png`, `sample_page_02.png`,
`phase3_action_page.png`, `eval_weapon_effects.png`, `eval_static_dialogue.png`) using the
real `qwen2.5-vl-7b-instruct` client, no fallback plan.

**Result: 6/15 runs across 2/5 pages produced a genuine multi-object plan, reproducibly (3/3
attempts each):**

| Page | Result (3/3 attempts identical) |
| --- | --- |
| `sample_page_01.png` | single-object: PRIMARY `character_hair` only |
| `sample_page_02.png` | all-STATIC |
| `phase3_action_page.png` | **multi-object**: PRIMARY `weapon` (rotate) + SECONDARY `character_clothing` + SECONDARY `flag_banner` + MICRO `character_hair` + MICRO `character_eye` (5 objects, 4 of them real motion) |
| `eval_weapon_effects.png` | **multi-object**: PRIMARY `weapon` (rotate) + SECONDARY `cloth` |
| `eval_static_dialogue.png` | all-STATIC |

This resolves the "Open questions" bullet above: the VLM genuinely does propose real,
simultaneous PRIMARY + SECONDARY/MICRO plans on real pages, not only in synthetic test
fixtures — observed directly, not inferred.

**A further, real, honest limitation found by then running the full pipeline
(`run_pipeline` with real grounding/segmentation/reconstruction clients, not just
`analyze_page`) against both multi-object pages**: both attempts failed before any
SECONDARY/MICRO object was reached, because their shared PRIMARY object (`weapon`) failed at
grounding/validation —

- `eval_weapon_effects.png`: 3 grounding candidates found, all rejected by Phase 3.2's target
  validation (two semantically wrong crops, one geometrically too large to `rotate` safely) —
  `PipelineStageError(stage="validation", ...)`, exactly the strict PRIMARY failure policy
  working as designed.
- `phase3_action_page.png`: 0 grounding candidates above threshold for the prompt `"weapon."`
  — `PipelineStageError(stage="grounding", ...)`, reproducing `docs/phase3.2-results.md`'s
  original finding on this exact page (`0 candidates above threshold`) again, on a fresh
  session.

Both failures are real and pre-existing Grounding DINO limitations with the phrase `"weapon"`
against this manga art style, not a Phase 4/5 multi-object defect — the SECONDARY/MICRO
objects were never even attempted, since the PRIMARY-first ordering (`objects_to_animate`,
`pipeline/orchestrator.py`) correctly fails the whole run before reaching them, per this ADR's
own PRIMARY safety policy. But it does mean **no real page has yet produced a successfully
rendered multi-object output** — every real multi-object plan observed so far shares this same
grounding-blocked PRIMARY. This is new, real, honest evidence, not the same gap as before: the
question is no longer "does the VLM propose multi-object plans" (yes, confirmed), it's "has a
multi-object plan ever rendered end-to-end on a real page" (not yet, blocked by an unrelated,
already-documented grounding limitation, not by anything this ADR's design got wrong). Real
future work — either a different real page whose PRIMARY isn't `"weapon"`-labeled, or
grounding-side improvement for that specific phrase — not attempted here (out of this audit's
scope; `analysis`/`grounding` prompt tuning wasn't this pass's mandate).
