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

**6. Reconstruction stays per-object** (`reconstruct_hidden_region`'s signature and
`_compute_hole_mask` are unchanged, called once per animated object exactly as before). The
multi-object awareness lives entirely in `composite_frame_stack`: an object's reconstructed hole
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
- No real evaluation dataset sample has yet produced a genuine multi-object (PRIMARY +
  SECONDARY/MICRO) plan in a real, observed VLM run — this capability is built and deterministically
  tested, but has not yet been exercised by a real page in this project's own evidence base.
  Flagged, not hidden.
