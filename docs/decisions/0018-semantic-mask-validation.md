# 18. Post-segmentation semantic mask validation (Phase 12)

Status: Accepted

## Context

Phase 11 ([`docs/phase11-results.md`](../phase11-results.md), section 6.4) confirmed, via a
live GPU capture of 12 real `SegmentationResult.mask` arrays across 3 real pages, a systemic
failure mode no earlier phase had evidence for: **SAM 2.1 can produce a mask that is
geometrically unremarkable — it passes every existing check (the bbox coverage bounds shared
by `validation/validate.py`/`segmentation/segment.py`, Phase 8.3's edge-asymmetry test, Phase
8.3's cross-object overlap guard) — but is semantically wrong**, capturing substantially more
or different real content than its assigned `semantic_label`:

| Object | Labeled as | Real mask actually covers |
| --- | --- | --- |
| `villainess_ending_scuffle` / `obj_cloth_5` | cloth | cloth + a full speech bubble + a hand |
| `wind_breaker_finish` / `obj_object_in_motion_12` (PRIMARY) | "a bicycle wheel in motion" | an incoherent vertical stripe through face, wheel, jersey, hand |
| `wind_breaker_finish` / `obj_character_hair_7` | character_hair | the character's entire face |
| `sss_hunter_gladiator` / `obj_character_hair_7` | character_hair | a creature's head + background drape |

Phase 11 tested four candidate purely-geometric signals (mask fragmentation, bbox density,
aspect ratio, convex-hull solidity) against these 4 confirmed-defective masks plus 8 real
non-defective ones from the same GPU capture — **none separated them cleanly** (section 7):
the worst-solidity mask (the PRIMARY wheel defect above) sits within 0.05 of a real
non-defective object, and a real non-defective object has the *highest* solidity of all 12,
above three of the four defects. Phase 11's own conclusion: "the underlying failure is
semantic (does this mask's content match its label), not geometric (does this mask's shape
look unusual) — no geometry-only check can be expected to catch it reliably," and named the
concrete direction for a future phase: "a post-segmentation semantic re-validation step (a
second, cheap VLM crop-verification call against the actual mask's silhouette/crop... checking
'does this exact masked region look like it's *entirely* the target object')."

This ADR is that follow-up.

## Why this is a distinct stage from `validation/validate.py`

`validate.py::validate_target` (ADR 0006) already runs a VLM crop-verification call, but on
the **grounding bbox**, **before segmentation runs** — its own docstring is explicit that no
mask exists yet at that point in the pipeline. It answers "is this box a plausible location
for the target," not "does this exact mask's content match the target." The two questions are
genuinely independent: a bbox can be a perfectly reasonable location for "cloth" while SAM's
own mask, given that box, still bleeds into an adjacent speech bubble and hand (exactly the
`obj_cloth_5` case above — `validate_target` accepted this candidate for real on Phase 11's own
GPU run, correctly, since the box itself was fine).

This is the same reasoning ADR 0008 already used to justify a second, distinct check
(transform-geometry) rather than folding it into `validate_target`: different pipeline point,
different available evidence (geometry there, mask content here), different question. Workstream
14's idealized stage order ("segmentation → geometric validation → semantic validation →
transform-aware validation → rendering") does not exactly match this codebase's stage order —
transform-aware geometric validation is necessarily pre-segmentation here (it validates a bbox,
which exists before a mask does), while this new semantic *mask* validation is necessarily
post-segmentation (it needs the real mask to exist). The actual stage order is:

    analysis -> grounding -> validation -> segmentation -> mask_semantics -> animation
    -> reconstruction -> compositing -> rendering

## Decision

Add `src/manga_animation/validation/mask_semantics.py`, a new stage (`Stage` literal value
`"mask_semantics"`) between segmentation and animation.

1. **Method chosen: VLM mask-crop verification ("Method B").** A margin-padded crop around the
   mask's tight bbox, with everything *outside* the mask dimmed to 35% brightness (so the VLM
   sees exactly what the mask covers while still seeing enough context to judge whether it
   reached into something it shouldn't have), sent to the same `VLMClient` protocol
   `validate.py` already uses — no new model dependency. The prompt asks whether the
   highlighted region shows *only* the target object, and (if not) what else it includes.

   Alternatives considered and rejected on the evidence above, not by assumption:
   - **Geometric-only re-thresholding** (Phase 11's Method C/E precursors) — directly
     disconfirmed by Phase 11's own 12-object real dataset (section 7); repeating it here
     with a different geometric formula, absent new evidence a different formula would
     behave differently, would repeat the exact mistake ADR 0006 already argues against
     ("do not make arbitrary confidence thresholds without calibration/evidence").
   - **CLIP-style crop/text embedding similarity ("Method A")** — would introduce a new model
     dependency this project has never benchmarked (ADR 0004/0005's shortlist doesn't include
     one), and per CLAUDE.md's compute-locality policy any new real model inference belongs on
     the remote GPU worker, batched with other real experiments, not adopted speculatively.
     Left as a documented, unimplemented candidate for a future dataset-expansion phase (see
     `docs/phase12-results.md`'s benchmarking section) rather than built without real
     comparative evidence it would help.

2. **Three-way verdict, not binary** (`MaskSemanticVerdict = "accept" | "reject" | "abstain"`):
   a VLM confidence read landing in `[0.4, 0.6]` — a documented, evidenced-but-NOT-
   statistically-calibrated band, the same status as this codebase's other thresholds (e.g.
   `transform_geometry.py`'s bounds) — abstains rather than forcing a binary call the evidence
   doesn't support. Only 12 real labeled objects exist for this method at design time (4
   confirmed-bad, 8 presumed-good); this is explicitly too small to statistically calibrate a
   numeric threshold, and this ADR does not pretend otherwise.

3. **Fail-closed policy, identical to every existing stage's**: PRIMARY REJECT/ABSTAIN raises
   `PipelineStageError(stage="mask_semantics", ...)`, failing the whole run — "a clean honest
   REJECTED is preferable to a visually corrupted PASS" (Phase 11's own framing). SECONDARY/
   MICRO REJECT/ABSTAIN drops the object via `DroppedObjectResult(failing_stage="mask_semantics")`
   without failing the run, matching grounding/validation/segmentation's existing non-fatal-drop
   policy for non-PRIMARY objects.

4. **Geometric signals are still computed and attached to every result, but never gate the
   decision.** Phase 11's own four signals (fragmentation, density, aspect ratio, solidity) are
   recomputed here as `MaskSemanticResult.geometric_signals` for forensic/explainability value
   (Workstream 8/9) — a future phase with materially more labeled real data might find a
   combination that adds value on top of the VLM signal (Method E), but nothing in this phase's
   evidence supports gating on them today.

5. **`PipelineConfig.enable_semantic_mask_validation`** (default `True`) lets a caller who has
   independently characterized this gate's real false-rejection rate for their own dataset
   disable it deliberately, rather than hardcoding it unconditionally — the same kind of
   escape hatch `analysis_mode` already provides for a different behavior change (ADR 0017).

## Consequences

- New `Stage` literal value `"mask_semantics"`; new `pipeline.types.MaskSemanticResult`/
  `MaskSemanticVerdict`; `DroppedObjectResult.failing_stage` and `ObjectRunResult`/
  `PipelineRunResult` gain a `mask_semantics` field (Phase 12).
- `evaluation.schemas.PageRunOutcome`/`ObjectAttemptOutcome` gain `MaskSemanticOutcome` fields
  (`schema_version` 4 → 5) so a semantic mask rejection is reported distinctly from generic
  `ERROR`/`"unexpected"` (Workstream 15/57) in machine-readable evaluation output.
- Every real object that reaches this stage now costs one additional VLM call (the same
  `Qwen25VLClient` instance analysis/validation already use — no new model load). Real
  per-object timing is measured as part of this phase's GPU validation (see
  `docs/phase12-results.md`'s performance-baseline section) rather than assumed.
- This gate can newly REJECT a render that Phase 8.3/9/10/11's own protections all accepted
  (that is its entire purpose) — every one of those earlier protections is unchanged and still
  runs first; this is a strictly additive gate, not a replacement for any of them.

## Open questions

- **Calibration remains unresolved** (see point 2 above) — the `[0.4, 0.6]` abstain band is a
  documented placeholder, not a statistically justified cutoff. Revisit once real GPU
  validation produces enough `MaskSemanticResult` records (both confirmed-bad and
  confirmed-good) to attempt an actual precision/recall sweep — see
  `docs/phase12-results.md`'s calibration-study section for why this phase's own real dataset
  (12 objects, all already used to *design* this gate) cannot also serve as an unbiased
  evaluation set for it without disclosing that overlap. **Real GPU evidence gathered after this
  ADR's original design (docs/phase12-results.md section 10.1, independently confirmed by
  adversarial review) makes this worse than "unresolved": every real confidence value observed
  this phase clustered on one of two round numbers per verdict class (1.0 for accept, 0.7-0.75
  for reject), never landing inside the abstain band at all — the confidence signal itself looks
  quantized/uncalibrated, not merely under-sampled. Treat ABSTAIN as structurally near-
  unreachable in production until real evidence shows otherwise, not as a working safety valve.**
- **Instance identity** (correct category, wrong physical instance — e.g. two characters' hair)
  is not directly addressed by this check: a mask that is entirely "character_hair" content but
  belongs to the *wrong* character would plausibly still read as "yes, this is hair" to the VLM
  prompt as currently phrased. Documented as a real, unresolved limitation, not silently
  covered — see `docs/phase12-results.md`'s instance-consistency section.
- **False-rejection rate on legitimately difficult objects** (large/irregular/thin masks that
  are genuinely correct) is not yet measured against enough real difficult-good examples to be
  confident this gate doesn't over-reject them — the 8 "presumed-good" objects in Phase 11's
  own dataset were characterized as clean by visual inspection, not exhaustively stress-tested
  for difficulty. Flagged as development, not held-out, evidence.
