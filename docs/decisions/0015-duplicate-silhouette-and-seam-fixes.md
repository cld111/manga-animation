# 15. Phase 8.3: root-causing and fixing the two real Phase 8 visual defects

Status: Accepted

## Context

Phase 8 (ADR 0014, `docs/phase8-results.md`) ran the real pipeline on a live Kaggle GPU worker
and found two real, previously-undiscovered mid-cycle visual defects by direct inspection of
downloaded output — left as disclosed, unfixed known limitations because root-causing them
would need the actual segmentation masks/reconstruction data at peak displacement, and the live
Kaggle worker that produced them was already gone by the time they were found:

- **Defect A ("duplicate silhouette")** — `verified_action_1`, panel mode: a visible
  semi-transparent duplicate contour across the hair/goggles/jaw region, at mid-cycle
  (frame 24/96) only, absent at rest (frame 0) and at the true loop-wrap frame (frame 95).
- **Defect B ("vertical seam")** — `phase3_action_page`, panel mode: a sharp, rigid vertical
  boundary cutting through both background wall texture and the panel border, also mid-cycle
  only, also absent at rest/wrap.

This phase's brief: reproduce each defect, prove its root cause with evidence (not assumption),
implement the smallest correct fix, add regression protection, and re-verify on the real
pipeline — explicitly forbidding fabricated GPU execution, hidden artifacts, or declaring
success because tests are green.

## Investigation

**Local, non-GPU evidence** (real MP4s/PNG frames from the original Phase 8 Kaggle run,
retained locally under `outputs/videos/phase8_evidence/` and `outputs/frames/phase8_evidence/`,
git-ignored per ADR 0002 but present on this checkout):

- Defect B: pixel-diff analysis of the real pre-encode frame 0 vs. frame 24 (`cv2`, no model
  inference) found the changed region's "first-changed-column" was constant at `x≈48` for 89% of
  the object's full vertical extent (708 of 716 rows) — a dead-straight, page-aligned vertical
  line, structurally inconsistent with a hair silhouette's natural curved/jagged boundary
  (confirmed by contrast against Defect A's own column profile, which varies smoothly with no
  such flat band).
- Defect A: the same analysis on `verified_action_1`'s real frames showed a smoothly-varying,
  spiky/jagged double-exposure pattern (visually: two overlapping copies of the same jagged hair
  silhouette, offset) — no hard rectangular signature, a genuinely different shape from Defect B.
- A deterministic mechanism, derivable purely from reading `compositing/__init__.py` and
  `animation/transforms.py`: at an object's rest pose (`t_frac=0`, the schema's
  `phase=0`/symmetric-easing convention), `generate_transformed_layer`'s affine matrix is the
  exact identity, so the transformed layer's pixels are bit-identical to the plate regardless of
  the mask's shape — **any mask-shape or cross-object-overlap defect is structurally invisible
  at rest and can only become visible once the object is actually displaced**, exactly matching
  both defects' own observed frame-0-clean/frame-24-defective/frame-95-clean pattern.
- A code-level scan confirmed **zero** cross-object mask-overlap checks anywhere in
  `grounding`/`validation`/`segmentation`/`pipeline.orchestrator`/`compositing`, and confirmed
  `validation/transform_geometry.py`'s `TransformKind.TRANSLATE` profile has
  `min_edge_margin_fraction=0.0` (deliberately, per ADR 0008's revision) — i.e. no existing
  check could have caught either failure mode.
- A synthetic, deterministic (no-GPU) reproduction against the REAL production code
  (`generate_transformed_layer` + `composite_frame_stack`, unmodified) and the REAL source
  images: (a) a hand-constructed rectangular ("loose") mask animated via the real
  `phase3_action_page` TRANSLATE MotionSpec reproduced a hard vertical duplicate-slab seam
  visually matching the real defect; (b) two hand-constructed overlapping (IoU≈0.68) hair-shaped
  masks, each independently TRANSLATE-animated with real `verified_action_1`-derived motion
  parameters, reproduced a double-exposure ghost visually matching the real defect. Both were
  bit-identical to the untouched original at the rest frame, confirming the "invisible at rest"
  mechanism above.

**Real GPU re-verification** (a fresh Kaggle Jupyter kernel, user-provided URL, dedicated to this
session — real Qwen2.5-VL-7B-Instruct, Grounding DINO, SAM 2.1, LaMa, same commit under test):
`run_pipeline` was re-run for real on both `phase3_action_page.png` (panel mode) and
`examples/verified_action/action_sample_1.png` (`verified_action_1`, panel mode), and the real
`GroundingResult`/`SegmentationResult`/`ReconstructionResult` objects for every animated object
were downloaded (bboxes, masks, hole masks, LaMa-filled pixels — not just the final composited
video).

- **Defect B — proven, not merely correlated.** This session's real SAM 2.1 output for a
  `character_hair`/TRANSLATE candidate on `phase3_action_page.png` had `segmentation_bbox =
  (48, 3834, 551, 4551)` — matching the ORIGINAL Phase 8 defect video's own measured region
  (`x∈[48,606], y∈[3834,4550]`) almost exactly, including the `x0=48` left edge to the pixel,
  reproduced independently across two live GPU sessions. Direct inspection of the downloaded
  mask array shows it is **not** a tight hair silhouette: its own tight bbox's LEFT edge is
  mask-covered for **45.5%** of its height, vs. **2.2%–20.2%** for five other real masks
  downloaded in the same investigation (a raised sword, two eyes, a second real hair region) —
  a large, roughly-rectangular over-segmentation into the adjacent wall/panel background,
  visually confirmed against the crop (the mask's left third covers grey wall texture, not
  hair). Feeding this exact real mask, real hole mask, and real LaMa `filled_pixels` through the
  unmodified real `generate_transformed_layer`/`composite_frame_stack` locally reproduced the
  real seam **pixel-for-pixel**, visually indistinguishable from the original defect video's
  frame 24, while the rest-pose frame matched the untouched original exactly
  (max abs diff `0`). This is a complete, closed causal chain from real model output to the
  observed defect.
- **Defect A — mechanism proven, exact original instance not reproducible.** This session's real
  VLM/grounding read for `verified_action_1` (panel mode) produced a different, but structurally
  analogous, multi-object plan (`raised_sword`/ROTATE primary + 4 real SECONDARY/MICRO objects,
  matching the original defect run's object count) — a real instance of this project's
  already-documented VLM/grounding nondeterminism (ADR 0009). In this specific session's read,
  none of the 4 SECONDARY/MICRO objects' real masks happened to overlap each other (each landed
  in a different panel), so this session's own render does not itself display the ghost. The
  original defect-producing GPU session is confirmed gone (already disclosed in
  `docs/phase8-results.md`), so the *exact* overlapping pair that produced the original video
  could not be re-obtained. The causal **mechanism**, however, is proven by construction (see
  the synthetic reproduction above, using this project's real compositing code) and by the
  complete absence of any code-level safeguard against it — `docs/phase8-results.md`'s own
  original finding (two real `character_hair`/SECONDARY object ids, `obj_character_hair_4`/`_9`,
  both legitimately validated and independently animated) remains the best available direct
  evidence of the original instance, now with a proven, reproducible mechanism behind it rather
  than a stated hypothesis.

## Decision

**1. Cross-object mask-overlap guard (Defect A)** —
`pipeline.orchestrator._drop_overlapping_secondary_objects`, run once per pipeline call right
after segmentation, before animation. Compares every SECONDARY/MICRO object's real mask against
every already-accepted object's mask (PRIMARY processed first and never a drop candidate — its
deliberate top z-order, `_Z_ORDER_BY_MOTION_TYPE`, is this codebase's existing, intentional way
to let a SECONDARY/MICRO object sit legitimately behind PRIMARY, which this check must not
interfere with); if the containment-style overlap fraction (`intersection / area of the smaller
mask` — chosen over IoU so a small object fully nested inside a larger one's mask is still
caught) exceeds `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION = 0.25`, the later object is dropped
(non-fatal, recorded in the existing `dropped_objects`/`DroppedObjectResult` mechanism, exactly
like an existing grounding/validation drop). `0.25` sits far below the real defect's evidenced
overlap (~0.68 in the synthetic repro, structurally consistent with two detections of the same
physical region) with real margin.

**2. Mask-shape validation gate (Defect B)** —
`segmentation.segment._validate_mask_shape`, run immediately after `segment_object` computes a
mask's tight bbox. Rejects (raises `PipelineStageError(stage="segmentation")`, the same pattern
`_validate_mask`'s existing coverage-fraction checks already use) a mask whose own tight bbox
shows **asymmetric edge coverage on one axis**: one side (left/right, or top/bottom) touched for
more than `_MAX_BBOX_EDGE_TOUCH_FRACTION = 0.3` of its length while the geometrically OPPOSITE
side is touched for `0.3` or less — the real, evidenced signature of over-segmentation into
adjacent background on just one side (the real defect: LEFT=45.5%, RIGHT=0.6%, TOP=0.2%,
BOTTOM=0.4%).

An earlier version of this check flagged ANY single edge above the bound, regardless of its
opposite edge. Independent review (a fresh, adversarial `qa-agent` audit of this phase's own
uncommitted change set — see `docs/phase8.3-results.md` section 10) caught a real, undisclosed
false-positive risk in that version: a genuinely
rectangular real object — e.g. the "cloth-banner-shaped region"/"energy-effect-shaped region"
this project's own dataset (`phase3_action_page.acceptable_outcome`, `eval_weapon_effects.
acceptable_outcome`) explicitly names as a valid target — would touch BOTH edges of an axis near
100% together, indistinguishable from a one-sided over-segmentation by magnitude alone; no
threshold value fixes this, since the two cases can be numerically identical on a single edge.
The asymmetry requirement resolves it using the same real evidence already gathered (the
confirmed defect's own opposite edge was near-zero, not also high) — a real rectangle is no
longer flagged (a new regression test, `test_segment_object_accepts_a_genuinely_rectangular_mask`,
locks this in), while the real, evidenced defect still is (re-verified against the actual
downloaded defect mask array: still rejected, identical 45.5%/0.6% figures).

`0.3` sits with real margin between the highest normal real value observed (0.202) and the one
confirmed-defective real value (0.455).

**3. A separate, pre-existing orchestration bug, found and fixed as a prerequisite for (2) to
behave correctly for non-PRIMARY objects**: the segmentation stage's per-object loop in
`run_pipeline` had no `try`/`except` around `segment_object` at all — unlike the grounding and
validation stages immediately above it (which already correctly soft-drop a SECONDARY/MICRO
failure without failing the whole run), a SECONDARY/MICRO object's segmentation failure used to
fail the entire run, contradicting this module's own documented policy ("A SECONDARY/MICRO
object failing at grounding/validation/segmentation does NOT fail the run"). Fixed to match the
existing grounding/validation pattern exactly: PRIMARY re-raises (hard fail, unchanged), a
non-PRIMARY object is dropped into `dropped_objects` (`failing_stage="segmentation"`, reason
from the `PipelineStageError.detail`).

Both new checks reuse the project's existing acceptance/rejection vocabulary and constants style
— no new model, no rewrite of grounding/segmentation/animation/compositing/reconstruction, no
change to the loop-continuity contract (`LoopMetrics`, ADR 0014) or any previously-accepted
object's behavior. When either check fires on a PRIMARY object, the affected page is honestly
`REJECTED` (stage=`"segmentation"`) rather than silently rendering the defect — a valid outcome
per "Static Is a Valid Result" (`docs/architecture.md`), not a failure to be masked.

## Consequences

- `DroppedObjectResult.failing_stage` widens from `Literal["grounding", "validation"]` to
  include `"segmentation"` — additive, does not change the meaning of the existing two values.
- A page whose only real animatable candidate has a defective mask shape or an unresolvable
  overlap now honestly fails/drops instead of rendering a visual defect — this can change a
  specific real sample's outcome (e.g. `phase3_action_page`'s PRIMARY, if a future real session
  reads `character_hair` as PRIMARY again with this same mask defect, would now REJECT at
  segmentation rather than complete) — an intended, evidenced consequence of this fix, not a
  regression: the prior "success" was the defect itself.
- Both thresholds (`0.25`, `0.3`) are evidenced-but-not-statistically-calibrated, the same
  status as every other deterministic threshold in this codebase (e.g. `validation/
  transform_geometry.py`'s per-kind bounds) — calibrated against the real evidence gathered in
  this investigation (one confirmed defect instance each, several real non-defective instances),
  not a large statistical sample. A future session with more real mask data could refine them.
- Defect A's fix protects against the *mechanism* (any two overlapping, independently-animated
  objects), verified by reproducing that mechanism against real production code and real source
  pixels — it was not possible to re-verify against the *exact* original overlapping pair, since
  that live GPU session is gone (ADR 0003's disposable-compute policy, working as intended).

## Open questions

- The asymmetry refinement (see "Decision" above) rules out the clearest false-positive case
  (a genuinely rectangular object) using only the real evidence already gathered, but the
  underlying signal is still purely geometric (mask shape vs. its own tight bbox) — it cannot,
  by construction, distinguish "over-segmented into background" from "a real, organic silhouette
  that happens to be asymmetrically flush against one side of its own bbox" using pixel/color
  evidence the way a human inspecting the crop can. No real counterexample was found or
  constructed during this investigation (the five real non-defective masks gathered here all
  score well under the bound on every edge), but none of them is a case of a real object that is
  *itself* asymmetric-but-correct (as opposed to symmetric-and-correct, which the new regression
  test does cover) — left open pending more real mask data.
- Whether `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION`/`_MAX_BBOX_EDGE_TOUCH_FRACTION` should
  eventually be replaced by a statistically-calibrated value once more real mask data exists
  across more real pages — left open, same status as every other uncalibrated threshold already
  in this codebase.
- Whether a dropped PRIMARY-candidate mask-shape failure should attempt the next-ranked
  grounding candidate (mirroring `validate_target`'s own retry loop) instead of failing
  outright — not attempted here per "smallest correct fix"; the existing grounding-candidate
  retry loop already exists at the validation stage and is unchanged, so a differently-ranked
  candidate that also clears validation will still reach segmentation and get its own
  independent mask-shape check.
