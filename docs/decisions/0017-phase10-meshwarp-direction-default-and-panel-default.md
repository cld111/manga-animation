# 17. Phase 10: mesh_warp direction-default fix and panel-mode default

Status: Accepted

## Context

Phase 9 (`docs/phase9-results.md`) ran the real pipeline on a live Kaggle GPU worker against a
new 10-sample real-world dataset and found three new, previously-undocumented mid-cycle visual
defects by direct inspection of downloaded output, left unfixed (root-causing was out of that
phase's scope, and the live GPU session was gone by the time they were found):

- `realworld_marika_love_meter` (page mode) — a duplicated/offset hand+hair silhouette.
- `realworld_wind_breaker_finish` (panel mode) — a vertical streaking/warping distortion.
- `realworld_villainess_ending_scuffle` (panel mode) — a hard vertical torso/skirt seam.

Phase 10's brief: forensically investigate all three, prove root cause with real evidence,
disclose what evidence is unavailable rather than fabricate it, implement the minimal
architectural fix, and add regression coverage — the same standard `docs/decisions/0015-
duplicate-silhouette-and-seam-fixes.md` (Phase 8.3) already set.

**Evidence available**: unlike Phase 8.3, Phase 9's live Kaggle session was already gone before
this phase started (disclosed in `docs/phase9-results.md` section 10/13) and no intermediate
`GroundingResult`/`SegmentationResult`/`ReconstructionResult` artifacts were saved for these
three samples (`outputs/debug/` only holds Phase 8.3's own saved arrays). What IS available
locally: the real rendered `.mp4` files and extracted frame PNGs/diff heatmaps
(`outputs/videos/phase9_evidence/`, `outputs/frames/phase9_evidence/`), the real source page
images (`examples/realworld/*.png`), the raw per-sample outcome JSON
(`outputs/experiments/phase9_evaluation_20260813T174730Z.json` — object ids, motion types,
render summaries, but no pixel-level mask/bbox data), and the full, unmodified Phase 9 production
code. This phase's forensics therefore reproduce each defect's *mechanism* deterministically
against the real code, real source images, and constructed masks consistent with the available
evidence — never a pixel-exact reconstruction of the original live-GPU instance. See
`docs/phase10-results.md` for the full evidence matrix.

## Investigation summary

**`wind_breaker_finish` and `villainess_ending_scuffle` share one proven root cause.** Both
renders include a MESH_WARP object (`obj_character_clothing_1`, `obj_cloth_5`) produced by
`analysis/plan_builder.py`'s `_MOTION_HEURISTICS` flag/banner/cloth/cape/cloak/drape/curtain
entry — which never sets `MotionSpec.direction` (only `transform_kind`, `amplitude`, `speed`,
`easing`, `pivot`). `animation/transforms.py::_mesh_warp_frame` used to default an unset
direction to a hardcoded `(1.0, 0.0)` (rightward) regardless of the object's own mask shape.
Since the function's `local` falloff and displacement axis are both tied to that direction, a
mask taller than it is wide (a sleeve, a clothing panel — the natural shape for these two real
objects) received the SAME horizontal displacement at every row from its top to its bottom: a
rigid sideways shear, uncorrelated with the object's own height, and unrelated to the top-anchor
convention the same heuristic's own `pivot=(0.5, 0.0, object_bbox)` clearly signals ("hangs from
a fixed point"). Reproduced deterministically against the real, unmodified
`generate_transformed_layer` using a real Phase 9 source image
(`examples/realworld/villainess_ending_scuffle.png`) and a constructed tall mask: under the old
default, every sampled row from `y=6` to `y=54` (a 5-49px range) shifted its mask x-range by the
identical `(30,33)` (from an original `(30,39)`) — a uniform shear independent of row position,
which is exactly the "hard, page-aligned vertical duplicate" visual signature both real defects
show. `wind_breaker_finish`'s own visual signature (a wavy streak rather than a single clean
seam) is consistent with the same mechanism interacting with fine internal mask/image detail
(the bicycle wheel/spokes) rather than a uniform fabric region — this is the mechanism's
predicted behavior on that kind of content, not an independently reproduced instance (no real
mask was available for that sample either).

**`marika_love_meter` has a different, unconfirmed root cause.** Only one object is animated in
this render (the PRIMARY, `semantic_label="clapping"`; the sole SECONDARY candidate,
`obj_greeting_1`, was dropped at grounding) — so this is not the Phase 8.3 Defect A mechanism
(no second object to overlap with). TRANSLATE is a rigid, mask-shape-preserving transform, so
the composited "duplicate ghost" silhouette's own shape (visible in
`outputs/frames/phase9_evidence/crops/page_marika_love_meter_diffheat.png` — hair, glasses-effect
box, torso outline, raised hand) must already have been present in the ORIGINAL segmentation
mask, not introduced by compositing. A locally-reproduced, deterministic run of
`analysis.panels.detect_panels` (classical CV, no GPU) against the real source image confirms the
same panel layout Phase 9's real run used (5 panels, matching `panel_count=5` in the saved
outcome JSON) and shows the diff-heatmap's changed region sits entirely inside `panel_01`
(`(0,241)-(850,859)`, the large "woman waving" panel), not `panel_00`
(`(0,0)-(850,258)`, the "CLAP!!" panel `docs/phase9-results.md` section 7.1.1 attributes this
motion to) — an apparent discrepancy with that document's own characterization, evidenced by this
locally-reproducible panel geometry plus the already-published diff heatmap, but NOT independently
confirmed against the real `AnimationPlan`/`GroundingResult` (gone with the live session). Whether
the true cause is a page-mode-specific panel/object misattribution, an over-inclusive Grounding
DINO box, or SAM 2.1 expanding beyond its prompted box cannot be determined from available
evidence — marked `UNKNOWN` (see `docs/phase10-results.md`). What IS established: the existing
Phase 8.3 `segmentation.segment._validate_mask_shape` check (asymmetric edge coverage) would not
catch an over-inclusive mask that isn't edge-hugging on one side — a real, disclosed gap this
phase does not close (see "Deferred" below), and panel mode's own INDEPENDENT grounding attempt
for the same page WAS caught by that exact check (`realworld_marika_love_meter`, panel mode:
segmentation REJECTED, "mask hugs its own tight bbox's bottom edge for 70.8%... while the
opposite edge is only 7.7%").

**The compositing/reconstruction invariant itself holds.** Direct reading of
`compositing/__init__.py::composite_frame_stack` and `reconstruction/__init__.py::
reconstruct_hidden_region` (unchanged since Phase 8.3) confirms: for a correct mask, an object is
composited exactly once per frame (mask>0 → transformed content; mask==0 & hole_mask!=0 & not
covered by another layer → reconstructed background; neither → untouched original), reconstruction
is computed from ALL `frame_count` frames' transformed masks (not a sparse sample), and Phase
8.3's cross-object overlap guard remains intact and unmodified. None of the three Phase 9 defects
are a violation of this invariant — they are cases where the INPUT to a structurally-correct
compositing pipeline (the mask, or the transform's own axis choice) was itself wrong. Phase 10's
fixes therefore target the animation/heuristic layer, not compositing, and require no change to
Phase 8/8.3's protections.

## Decision

**1. `animation/transforms.py::_mesh_warp_frame`'s direction fallback now follows the object's
own bbox shape.** When `motion.direction is None`: a mask at least as tall as it is wide
(`(y1-y0) >= (x1-x0)`) defaults to `(0.0, 1.0)` (downward sway, anchored at the top — matching
the flag/cloth heuristic's own `pivot=(0.5, 0.0, object_bbox)`); a wider-than-tall mask keeps the
previous `(1.0, 0.0)` (rightward sway, anchored at the left) — the real, already-validated
flag/banner behavior is unchanged. An explicit `motion.direction`, when given, is used exactly as
before regardless of mask shape — this fix only changes the *unset*-direction fallback.

**2. `pipeline/orchestrator.py::run_pipeline`'s `analysis_mode` default changes from `"page"` to
`"panel"`.** Real, already-published Phase 9 evidence (`docs/phase9-results.md` section 5.3):
`end_to_end_completion_rate` 20%→60%, `grounding_success_rate` 50%→100%, ERROR-classified
outcomes 5→0, on a real 10-sample diverse dataset. Phase 10's own forensics add a second,
concrete data point: page mode's single whole-page VLM call is the most plausible proximate cause
of `marika_love_meter`'s defect (see "Investigation" above), and panel mode's independent,
per-panel grounding attempt for the exact same page was honestly REJECTED by an existing safety
check instead of silently rendering the defect. This satisfies the Phase 10 brief's own
conditions for changing the default in this phase ("trivial, isolated, backward-compatible, and
covered by tests"): a one-line default change, every existing caller that explicitly passes
`analysis_mode` is unaffected, `analysis_mode="page"` remains fully available and behaviorally
unchanged for any caller that asks for it, and both the changed default and the still-available
explicit page path are covered by dedicated tests
(`test_run_pipeline_analysis_mode_defaults_to_panel_level`,
`test_run_pipeline_analysis_mode_page_still_available_explicitly`).

Both fixes are minimal and additive: no new model, no change to grounding/segmentation/
compositing/reconstruction/rendering, no change to any Phase 8/8.3 validation gate or threshold.

## Consequences

- Any MESH_WARP object whose mask is taller than wide and has no explicit `direction` now sways
  vertically instead of shearing horizontally — a behavior change for every such object, all in
  the direction of correctness (matches the heuristic's own top-anchor pivot convention). No
  existing test exercised this exact case with a specific expected pixel outcome (the only
  pre-existing MESH_WARP direction=None coverage was a bare determinism check), so this is a real
  behavior change, not merely a bug fix invisible to the test suite.
- `run_pipeline`'s default now performs panel detection + one VLM call per detected panel instead
  of one whole-page call — measurably slower for pages with several panels (more VLM calls) but,
  per Phase 9's own evidence, substantially more likely to complete successfully and, per this
  phase's own finding, less likely to render an undetected visual defect. Any caller that needs
  the old behavior (fewer VLM calls, or a specific reason to prefer whole-page reads) must now
  pass `analysis_mode="page"` explicitly.
- Two existing Phase 7.1 regression tests
  (`test_run_pipeline_multi_object_no_color_bleed_between_objects_across_the_loop`,
  `test_run_pipeline_multi_object_e2e_encode_decode_regression`) now pass `analysis_mode="page"`
  explicitly — their `FakeVLMClient` fixture returns identical canned decisions regardless of
  which panel it is asked about, so panel mode's one-call-per-panel behavior would pool duplicate
  objects across arbitrary panel ids on their synthetic multi-block test image. This is a fixture/
  mocking artifact of the default change, not evidence of a real production bug — confirmed by
  the fact that pinning `analysis_mode="page"` restores their exact original passing behavior
  with no other change.

## Deferred / open questions (NOT implemented this phase)

- **`marika_love_meter`'s exact root cause remains `UNKNOWN`** at the grounding/segmentation
  level — no live GPU access was available to download the real `GroundingResult`/
  `SegmentationResult` this phase would need to distinguish "VLM assigned the wrong panel_id",
  "Grounding DINO's box was already too large", and "SAM 2.1 expanded beyond its prompted box".
  The panel-mode-default change (Decision 2) mitigates this specific instance (panel mode's own
  independent attempt was safely rejected) but does not close the general gap: a general
  "over-inclusive/organic segmentation mask" detector, complementary to
  `_validate_mask_shape`'s edge-asymmetry-only check, remains unimplemented — deliberately, since
  no real mask/bbox pair for this defect exists to evidence a calibrated threshold, and the Phase
  10 brief explicitly forbids introducing "speculative heuristics without evidence."
- **MESH_WARP's `strength = value * amplitude * max(bbox_width, bbox_height)` has no upper bound**
  relative to the panel/page — a legitimately large mesh_warp object (or one with an inflated
  bbox from a scattered/disconnected mask) can still produce an arbitrarily large per-pixel
  displacement even after Decision 1's axis fix. This is a plausible *contributing* factor to
  `wind_breaker_finish`'s severity specifically, but not independently confirmed (no real mask
  data for that sample) — left open pending real calibration evidence, same status as this
  codebase's other deterministic thresholds before they had real defect data behind them (see
  ADR 0015's own thresholds' history).
- **The `_MOTION_HEURISTICS` "cloth" keyword substring also matches "clothing"/"clothes"** (e.g.
  `character_clothing`), giving a general garment object the same `amplitude=0.12` tuned for a
  flag/banner. Plausibly not ideal, but not independently evidenced as wrong (clothing sway is
  not physically implausible), and changing keyword-matching or amplitude tuning is a speculative
  parameter change outside this phase's evidenced scope — left unchanged, noted here for
  visibility.
- **`check_transform_geometry`'s `MESH_WARP` profile bounds bbox area/aspect ratio but not
  `strength`** — the pre-segmentation geometry check and the post-segmentation transform math are
  not connected; a future phase could thread a `strength` bound through, but doing so without a
  real evidenced magnitude risks the same "speculative heuristic" problem as the point above.
