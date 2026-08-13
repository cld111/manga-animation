# Phase 8.3 results: root-causing and fixing the two real Phase 8 visual defects

Status: **completed** — root cause proven for both defects with real evidence (local
reproduction plus two fresh live Kaggle GPU sessions), fixes implemented and regression-tested
locally (484 tests), real-GPU re-verification performed post-fix, and an independent adversarial
`qa-agent` audit completed (section 10) with one real gap found and fixed (the mask-shape check's
false-positive risk on legitimately rectangular objects, e.g. banners) before this phase was
considered closed. See [ADR 0015](decisions/0015-duplicate-silhouette-and-seam-fixes.md) for the
full design and evidence; this document records the concrete run-by-run numbers.

## 1. Scope

Phase 8 (`docs/phase8-results.md`) found two real, previously-undiscovered mid-cycle visual
defects on real Kaggle GPU output and left them as disclosed, unfixed known limitations —
root-causing them needed the actual segmentation masks/reconstruction data at peak displacement,
unavailable once that GPU session ended. This phase's brief: reproduce, prove root cause with
evidence, fix minimally, regression-test, and re-verify on the real pipeline.

## 2. Local, non-GPU forensics

Real MP4/PNG artifacts from the original Phase 8 defect-producing run were still present locally
under `outputs/videos/phase8_evidence/` and `outputs/frames/phase8_evidence/` (git-ignored,
present on this checkout). Pixel-diff analysis (plain `cv2`/`numpy`, no model inference) of the
real frame 0 vs. frame 24 for each defect found:

- **Defect B** (`phase3_action_page`): the changed region's leading edge sat at a constant
  `x≈48` for 708 of 716 rows (89%) of the object's full height — a dead-straight, page-aligned
  vertical line, inconsistent with a hair silhouette's natural curved boundary.
- **Defect A** (`verified_action_1`): a smoothly-varying, spiky double-exposure pattern with no
  hard rectangular signature — visually and quantitatively a different shape from Defect B,
  confirming the brief's own caution not to assume a shared cause.

Reading `compositing/__init__.py` and `animation/transforms.py` directly established the
mechanism that explains both defects' shared "invisible at rest, visible mid-cycle" pattern: at
an object's rest pose (`t_frac=0`, `phase=0`), the affine transform is an exact identity, so a
layer's pixels are bit-identical to the plate regardless of mask shape — any mask-shape or
cross-object-overlap defect is structurally invisible until the object is actually displaced.

A synthetic (no-GPU) reproduction against the unmodified real `generate_transformed_layer` +
`composite_frame_stack` code, using the real source images and real per-sample `MotionSpec`
values, reproduced both defects' visual signature from first principles: a hand-built
rectangular mask (Defect B) and two hand-built overlapping masks (Defect A, IoU≈0.68).

## 3. Real GPU forensics (live Kaggle session, user-provided URL)

A fresh, dedicated Jupyter kernel (2x Tesla T4) was used to `git clone` the repo at the exact
commit under test (`aaf69b0`), `uv sync --extra dev --extra cv --extra video --extra ml`, and
re-run `pipeline.orchestrator.run_pipeline` for real (real Qwen2.5-VL-7B-Instruct, Grounding
DINO, SAM 2.1, LaMa) against `phase3_action_page.png` and
`examples/verified_action/action_sample_1.png` (`verified_action_1`), both in panel mode — the
same real model versions and mode Phase 8 used. Every real `GroundingResult`/
`SegmentationResult`/`ReconstructionResult` for every animated object (not just the final
composited frames) was downloaded via the Jupyter Contents API for direct inspection.

### 3.1 Defect B — proven

This session's real SAM 2.1 output for a `character_hair`/TRANSLATE candidate on
`phase3_action_page.png` had `segmentation_bbox = (48, 3834, 551, 4551)` — matching the
*original* Phase 8 defect video's own measured region almost exactly (including the `x0=48`
left edge to the pixel), independently reproduced across two live GPU sessions. Direct
inspection of the downloaded mask confirmed it is not a tight hair silhouette: its own tight
bbox's LEFT edge is mask-covered for **45.5%** of its height, vs. **2.2%–20.2%** for five other
real masks downloaded in this same investigation (a raised sword, two eyes, a second real hair
region) — visually confirmed against the source crop (the mask's left third covers grey wall
texture, not hair).

**Decisive test**: this exact real mask, real hole mask, and real LaMa `filled_pixels` were fed
through the unmodified real `generate_transformed_layer`/`composite_frame_stack` locally. The
rest-pose frame matched the untouched original exactly (max abs diff `0`); the mid-cycle frame
reproduced the real seam pixel-for-pixel, visually indistinguishable from the original defect
video. A complete, closed causal chain from real model output to the observed defect.

### 3.2 Defect A — mechanism proven, original instance not re-obtainable

This session's real VLM/grounding read for `verified_action_1` produced a different (but
structurally analogous — same object count) multi-object plan than the original defect-producing
session, a real instance of this project's already-documented VLM/grounding nondeterminism
(ADR 0009). In this read, none of the 4 real SECONDARY/MICRO objects' masks happened to overlap
each other. The original defect-producing GPU session is confirmed gone (already disclosed in
`docs/phase8-results.md`), so the exact overlapping pair could not be re-obtained for direct
inspection. The causal *mechanism* is proven by the synthetic reproduction (section 2) and by
the complete, code-level absence of any cross-object overlap safeguard prior to this phase's fix.

## 4. Root cause summary

| Defect | Root cause |
| --- | --- |
| A (duplicate silhouette) | `compositing.composite_frame_stack` alpha-blends every animated layer independently, with no check for whether two layers' masks represent the same physical region. Two independently-accepted SECONDARY/MICRO objects whose masks substantially overlap, each animated with its own `MotionSpec`, double-expose once their motions diverge. |
| B (vertical seam) | A real SAM 2.1 mask over-segmented into adjacent background along one edge of its own tight bbox (45.5% edge coverage vs. 2.2–20.2% normal). `TransformKind.TRANSLATE`'s geometry-validation profile has `min_edge_margin_fraction=0.0` by deliberate design (ADR 0008), and no stage validates mask *shape* — so nothing caught the over-segmentation before it was animated, dragging the erroneous background region along with the real hair. |

## 5. Remediation

See [ADR 0015](decisions/0015-duplicate-silhouette-and-seam-fixes.md) for full rationale.

- `pipeline.orchestrator._drop_overlapping_secondary_objects` — new cross-object mask-overlap
  guard, run after segmentation. Non-fatal drop of a later-processed conflicting SECONDARY/MICRO
  object (`_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION = 0.25`, evidenced against a real ~0.68
  overlap and the mechanism's synthetic reproduction). PRIMARY is never a drop candidate.
- `segmentation.segment._validate_mask_shape` — new mask-shape validation gate, run right after
  a mask's tight bbox is computed. Rejects (`PipelineStageError(stage="segmentation")`) a mask
  whose own bbox shows asymmetric edge coverage — one side of an axis touched for more than
  `_MAX_BBOX_EDGE_TOUCH_FRACTION = 0.3` of that side's length while the geometrically opposite
  side is not (evidenced: 0.202 highest normal real value on any single edge, 0.455/0.006 the
  confirmed defect's own left/right pair) — refined from an earlier any-single-edge version
  after independent review found it would also flag a genuinely rectangular real object (see
  section 10).
- A separate, pre-existing orchestration bug found and fixed as a prerequisite: the segmentation
  stage's per-object loop had no `try`/`except` around `segment_object` at all, unlike grounding
  and validation immediately above it — a SECONDARY/MICRO segmentation failure used to fail the
  *whole* run, contradicting `run_pipeline`'s own documented drop policy. Now matches the
  grounding/validation pattern exactly.

No new model was introduced; no existing stage (grounding, validation, SAM segmentation,
deterministic CV animation, original-image compositing, multi-object support, loop construction)
was rewritten.

## 6. Regression tests

- `tests/test_segmentation.py::test_segment_object_raises_on_a_mask_that_hugs_one_bbox_edge_but_not_the_opposite_one`
  — fails on the pre-fix behavior (would have accepted the mask), passes on the fix.
- `tests/test_segmentation.py::test_segment_object_accepts_a_mask_that_only_touches_bbox_edges_near_their_midpoint`
  — negative control: a realistic mask shape must not be rejected.
- `tests/test_segmentation.py::test_segment_object_accepts_a_genuinely_rectangular_mask` —
  negative control added after independent review: a real, legitimately rectangular object
  (e.g. a banner/flag) must not be rejected merely for having straight edges.
- `tests/test_pipeline.py::test_run_pipeline_drops_a_secondary_object_whose_mask_overlaps_an_already_accepted_one`
  — Defect A invariant: two overlapping SECONDARY masks must never both reach the render.
- `tests/test_pipeline.py::test_run_pipeline_keeps_two_secondary_objects_with_genuinely_distinct_masks`
  — negative control: genuinely distinct multi-object rendering (ADR 0010's core guarantee) must
  not regress.
- `tests/test_pipeline.py::test_run_pipeline_keeps_two_secondary_objects_whose_bboxes_intersect_but_masks_do_not`
  — stronger negative control added after independent review: exercises the actual pixel-overlap
  arithmetic (not just the cheap bbox short-circuit) on a real intersecting-bbox case.
- `tests/test_pipeline.py::test_run_pipeline_raises_stage_segmentation_when_primary_mask_hugs_a_bbox_edge`
  — Defect B on PRIMARY: honest hard failure, not a silently-rendered defect.
- `tests/test_pipeline.py::test_run_pipeline_drops_a_secondary_whose_mask_hugs_a_bbox_edge_without_failing_the_run`
  — Defect B on SECONDARY, plus the separate pre-existing orchestration-gap fix: non-fatal drop,
  not a whole-run failure.
- Existing fake-client test fixtures (`FakeSegmentationClient` in both `test_pipeline.py` and
  `test_segmentation.py`) previously produced solid-rectangle masks, which the new, real,
  evidenced check now correctly rejects — updated to a diamond shape inscribed in the same box
  (same tight-bbox-equals-box property, realistic edge-touch fractions), not to weaken the check.

Full local suite: `uv run pytest -q` — **484 passed, 2 deselected** (up from Phase 8's 472; +12
from this phase — 4 evaluation, 8 pipeline/segmentation, including 2 added after independent
review). `uv run ruff check .` clean. `uv run mypy src` clean, 41 files. Independently
reproduced by a fresh `qa-agent` audit (section 10), including mutation testing on both new
checks.

## 7. `classify_outcome` contract fix (brief section 13)

A real, previously-undocumented mismatch: `phase3_action_page` and `eval_weapon_effects` both
have confident (`ground_truth_uncertain=False`) `animation_possible="yes"` ground truth, but
their own `acceptable_outcome` prose has always explicitly allowed an honest, attributed
grounding/validation failure too (effect-heavy motion cues, not one concrete easily-prompted
object) — `classify_outcome` only consulted structured fields and classified both as `ERROR` on
real Kaggle GPU output (`docs/phase8-results.md` section 6.2), contradicting each sample's own
written contract. Fixed via a new structured `EvalSample.honest_failure_acceptable: bool` field
(formalizing the existing prose, not inventing new ground truth) and a small `classify_outcome`
carve-out: an *attributed* failure on such a sample is `REJECTED`, not `ERROR`; an unattributed
one still is. Both real samples bumped to `annotation_version=2`. Regression tests: 4 new
`classify_outcome` tests plus a real-dataset consistency check
(`test_real_dataset_honest_failure_acceptable_matches_documented_samples`).

## 8. Golden dataset gaps (brief section 14)

Re-checked for any newer real evidence: none found. `partially_occluded_object` and
`scale_or_deformation` remain genuine, disclosed coverage gaps, exactly as ADR 0014 already
recorded — the only related real evidence (`docs/phase7-results.md` section 6.5's SCALE/
MESH_WARP runs against `sample_page_01.png`) used the controlled-fallback plan override, not the
golden dataset's own automatic E2E flow, so it cannot honestly be added to that sample's
`golden_categories` without misrepresenting what the automatic pipeline demonstrates for it — a
tension ADR 0014 itself already anticipated. No fixture manufactured to force 10/10 coverage.

## 9. Real GPU E2E re-verification (post-fix)

The fixed `pipeline/orchestrator.py`/`segmentation/segment.py` were uploaded to the same live
Kaggle kernel (already-loaded real Qwen2.5-VL/Grounding DINO/SAM 2.1/LaMa clients reused after a
kernel restart to force a clean re-import), and `run_pipeline` was re-run for real, panel mode,
on both affected samples.

**`phase3_action_page`**: this session's real VLM read again selected `character_hair` as
PRIMARY (the same object class as the original defect). The run **REJECTED** at the segmentation
stage: `"mask hugs one side of its own tight bbox for 45.5% of that side's length, exceeding the
30% bound..."` — the exact same 45.5% figure independently measured from the downloaded mask
array in section 3.1, now firing live, in-pipeline, on a fresh real inference pass. No defective
video was produced; the page honestly rejects for a demonstrably safe, evidenced reason, per the
brief's own completion criterion.

**`verified_action_1`**: **COMPLETED** — `raised_sword`/ROTATE primary, 3 real SECONDARY/MICRO
objects rendered (`character_hair`×2, `eye`), 5 dropped. One of the 5 drops was the new overlap
guard firing for real, on a genuine live-inference mask, not a staged case: `obj_character_eye_3`
(`character_eye`/MICRO/opacity) was dropped because its real mask overlapped the accepted PRIMARY
sword's mask by **26.4%** of its own area (`> 25%` bound) — direct, real-world confirmation the
guard operates correctly on unstaged data. `render.seamless_loop_verified=True`, 96 frames.

The real, decoded output video was downloaded and visually inspected (frame 0 vs. frame 24, at
both the original composited resolution and 3x-zoomed native-resolution crops around each real
hair object's exact bbox, plus a pixel-diff heatmap) — both hair objects show clean, single-copy
TRANSLATE motion with no duplicate silhouette. (A first-pass wide, scaled-down crop appeared to
show a ghost near `obj_character_hair_9`; a tight, native-resolution re-crop at the object's
exact bbox and a 3x zoom on the most distinctive feature — the goggle — showed this was a
viewing/downsampling artifact, not a real defect in the rendered pixels, confirmed by comparing
byte-identical crop data at two different display scales.) This session's real VLM/grounding
read did not happen to reproduce the *exact* overlapping `character_hair` pair from the original
defect video (ADR 0009 nondeterminism, also disclosed in ADR 0015) — the mechanism is
confirmed working correctly on real data via the `character_eye_3`/sword drop instead.

**Independently checkable artifacts**: unlike section 3's evidence (present locally from Phase
8's own original run), the post-fix re-verification session's live Kaggle kernel is now gone
too (same disposable-compute pattern, ADR 0003). The specific evidence this section's claims
rest on was copied locally before that session ended and is present on this checkout
(git-ignored per ADR 0002, same convention as `outputs/frames/phase8_evidence/`):
`outputs/debug/phase8_3_verification/phase3_action_page_hair6_real_sam_mask.npy` (the real
downloaded mask array; loading it and calling `segmentation.segment._validate_mask_shape`
directly reproduces the exact 45.5%/0.6% figures and the rejection) plus its `_info.json`
(bbox, motion, etc.); `outputs/debug/phase8_3_verification/verified_action_1_postfix_dropped_
objects.json` (the real `character_eye_3` 26.4% overlap drop, verbatim from the live run);
`outputs/frames/phase8_3_verification/verified_action_1_postfix_frame_{0000,0024,0095}.png`
(the real decoded post-fix video frames) and the tight-crop/goggle-zoom/marked-wide-crop PNGs
used for the "false alarm" determination above, so a skeptical reader can independently re-derive
the same conclusion from the same bytes without needing live GPU access.

One honesty note this document did not originally make explicit, raised by the independent
review below: the `x0=48` bbox match to the pixel across two independent live Kaggle sessions
(section 3.1) is presented as strong corroboration, without connecting it to this project's own
extensively-documented VLM/grounding nondeterminism (ADR 0009). The likely reason the two
sessions agreed this precisely is that Grounding DINO and SAM 2.1 are feed-forward (not
autoregressive/sampling) models, plausibly far more deterministic run-to-run than the VLM's own
free-text analysis stage — but this is a plausible explanation offered here, not something this
investigation independently verified against the model internals.

## 10. Independent review

A fresh `qa-agent` session (no access to this investigation's own reasoning beyond what was
already committed/written to disk) audited the uncommitted change set: read the full `git diff`,
the ADR, this document, and all changed/new source and test files; independently re-ran
`uv run pytest -q`/`ruff check .`/`mypy src`; independently re-derived the overlap-fraction
arithmetic for the Defect A test fixtures by hand; independently confirmed (via `git show HEAD`)
that the segmentation-stage orchestration bug genuinely pre-existed; performed real mutation
testing on both new checks (disabling each call site, confirming the relevant new tests fail,
reverting, confirming `git diff` was restored exactly); and independently re-inspected the
pre-fix defects' visual character directly from the retained `outputs/frames/phase8_evidence/`
crops (not just this document's prose).

**Overall verdict**: both fixes address the real mechanism, not the visual symptom — confirmed
by mutation testing, independent arithmetic verification, and independent re-derivation of the
pre-existing orchestration bug.

**Two real findings, both acted on before this phase was considered closed**:

1. **Defect B's mask-shape check had an undisclosed false-positive risk** on legitimately
   rectangular real objects (a banner/flag/sign — explicitly named as a valid target in this
   project's own dataset). Neither the code comment nor the ADR's "Consequences"/"Open
   questions" originally disclosed this. **Fixed**: the check now requires asymmetric edge
   coverage (one side of an axis hugged, the opposite side not — the real defect's own
   evidence, LEFT=45.5% vs. RIGHT=0.6%), which a genuine rectangle does not exhibit (both
   opposite edges would be high together). Re-verified against the real downloaded defect mask:
   still correctly rejected, identical figures. New regression test
   (`test_segment_object_accepts_a_genuinely_rectangular_mask`) locks this in. See ADR 0015's
   "Decision" section for the full before/after.
2. **Section 9's post-fix claims were not independently checkable from the repo** (the live
   Kaggle session that produced them was already gone by the time of review, unlike the
   pre-fix evidence, which the reviewer could and did re-inspect locally). **Fixed**: the
   specific mask array, dropped-object JSON, and rendered frames the claims in section 9 rest on
   were copied locally (git-ignored, `outputs/debug/phase8_3_verification/`,
   `outputs/frames/phase8_3_verification/`) before that session ended — see section 9's own
   "Independently checkable artifacts" paragraph for the exact files and how to reproduce the
   claim (`_validate_mask_shape` against the real saved mask array reproduces the exact 45.5%
   rejection with no GPU needed).

A smaller documentation point (this file's own "Status" header contradicting its section 9's
completed-sounding prose and README's "completed" table entry) was also raised and is fixed by
this section's own presence — the three now agree.

A real, disclosed remaining gap the review did **not** ask to be closed further (see ADR 0015's
"Open questions"): the asymmetry refinement is evidenced against the real data gathered in this
investigation, not a large statistical sample, and cannot distinguish "over-segmented into
background" from "a real object that happens to be asymmetrically flush against one side of its
own bbox" using geometry alone — no counterexample was found, and closing this further would
need either more real mask data or a genuinely new signal (e.g. color/texture continuity across
the touched edge), which is out of this phase's "smallest correct fix" scope.

Also independently confirmed: the `classify_outcome`/`honest_failure_acceptable` fix (section 7)
applies to exactly the two intended samples and no others, verified against
`configs/phase3_3_eval_dataset.yaml`'s own prose directly.

## 11. Known limitations / remaining gaps

- `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION` (0.25) and `_MAX_BBOX_EDGE_TOUCH_FRACTION` (0.3) are
  evidenced against one confirmed-defective real instance each plus a handful of real
  non-defective instances gathered in this investigation — not a large statistical sample, same
  disclosed status as this codebase's other deterministic thresholds.
- `_validate_mask_shape`'s asymmetry requirement (added after independent review, section 10)
  removes the clearest false-positive case (a genuinely rectangular object) but the underlying
  signal is still purely geometric — it cannot rule out a real, organic object that happens to be
  asymmetrically flush against one side of its own bbox by construction. No real counterexample
  was found; left open (ADR 0015's "Open questions") pending more real mask data.
- Defect A's exact original overlapping pair (the specific masks that produced the original
  video) could not be re-obtained — the mechanism is proven, the exact original instance is not
  independently re-confirmed pixel-for-pixel (unlike Defect B, which was).
- `partially_occluded_object`/`scale_or_deformation` golden-dataset coverage gaps remain (see
  section 8) — unchanged by this phase, already disclosed.
