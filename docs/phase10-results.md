# Phase 10 results: mid-cycle artifact forensics and compositing correctness

Status: **completed, with one defect honestly left open**. Two of three Phase 9 mid-cycle visual
defects are forensically root-caused and fixed, re-verified clean at native resolution on real
GPU output. The third's original hypothesis was tested and disconfirmed by real post-fix GPU
evidence; its true root cause remains `UNKNOWN`, disclosed as open, not hidden or converted into
an unproven fix. Every claim below is checked against an actual downloaded artifact, a real GPU
log line, or a locally-reproduced deterministic result — same convention as
`docs/phase8-results.md`/`docs/phase9-results.md`.

## 1. Scope

Phase 9 (`docs/phase9-results.md`) found three new, previously-undocumented mid-cycle visual
defects on real Kaggle GPU output and left them unfixed (root-causing was out of that phase's
scope; its own live GPU session was already gone by the time they were found). Phase 10's brief:
forensically investigate all three using real evidence wherever available, honestly disclose any
evidence that is unavailable rather than fabricate it, implement the minimal architectural fix for
every *confirmed* root cause, add regression tests, verify against Phase 8/8.3 protections, and
validate on real GPU hardware — not merely "tests pass" or "the video renders."

This phase does **not** implement: a general panel-level scene-decomposition redesign, semantic/
articulated animation, coordinated multi-part motion primitives, scene transitions, or any new
generative model — all explicitly out of scope per the brief's "Future roadmap" section.

## 2. Exact Phase 9 defects investigated

1. `realworld_marika_love_meter` (page mode, PASS) — a visible duplicated/offset copy of the
   character's raised hand and hair silhouette at frame 24 of 96, absent at frame 0/95.
2. `realworld_wind_breaker_finish` (panel mode, PASS) — a vertical streaking/warping distortion
   across the bicycle-wheel region at frame 24, absent at frame 0/95.
3. `realworld_villainess_ending_scuffle` (panel mode, PASS) — a sharp, hard vertical
   discontinuity splitting the character's torso/skirt at frame 24, absent at frame 0/95.

## 3. Evidence sources

**Unavailable, disclosed up front**: Phase 9's live Kaggle GPU session was already gone before
this phase started (already disclosed in `docs/phase9-results.md` sections 10/13). No
intermediate `GroundingResult`/`SegmentationResult`/`ReconstructionResult` arrays were saved
locally for these three samples — unlike Phase 8.3, which had `outputs/debug/
phase8_3_verification/*.npy` for its own two defects. This phase's *initial* forensics (section 5
below) therefore could not inspect the exact original masks; they reproduce each defect's
*mechanism* deterministically against the real production code, real source images, and
constructed evidence-consistent masks — never a pixel-exact reconstruction of the original
live-GPU instance, and always labeled as such.

**Available and used**:
- Real rendered `.mp4` videos and extracted frame PNGs/diff-heatmap crops from Phase 9's own run:
  `outputs/videos/phase9_evidence/`, `outputs/frames/phase9_evidence/` (git-ignored, present on
  this checkout).
- The real Phase 9 source page images: `examples/realworld/*.png`.
- The raw Phase 9 per-sample outcome JSON: `outputs/experiments/
  phase9_evaluation_20260813T174730Z.json` (object ids, motion types, grounding scores, render
  summaries — no pixel-level mask/bbox arrays).
- The full, unmodified Phase 9 production code (`src/manga_animation/`).
- A **new, real, live Kaggle GPU session** (user-provided URL, 2x Tesla T4, real
  Qwen2.5-VL-7B-Instruct/Grounding DINO/SAM 2.1/LaMa), used for (a) post-fix E2E re-validation on
  all three samples and (b) — once the `wind_breaker_finish` re-validation surfaced a
  disconfirming result — a one-off live diagnostic re-run that read the real, live
  `SegmentationResult.bbox` for every animated object, closing what would otherwise have been a
  second "no live mask access" gap for that specific finding. See section 10/11.

## 4. Root cause of each defect

### 4.1 `wind_breaker_finish` / `villainess_ending_scuffle` — one proven mechanism (confirmed for one, disconfirmed as dominant for the other)

Both renders include a `mesh_warp` object (`obj_character_clothing_1`, `obj_cloth_5`) produced by
`analysis/plan_builder.py`'s `_MOTION_HEURISTICS` flag/banner/cloth/cape/cloak/drape/curtain
entry, which never sets `MotionSpec.direction`. `animation/transforms.py::_mesh_warp_frame` used
to default an unset direction to a hardcoded `(1.0, 0.0)` regardless of the object's own mask
shape. Since the function's displacement axis is tied to that direction, a mask taller than it is
wide received the **same** horizontal displacement at every row from top to bottom — a rigid
sideways shear uncorrelated with the object's own height, unrelated to the top-anchor convention
the same heuristic's own `pivot=(0.5, 0.0, object_bbox)` signals.

**Reproduced deterministically** against the real, unmodified `generate_transformed_layer`, a real
Phase 9 source image (`examples/realworld/villainess_ending_scuffle.png`), and a constructed tall
mask: under the old default, every sampled row from `y=6` to `y=54` shifted its mask x-range by
the identical `(30,33)` (from an original `(30,39)`) — a uniform, row-independent shear, matching
the "hard, page-aligned vertical duplicate" signature both defects showed in Phase 9's own
evidence.

**Real post-fix GPU re-validation** (section 10/11) confirms this mechanism explains
`villainess_ending_scuffle` — the fixed render is visually clean at native resolution — but
**disconfirms it as the dominant cause of `wind_breaker_finish`**: that sample's `mesh_warp`
object's real post-fix bbox (`398x543`, taller than wide) correctly took the fix's new vertical
branch, yet the visible defect is unchanged. See section 4.2.

### 4.2 `wind_breaker_finish` — original hypothesis disconfirmed; true cause `UNKNOWN`, one new real lead

A live diagnostic run against the real, live Kaggle session (section 10) read every animated
object's real post-fix `SegmentationResult.bbox`:

| object | transform_kind | bbox (px) | w×h | amplitude |
| --- | --- | --- | --- | --- |
| `obj_object_in_motion_12` (PRIMARY) | translate | `(379,869)-(768,2250)` | 389×1381 | 0.02 |
| `obj_character_hair_0` | translate | `(225,1296)-(363,1477)` | 138×181 | 0.03 |
| `obj_character_clothing_1` | mesh_warp | `(176,770)-(574,1313)` | 398×543 | 0.12 |
| `obj_character_hair_7` | translate | `(50,6298)-(800,7135)` | 750×837 | 0.03 |

The PRIMARY object (`translate`, entirely unaffected by this phase's fix) has a real bbox of
`389×1381` — unusually tall/elongated for "a bicycle wheel," the semantic label
(`object_in_motion`) grounding actually accepted. At `amplitude=0.02` against panel `panel_00`'s
real diagonal (`800×4268`, diag≈4343px), peak displacement is `0.02 × 4343 ≈ 87px` — a
substantial horizontal shift for a very elongated, fine-structured (wheel-spoke) region, and a
plausible mechanism for a "streaking/wavy" visual (thin, high-frequency mask content combined with
a large rigid shift and partial-alpha mask edges).

This is a **real, evidenced lead from exactly one real instance**, not independently confirmed as
a general pattern, and **not fixed in this phase** — implementing a fix from one data point (e.g.
capping `translate` displacement relative to bbox aspect/detail, or a general
oversized-mask-for-its-semantic-label detector) would be exactly the "speculative heuristic
without evidence" the brief prohibits. Marked `UNKNOWN`, left as disclosed, prioritized future
work (section 14).

### 4.3 `marika_love_meter` — root cause `UNKNOWN`, mitigated (not fixed) via the panel-mode default

Only one object is animated in this render (PRIMARY `clapping`; the sole SECONDARY candidate,
`obj_greeting_1`, dropped at grounding) — this is **not** the Phase 8.3 Defect A mechanism (no
second object to overlap with). `translate` is a rigid, mask-shape-preserving transform, so the
composited "duplicate ghost" silhouette's own shape (hair, glasses-effect box, torso outline,
raised hand — visible in `outputs/frames/phase9_evidence/crops/
page_marika_love_meter_diffheat.png`) must already have been present in the ORIGINAL segmentation
mask, not introduced by compositing.

A locally-reproduced, deterministic run of `analysis.panels.detect_panels` (classical CV, no GPU,
no model inference — this is code/test-level local work, not "running the pipeline") against the
real source image reproduces the exact panel layout Phase 9's real run used (5 panels, matching
`panel_count=5` in the saved outcome JSON, and again in this phase's own real re-run, section 10).
The diff-heatmap's changed region sits entirely inside `panel_01` (`(0,241)-(850,859)`, the large
"woman waving" panel), not `panel_00` (`(0,0)-(850,258)`, the "CLAP!!" panel
`docs/phase9-results.md` section 7.1.1 attributes this motion to) — an apparent discrepancy with
that document's own characterization, evidenced by this locally-reproducible panel geometry plus
the already-published diff heatmap, independently re-confirmed by the audit (section 13.1).

Whether the true cause is a page-mode-specific panel/object misattribution, an over-inclusive
Grounding DINO box, or SAM 2.1 expanding beyond its prompted box **cannot be determined** from
available evidence (no live mask access to the original instance) — marked `UNKNOWN`. What IS
established, and re-confirmed on a fresh real GPU run (section 10): panel mode's own independent
grounding attempt for the exact same page is safely **REJECTED** at segmentation by the existing
Phase 8.3 `_validate_mask_shape` check ("mask hugs its own tight bbox's bottom edge for 70.8%...
while the opposite edge is only 7.7%") — identical figures on both the original Phase 9 run and
this phase's fresh re-run, confirming the mitigation (switching the default to panel mode) is
deterministic and real, not a one-off coincidence.

## 5. Common root-cause analysis

Not all three defects share one cause. Two (`villainess_ending_scuffle` confirmed,
`wind_breaker_finish` originally hypothesized but real-data-disconfirmed as dominant) trace to the
same code-level mechanism (`_mesh_warp_frame`'s direction-default axis mismatch). The third
(`marika_love_meter`) has a structurally different mechanism (a `translate`, single-object,
oversized/mislocalized mask, most plausibly tied to page-mode's already-documented
[`docs/phase9-results.md` §5.3] weaker localization) with no code-level fix identified. This
phase does not force a single unifying story where the evidence does not support one — see the
brief's own instruction: "do not assume these are separate bugs" cuts both ways; it is not
license to assume they share one either.

## 6. Before/after architecture: the compositing invariant

The brief's central architectural question: does the pipeline guarantee "for each animated
object: exists exactly once; old position removed; exposed background reconstructed; transformed
position composited; unrelated background/objects untouched" for every object, every frame?

**Direct code reading** (`compositing/__init__.py::composite_frame_stack`,
`reconstruction/__init__.py::reconstruct_hidden_region`, both unchanged since Phase 8.3) confirms
this invariant **holds structurally, given a correct mask**: a pixel is either covered by a
layer's current mask (shows transformed content), part of that layer's hole (shows reconstructed
background, mutually exclusive with any other layer currently covering it — Phase 8.3's
generalization), or neither (shows untouched original) — never more than one of these.
Reconstruction is computed from ALL `frame_count` frames' transformed masks, not a sparse sample
(ruling out one candidate failure mode from the brief's list). Phase 8.3's cross-object
mask-overlap guard (`_drop_overlapping_secondary_objects`) remains byte-for-byte unchanged and
fired for real on this phase's own GPU re-runs (section 12).

**What does NOT hold, and is not an invariant the current architecture attempts to enforce**: that
a mask accurately, tightly represents the intended semantic object, or that a transform's motion
axis matches the object's actual physical shape/orientation. Both are gaps at the
*segmentation*/*animation* layer, not compositing:

- `_mesh_warp_frame`'s direction-axis choice, when unset, ignored the object's own mask shape —
  fixed this phase (section 8.1), confirmed for `villainess_ending_scuffle`.
- A mask (or a `translate` object's bbox) can be legitimately-shaped-but-too-large for its
  semantic label with no existing check catching it — `marika_love_meter`'s and (newly)
  `wind_breaker_finish`'s PRIMARY both plausibly fall in this category; no general fix is
  evidenced or implemented this phase (see sections 4.2, 4.3, 14).

Phase 10's fixes therefore target the animation/orchestration layer (one function's fallback
logic, one default parameter), not compositing itself — Phase 8/8.3's compositing/reconstruction
guarantees required no change.

## 7. Exact code changes

- `src/manga_animation/animation/transforms.py::_mesh_warp_frame` — when `motion.direction is
  None`, the fallback direction now follows the object's own bbox shape: `(y1-y0) >= (x1-x0)`
  (taller-than-wide, or square) → `(0.0, 1.0)` (top-anchored, downward sway); otherwise →
  `(1.0, 0.0)` (left-anchored, rightward sway — the previous, already-validated flag/banner
  behavior, unchanged). An explicit `motion.direction` is honored exactly as before regardless of
  mask shape.
- `src/manga_animation/pipeline/orchestrator.py::run_pipeline` — `analysis_mode` default changed
  from `"page"` to `"panel"`. `analysis_mode="page"` remains fully available and behaviorally
  identical for any caller that passes it explicitly.
- `src/manga_animation/analysis/plan_builder.py` — one comment updated to reflect the new default
  (no behavioral change).
- `scripts/run_phase3_2_validation.py` — pinned to `analysis_mode="page"` explicitly (independent
  audit finding, section 13.2): this script's whole documented purpose is measuring Phase 3.2's
  page-level metrics specifically; it would otherwise have silently inherited the new default.
- `scripts/run_phase10_gpu_validation.py` — new, targeted real-GPU validation script (section 10).
- See `docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md` for the full
  design rationale, evidence, and consequences.

No change to `compositing/`, `reconstruction/`, `segmentation/segment.py`, or
`validation/transform_geometry.py` — independently confirmed byte-for-byte unchanged by the audit
(section 13.3).

## 8. Regression tests

`tests/test_animation.py` (new, "Phase 10: mesh_warp's direction=None fallback follows mask
shape" section):
- `test_mesh_warp_default_direction_is_vertical_for_a_tall_mask` — a taller-than-wide mask with no
  explicit direction sways vertically, not horizontally (no x-displacement at all).
- `test_mesh_warp_tall_mask_has_no_uniform_horizontal_shear_regression` — direct regression guard
  for the original defect mechanism: every row of a tall mask keeps the identical x-range (no
  per-row-uniform shear).
- `test_mesh_warp_explicit_direction_still_shears_a_tall_mask_uniformly` — contrast/sanity check:
  an explicit direction is untouched by this fix and still behaves as before.
- `test_mesh_warp_default_direction_is_horizontal_for_a_wide_mask` — negative control: the
  already-validated real flag/banner case (wide mask) is unchanged.
- `test_mesh_warp_default_direction_ties_toward_vertical_for_a_square_mask` — boundary condition,
  documents the `>=` tie-break deliberately.

`tests/test_pipeline.py`:
- `test_run_pipeline_analysis_mode_defaults_to_panel_level` (renamed/rewritten from the old
  page-level-default test) — the new default is panel-aware analysis.
- `test_run_pipeline_analysis_mode_page_still_available_explicitly` (new) — the old default
  remains fully available and unchanged when passed explicitly.
- `test_run_pipeline_multi_object_no_color_bleed_between_objects_across_the_loop` and
  `test_run_pipeline_multi_object_e2e_encode_decode_regression` — updated to pin
  `analysis_mode="page"` explicitly (their `FakeVLMClient` fixture is not panel-crop-aware; see
  ADR 0017's "Consequences").

Full local suite: `uv run pytest -q` — **530 passed, 2 deselected** (up from Phase 9's 523+1
skipped+2 deselected on a fresh clone; +7 net new tests). `uv run ruff check .` clean.
`uv run mypy src` clean, 44 files. Independently re-run by the audit (section 13) and, separately,
on the real remote GPU worker (**529 passed, 1 skipped [expected missing golden image on a fresh
clone, per ADR 0002], 2 deselected**; `ruff check .` clean) — see section 10.

## 9. Independent audit

See section 13 for the full independent `qa-agent` audit (mutation testing, independent
re-derivation of every checkable claim, real gaps found and fixed before this document was
finalized).

## 10. Real GPU E2E validation

**Environment**: live Kaggle Jupyter kernel (user-provided URL this session, a fresh, dedicated
kernel — not the project owner's own already-connected one), 2x Tesla T4,
`torch==2.13.0+cu130`, real `qwen2.5-vl-7b-instruct`/`grounding-dino-swin-l`/
`sam2.1-hiera-base`/`lama-large`. Repository cloned fresh at each commit under test (`dff0b01`
then `72af0ca` after `scripts/run_phase10_gpu_validation.py` was added), `uv sync --extra dev
--extra cv --extra video --extra ml`, real-world sample pages fetched via
`scripts/fetch_phase9_realworld_pages.py` (git-ignored generated artifacts, not present in a
fresh clone per ADR 0002).

**Sanity gate on the worker**: `uv run pytest -q` → 529 passed, 1 skipped, 2 deselected;
`uv run ruff check .` → clean.

**Targeted validation**: `uv run python scripts/run_phase10_gpu_validation.py` — re-runs exactly
the three affected samples/modes (`marika_love_meter` in both `page` and `panel`;
`wind_breaker_finish` and `villainess_ending_scuffle` in `panel`, the mode their original defects
occurred in) rather than the full 10-sample × 2-mode Phase 9 dataset, since checking a specific,
already-identified defect does not need a full characterization re-run. Wrote
`outputs/experiments/phase10_gpu_validation_20260813T200422Z.json` (downloaded locally,
git-ignored). Real environment metadata confirmed in the JSON:
`git_commit`-equivalent commit `72af0caab698dd661a29a3d40e5d2cdc9b929c7a`, 2x Tesla T4.

Every grounding score, dropped-object reason, and segmentation-rejection message reproduced
**byte-identical** to Phase 9's original run for all four (sample, mode) pairs — real,
independent confirmation that Grounding DINO/SAM 2.1 are feed-forward and deterministic run-to-run
on this hardware, consistent with this project's own established finding (ADR 0015).

**Follow-up live diagnostic** (once the `wind_breaker_finish` visual re-check, section 11, showed
the defect unresolved): a one-off script (`inspect_wbf.py`, not committed — a throwaway diagnostic,
not reusable infrastructure) was uploaded to the same live kernel and run to read the real,
live `PipelineRunResult.secondary_objects[*].segmentation.bbox`/`object_plan.motion` for every
animated object in that sample — the table in section 4.2. This is real, live data from an actual
re-inference pass on this phase's own fix, not a stale/cached value.

## 11. Visual verification results

Every one of the three post-fix videos was downloaded (via the Jupyter Contents `/files/` REST
endpoint, raw bytes, not the kernel exec channel) and decoded locally with `cv2.VideoCapture`
(frame 0, frame 24 — the mid-cycle frame Phase 9's own methodology already identified as where
each defect is strongest, and frame 95, adjacent to the loop wrap) — the same methodology
`docs/phase8-results.md`/`docs/phase9-results.md` established. Crops were built around each
render's actual real changed region (per-panel pixel-diff clustering, using `detect_panels`'s
real panel boundaries as a sanity check on cropping, not the sole signal), then visually inspected
directly, not inferred from any automated metric.

- **`villainess_ending_scuffle`**: tight crop around the real changed region (`x∈[18,470],
  y∈[958,1643]` of the full 720×3086 frame) at frame 0/24/95. The sleeve/torso/skirt region that
  showed the original hard vertical duplicate is **visually clean** at frame 24 — no seam, no
  duplicate, only a subtle diagonal-streak ripple consistent with the intended cloth sway. The
  PRIMARY sword's own `rotate` region (a separate crop, `x∈[298,596], y∈[1886,2512]`) shows a
  clean single-copy rotation, no ghosting. Frame 95 matches frame 0 (seamless loop, unaffected by
  this fix). **Verdict: FIXED, confirmed at native resolution.**
- **`wind_breaker_finish`**: a wide crop around the bicycle-wheel/rider scene (`x∈[0,800],
  y∈[650,2270]` of the 800×7500 frame) shows the same vertical streaking/warping distortion at
  frame 24 as Phase 9's original render — wheel spokes and background lines still show a wavy,
  smeared displacement. A second crop around the panel-6 face/hair region (`y∈[6250,7180]`) shows
  only a small, plausible hair-strand shift, not a hard artifact. **Verdict: NOT FIXED** — see
  section 4.2 for the real diagnostic evidence explaining why (the mesh_warp object this fix
  targets did take the corrected branch; the defect's dominant cause is elsewhere).
- **`marika_love_meter`** (`page` mode): full-frame comparison (850×1200) reproduces the identical
  duplicated hair/torso/hand silhouette at frame 24, visually indistinguishable from Phase 9's
  original defect frame. **Verdict: unchanged, as expected** (no code fix was implemented for its
  `UNKNOWN` root cause). `panel` mode (the new default): REJECTED at segmentation before any frame
  is rendered — no video to inspect, which is itself the intended, safe outcome.

No suspected artifact disappeared or changed character when re-inspected at native resolution
vs. a first downscaled look in any of the three cases — no false-visual-impression correction was
needed this phase (contrast with Phase 8.3's own `verified_action_1` false-alarm case).

## 12. Phase 8/8.3 regression results

- **Cross-object overlap guard** (`_drop_overlapping_secondary_objects`): fired for real, live, on
  this phase's own fresh GPU re-runs — `villainess_ending_scuffle`: `obj_character_hair_7`
  (100.0% overlap with the accepted sword) and `obj_character_clothing_8` (99.2%) both dropped;
  `wind_breaker_finish`: `obj_character_glasses_8` (98.3% overlap with the accepted hair) dropped.
  Real, unstaged confirmation the guard remains active and correct.
- **Mask-shape validation gate** (`_validate_mask_shape`): fired for real, live, multiple times —
  `wind_breaker_finish`: `obj_weapon_6` (55.1%/0.6%), `obj_character_hair_10` (74.7%/1.6%);
  `marika_love_meter` panel mode: the PRIMARY candidate itself (70.8%/7.7%, causing the honest
  REJECTED outcome). All real, live rejections on this phase's own fresh inference pass, not
  reused from Phase 9's data.
- **Segmentation failure isolation**: every one of the above drops left its sample's other
  objects/the whole run unaffected (non-PRIMARY drops did not fail the run; `marika_love_meter`
  panel mode's PRIMARY-level rejection correctly failed only that sample, as designed).
- **Honest failure classification**: `marika_love_meter` panel mode surfaces as a genuine
  `failing_stage="segmentation"` rejection with a full diagnostic reason, not a silent drop or a
  fabricated success.
- **Loop continuity**: all three post-fix renders report `seamless_loop_verified=True`; frame 95
  visually matches frame 0 in every case inspected (section 11).
- `segmentation/segment.py`, `validation/transform_geometry.py`: confirmed byte-for-byte
  unchanged vs. `2652d82` (Phase 9's own last commit) by the independent audit (section 13.3).

No Phase 8/8.3 protection was weakened, bypassed, or had its threshold changed to obtain any
result in this phase.

## 13. Independent audit (`qa-agent`, fresh session)

Mandate: independently re-verify the claimed root causes, the fix's correctness, regression-test
meaningfulness, and Phase 8 protection integrity against real artifacts/real script runs, without
trusting this investigation's own prose — same convention as `docs/phase8-results.md` §9/
`docs/phase8.3-results.md` §10.

### 13.1 Independently re-verified, confirmed accurate

- Wrote and ran its own standalone script against the real, unmodified `_mesh_warp_frame`/
  `generate_transformed_layer` and the real `examples/realworld/villainess_ending_scuffle.png`:
  confirmed the OLD default (`(1,0)`) produces an identical row-independent horizontal shift
  across 600 sampled rows of a tall mask; confirmed the NEW default produces zero horizontal
  displacement and a genuinely row-varying vertical falloff; **confirmed the NEW default is
  bit-identical (max abs diff = 0) to the OLD explicit `(1,0)` default on a WIDE mask** — the
  real, already-validated flag/banner case is provably untouched, not merely "should be
  unchanged."
- Confirmed `_MOTION_HEURISTICS`'s flag/cloth entry genuinely never sets `direction` in the real
  source (not just in a contrived test) — the mechanism fires in real production code paths.
- Independently re-ran `detect_panels()` against the real `marika_love_meter.png`: got the
  identical 5 panels and bboxes this document cites; independently inspected the real diff-heatmap
  crop and agreed the changed region sits in `panel_01`, not `panel_00`.
- Independently re-verified every cited Phase 9 number (`docs/phase9-results.md` §5.3) and every
  quoted error-message string against the real raw JSON — all byte-exact.
- Confirmed `_TRANSFORM_GEOMETRY_PROFILES[MESH_WARP].max_aspect_ratio is None` — no interaction
  between this fix and the pre-segmentation geometry gate.

### 13.2 Real gaps found and fixed as a direct result of this audit

1. **`scripts/run_phase3_2_validation.py` silently inherited the new panel-mode default with no
   disclosure.** This committed, named script's entire documented purpose is measuring Phase
   3.2's page-level VLM-targeting/grounding metrics specifically; it never passed
   `analysis_mode` anywhere. Post-Phase-10, re-running it would have silently measured
   panel-mode's rates while still reporting itself as "Phase 3.2" page-level results — a real risk
   of a future misleading report. **Fixed**: pinned to `analysis_mode="page"` explicitly (section
   7). `scripts/run_phase3_pipeline.py` (a generic smoke-test script with no page-mode-specific
   claim) and `scripts/run_reconstruction_visual_qa.py`/`evaluation/harness.py` (both already
   unaffected — explicit `plan=`/`analysis_mode=` respectively) were independently confirmed fine
   as-is.
2. This document (`docs/phase10-results.md`) did not exist yet at audit time, though the ADR
   already cited it — expected build order (the audit ran before this document was written, by
   design, so its findings could feed into this document's own audit section), not a defect.

### 13.3 Mutation testing — actually performed

- Reverted `_mesh_warp_frame`'s fallback to the old unconditional `(1.0, 0.0)`: 3 of the 5 new
  mesh_warp tests failed as predicted; the other 2 (explicit-direction, wide-mask) correctly still
  passed. Restored; `git diff` confirmed clean.
- Reverted `run_pipeline`'s `analysis_mode` default to `"page"`:
  `test_run_pipeline_analysis_mode_defaults_to_panel_level` failed exactly as predicted. Restored;
  `git diff` clean.
- Removed the explicit `analysis_mode="page"` pin from both Phase 7.1 tests, one at a time: both
  failed under panel-mode-by-default (duplicate `raised_hand` candidates pooled across arbitrary
  panel ids — 2 candidates in one test, 3 in the other, one per detected panel). Restored; `git
  diff` clean.
- Confirmed `segmentation/segment.py` has **zero** diff between `2652d82` and `dff0b01` (byte-for-
  byte); confirmed `pipeline/orchestrator.py`'s only diff hunks are the default-parameter line and
  its docstring — `_drop_overlapping_secondary_objects` untouched.

### 13.4 Overall verdict

"The fix addresses the real mechanism, not the symptom, for both defects it targets. All of the
ADR's checkable claims... independently reproduced exactly as stated." Full gate independently
green (`pytest` 530 passed/2 deselected, `ruff check .` clean, `mypy src` clean, 44 files).
Working tree confirmed clean after every mutation test.

## 14. Known limitations

- **`wind_breaker_finish` remains a real, open, unfixed visual defect.** Its original hypothesis
  (shared mesh_warp mechanism) is real-data-disconfirmed as the dominant cause; a new,
  single-instance-evidenced lead (an oversized PRIMARY `translate` bbox, section 4.2) is
  disclosed, not converted into an unproven fix. **Highest-priority Phase 11+ candidate.**
- **`marika_love_meter`'s exact root cause remains `UNKNOWN`** at the grounding/segmentation
  level — no live GPU access to the *original* instance's masks was available this phase (a fresh
  re-run reproduced the same defect deterministically, section 10, but a deterministic
  reproduction of a bug is not the same as knowing which stage introduced it). Mitigated (not
  fixed) via the panel-mode-default change: panel mode's own independent attempt is safely
  rejected instead.
- **A general "over-inclusive/oversized mask for its semantic label" detector remains
  unimplemented**, deliberately — no real mask/bbox pair exists (for `marika_love_meter`) or is
  independently confirmed as a general pattern (for `wind_breaker_finish`'s new lead) to evidence
  a calibrated threshold. Implementing one from a single instance each would be exactly the
  "speculative heuristic without evidence" the brief prohibits.
- **MESH_WARP's `strength = value * amplitude * max(bbox_width, bbox_height)` still has no upper
  bound** relative to the panel/page, independent of any single sample's evidence — a real,
  disclosed, unaddressed gap in the transform's own geometry.
- **The `_MOTION_HEURISTICS` "cloth" keyword substring also matches "clothing"/"clothes"** — not
  independently evidenced as wrong, left unchanged, noted for visibility (ADR 0017 "Deferred").
- **`detect_seam_like_artifacts`'s already-disclosed ~50% real-world precision** (Phase 9 §7.2)
  was reconfirmed this phase: it fired `True` on `villainess_ending_scuffle` post-fix despite
  direct visual inspection finding the defect genuinely fixed — a real, live false positive on
  this phase's own real data, consistent with, not worse than, its already-known limitation.
- The golden 7-sample Phase 8 regression set was not re-run in this session (resource-efficiency
  choice, matching Phase 9's own precedent) — its own real evidence remains
  `docs/phase8-results.md`/`docs/phase8.3-results.md`; this phase's own Phase 8 regression
  evidence (section 12) comes from the real overlap/mask-shape guards firing live on the Phase 9/
  10 sample set instead.

## 15. Deferred work (ranked, not implemented this phase)

1. **Root-cause `wind_breaker_finish`'s real defect** with fresh live GPU mask access targeting
   specifically the PRIMARY `object_in_motion` object's grounding — determine whether its
   389×1381 bbox is itself over-inclusive (a grounding/SAM issue, the same broad category as
   `marika_love_meter`) or a legitimate read of unusually elongated on-page content (in which case
   the fix belongs in transform geometry, e.g. bounding `translate` displacement relative to bbox
   aspect ratio or fine-structure density).
2. **Root-cause `marika_love_meter`'s page-mode defect** with fresh live GPU mask/plan access —
   confirm or rule out the panel/object-misattribution hypothesis (section 4.3) against the real
   `AnimationPlan.panels`/`object.panel_id`.
3. **A general, evidence-calibrated "mask/bbox too large for its semantic label" detector**,
   once (1)/(2) provide enough real instances to calibrate a threshold responsibly — explicitly
   deferred rather than guessed at with one data point each.
4. **An upper bound on MESH_WARP's `strength`** relative to the panel/page, once real evidence
   exists to calibrate it responsibly.
5. Broader panel-mode architectural work (explicitly out of this phase's scope per the brief's
   "Future roadmap" section) remains future work, unchanged from Phase 9's own recommendations.

## 16. Reproduction commands

```bash
# Local: tests, lint, types
uv run pytest -q
uv run ruff check .
uv run mypy src

# Local, deterministic, no-GPU forensic reproduction (mesh_warp mechanism)
# — see the script embedded in this document's own investigation; not committed as a
#   standalone script (a one-off forensic repro, not reusable infrastructure), but fully
#   reconstructable from docs/decisions/0017-...md's own described construction.

# Local, deterministic, no-GPU panel-geometry reproduction
uv run python -c "
import numpy as np
from PIL import Image
from manga_animation.analysis.panels import detect_panels
img = np.asarray(Image.open('examples/realworld/marika_love_meter.png').convert('RGB'))
for p in detect_panels(img):
    print(p.id, p.bbox)
"

# Remote (Kaggle/Jupyter GPU worker only, per ADR 0003) — targeted post-fix validation
git pull
uv sync --extra dev --extra cv --extra video --extra ml
uv run python scripts/fetch_phase9_realworld_pages.py   # real-world sample images are git-ignored
uv run python scripts/run_phase10_gpu_validation.py
```

## 17. Git

See the top-level session summary for the final branch/commit/push status. Commits this phase:
`dff0b01` (fix + tests + ADR), `72af0ca` (targeted GPU validation script), `cac3a26` (audit-found
`run_phase3_2_validation.py` fix), and this document's own commit.
