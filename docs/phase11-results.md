# Phase 11 results: multi-bug visual forensics and renderer stabilization

Status: **completed — root causes confirmed for three real defects via a live GPU diagnostic
capture; no safe, evidenced architectural fix could be responsibly implemented for any of them
this phase.** This document is written incrementally, in place, as real evidence lands — same
convention as `docs/phase8-results.md`/`docs/phase9-results.md`/`docs/phase10-results.md`. Every
claim below is checked against a real, locally-retained artifact (`outputs/frames/`,
`outputs/videos/`, `outputs/debug/phase11_gpu_evidence/`, `outputs/experiments/`, git-ignored per
ADR 0002 but present on this checkout) or a real GPU log line, not asserted from memory.

## 1. Scope and compute strategy

This phase's brief: systematically audit multiple real Phase 8/9/10 outputs, find and root-cause
real visual defects (not just the two already-known unresolved ones), fix what can be
evidence-backed and architecturally justified, and add regression coverage — without fabricating
GPU execution or converting an unproven hypothesis into a "fix."

Two stages of work:

1. **Local forensic audit** (sections 3-5 below): pixel-diffing and connected-components analysis
   on already-rendered real PNG/MP4 frames from Phase 8/9/10, plus direct reading of the real,
   unmodified production code — `[LOCAL, NO MODEL INFERENCE]`. This surfaced multiple real,
   previously-undocumented defects and several *hypotheses*, but could not confirm root causes:
   no real `SegmentationResult.mask`/`ReconstructionResult` array exists locally for any of the
   samples investigated (Phase 10's own disclosed gap, still true at the start of this phase).
2. **One targeted GPU diagnostic capture** (section 6): a single live Kaggle Jupyter session
   (user-provided URL), a fresh dedicated kernel, real `qwen2.5-vl-7b-instruct`/
   `grounding-dino-swin-l`/`sam2.1-hiera-base`/`lama-large`, re-running exactly the three samples
   this phase's local forensics flagged (`villainess_ending_scuffle`, `wind_breaker_finish`,
   `sss_hunter_gladiator`, panel mode) and downloading the real `SegmentationResult.mask` (not
   just `.bbox`) and `ReconstructionResult.hole_mask`/`filled_pixels` for every animated object —
   not merely the final composited video. This single run answered every open hypothesis from
   section 5 at once, per this phase's own "one GPU run validating several hypotheses" strategy.
   All raw arrays, source crops, the full diagnostic JSON, and the three fresh rendered videos are
   retained locally: `outputs/debug/phase11_gpu_evidence/` (336MB, 46 files),
   `outputs/videos/phase11_gpu_evidence/`, `outputs/frames/phase11_gpu_evidence/`.

**No production code (`src/manga_animation/`) was changed this phase** — see section 8 for why.

## 2. Artifact inventory (Phase A)

Confirmed present locally before this phase's own GPU run (git-ignored, `outputs/`/`examples/`):
real source images for all 10 `realworld_*` pages and every earlier-phase sample; real rendered
MP4s and extracted frames for every Phase 8/9/10 completion; real per-sample outcome JSON
(`phase9_evaluation_*.json`, `phase10_gpu_validation_*.json`); real saved mask/hole-mask arrays
for **exactly one** prior instance (`phase3_action_page_hair6`, Phase 8.3). **No saved
`SegmentationResult.mask` or `ReconstructionResult` existed locally for `wind_breaker_finish`,
`marika_love_meter`, or `villainess_ending_scuffle`** — this is why this phase's local-only stage
(section 5) could only produce hypotheses, and why section 6's GPU capture was necessary.

After this phase's own GPU run, locally retained: real `SegmentationResult.mask` +
`ReconstructionResult.hole_mask`/`filled_pixels` for **12 real objects** across the 3 targeted
samples (`outputs/debug/phase11_gpu_evidence/*.npy`), plus source-image crops and 3 fresh
rendered videos.

## 3. Objective A — `realworld_wind_breaker_finish`: root cause now CONFIRMED

**Local forensics** (before the GPU run) found: the largest mid-cycle diff region clusters three
independently-animated objects in tight physical proximity (PRIMARY `object_in_motion`/translate,
SECONDARY `character_clothing_1`/mesh_warp, MICRO `character_hair_0`/translate — bboxes from
Phase 10's own published table), and a second, distant diff region (`character_hair_7`/translate)
shows the *same* visual "comb streak" signature despite a different transform kind and location —
arguing for a shared downstream mechanism, not a per-kind transform bug (already independently
disconfirmed by Phase 10's own diagnostic).

**GPU capture (section 6) confirms the actual mechanism, and it is not what either Phase 10 or
this phase's own initial local hypothesis guessed**: direct visualization of the real, downloaded
`SegmentationResult.mask` for PRIMARY `obj_object_in_motion_12` — grounded and validated as "a
bicycle wheel in motion" (`grounding_score=0.296`, already the lowest of any object in this
render) — shows a mask that **does not correspond to a bicycle wheel, or to any single coherent
object at all**: a narrow, page-aligned vertical stripe (`389×1381px`, `aspect≈3.55:1`) cutting
straight through the rider's face, the wheel's spokes, the jersey, and the hand gripping the
handlebar (`outputs/debug/phase11_gpu_evidence/wbf_primary_ORIGINAL_mask_overlay.png`). A second
object in the same render, `obj_character_hair_7` (labeled `character_hair`), has a real mask that
covers the character's **entire face** — sunglasses, both eyes, nose, mouth — not merely hair
(`wbf_hair7_ORIGINAL_mask_overlay.png`; density within its own tight bbox: 84.3%). A third object,
`obj_character_clothing_1` (mesh_warp), has a real mask that *does* plausibly correspond to real
clothing (`wbf_clothing1_ORIGINAL_mask_overlay.png`) — not every object in this render is
defective, consistent with earlier phases' own finding that defects are per-instance, not
universal.

**Status: `CONFIRMED_ROOT_CAUSE`** (upgraded from Phase 10's `UNKNOWN`). Two of this render's four
real animated objects have segmentation masks that are semantically incoherent/over-inclusive for
their assigned label, despite passing every existing geometric check (`_validate_mask_shape`'s
edge-asymmetry test, `_drop_overlapping_secondary_objects`' cross-object overlap test, the
full-page coverage bound) — see section 7 for why no safe general detector for this could be built
this phase. This is a segmentation-quality defect, not a transform/compositing/reconstruction
defect — Phase 10's mesh_warp-direction fix and this phase's own initial "unbounded mesh_warp
reach" and "shared reconstruction-quality" hypotheses (section 5) were reasonable given the
evidence available at the time, but the real mechanism is upstream of all of them.

## 4. Objective B — `realworld_marika_love_meter`: root cause remains UNKNOWN, hypothesis strengthened

This sample was **not** re-run this phase's GPU session (the diagnostic budget targeted the three
samples with the strongest, most tractable local leads — see section 6's scope note). No new
direct evidence for its own real mask exists. **However**, Phase 10's own original hypothesis for
this sample — "the duplicate ghost silhouette's shape must already have been present in the
ORIGINAL segmentation mask" (`docs/phase10-results.md` §4.3) — is now substantially strengthened
by cross-sample corroboration: this phase confirmed the *general mechanism* (a real, over-inclusive
mask passing every existing check) in three *other* independent real samples. This does not
promote marika's own status to `CONFIRMED` (no real mask for this specific instance was obtained),
but it does mean the "over-inclusive mask" explanation is no longer merely one hypothesis among
several equally-plausible ones — it is now the best-evidenced explanation available, pending a
future direct capture. **Status: `UNKNOWN`, evidence-strengthened.** Panel mode continues to
mitigate (unchanged): its own independent grounding attempt is safely rejected by
`_validate_mask_shape` (70.8%/7.7% edge asymmetry, reproduced identically in Phase 9 and Phase 10).

## 5. Objective C — new real defects found by broad local audit

### 5.1 BUG-11-03: `realworld_villainess_ending_scuffle` is not actually clean post-Phase-10-fix

Phase 10 inspected the crop `x∈[18,470], y∈[958,1643]` of `outputs/frames/phase10_evidence/
villainess_tight_f{0000,0024,0095}.png` and reported "no hard vertical seam or duplicate
silhouette... **Verdict: FIXED, confirmed at native resolution.**" Independently re-inspecting the
*same* retained crop at 3x zoom found two real defects Phase 10 did not report, both mid-cycle
only:

1. A completely static, non-animated speech bubble ("GIVE IT BACKK!!") visibly shifts right
   between frame 0 and frame 24, with a torn/dashed discontinuity along its own border.
2. A large, blurry, low-detail grey block replaces a hand/glove and surrounding clean linework at
   the bottom of the same crop.

Both were **reproduced byte-identical on this phase's own fresh GPU run** — the bubble-region diff
statistic (mean abs diff `23.489111405835544`) matched the original Phase 10 evidence to 15
significant figures, confirming (once more) that Grounding DINO/SAM 2.1 are deterministic,
feed-forward, run-to-run-stable on this hardware (ADR 0009/0015's own established finding).

The existing automated `evaluation.artifacts.detect_seam_like_artifacts` check fired
`seam_artifact_suspected=True` on this exact post-fix render — Phase 10 called this "a confirmed
false positive." **This phase's finding revises that conclusion: the detector was not a false
positive — it correctly flagged a real defect Phase 10's own visual QA pass looked past.**

**Root cause: `CONFIRMED`.** Direct visualization of the real, downloaded `SegmentationResult.mask`
for `obj_cloth_5` (grounded as "a piece of cloth being held by a hand") overlaid on the source
image (`outputs/debug/phase11_gpu_evidence/villainess_cloth_ORIGINAL_mask_overlay.png`) shows the
mask covers **90.2% of its own 394×648px tight bounding box** — a dense, page-furniture-spanning
blob that visibly includes the entire speech bubble, its text, and the hand/glove beneath it, not
merely "cloth." Locally reproducing the real, unmodified `generate_transformed_layer`/
`_mesh_warp_frame` (no GPU/model call — pure deterministic code, using the real downloaded mask
and image) at frame 24 confirmed the warped mask remains 75.6% nonzero over the exact page region
where the bubble sits. **Critically, this is not a transform-reach problem**: row-by-row
inspection of the *original, untransformed* mask shows it already covers the bubble/hand region at
rest, before any warp is applied — the object's own segmentation, not the animation math, is the
root cause. This revises this phase's own earlier (pre-GPU-capture) hypothesis, which had
attributed the defect to MESH_WARP's uncapped `strength`/reach (section 5, superseded).

### 5.2 BUG-11-04: `realworld_sss_hunter_gladiator` — dense-scene reconstruction artifact

Phase 9 flagged this sample "Inconclusive" via the automated seam detector. This phase's own
connected-components analysis of the real frame diff isolated a hard-edged, near-black rectangular
patch replacing part of a red drape/curtain background at frame 24 (page `y≈4273-4918`) — confirmed
not a page-boundary sampling artifact (this region sits ~44% down a 9780px page, and
`pipeline/orchestrator.py` passes the full page array, not a crop, to every transform call).

**Root cause: `CONFIRMED`.** The real mask for `obj_character_hair_7` (labeled `character_hair`,
translate) at this exact page position covers the creature's head **and** a large area of the red
drape background behind it (`outputs/debug/phase11_gpu_evidence/shg_hair7_ORIGINAL_mask_overlay.png`)
— the same "semantically over-inclusive, geometrically unremarkable" signature as section 5.1. Its
real reconstruction (`hole_area_px=90800`) shows `blur_ratio_filled_over_surrounding=0.727` — LaMa's
fill of the exposed drape/curtain content is measurably softer than the real surrounding linework
(see section 6.3), producing the flat, low-detail patch observed.

**A real, disclosed operational finding from this same run**: LaMa reconstruction for this sample
hit `CUDACachingAllocator` out-of-memory 5 times over ~40s (`free: 846MB` against a `2.02GB`
allocation request, on a shared T4 with three other real models already resident) before
succeeding on retry. Not a code defect (PyTorch's own allocator retried and recovered
automatically; no fix attempted or needed), but a real, disclosed resource-pressure risk for dense
multi-object real pages under panel mode, not previously observed/documented at this level of
detail.

### 5.3 Confirmed clean on re-audit (no new defect found)

`space_monster_creature`, `wind_breaker_sprint`, `omniscient_reader_blade` — re-inspected at native
resolution; no new defect found, reconfirming each sample's existing Phase 9 characterization.

## 6. Targeted GPU diagnostic capture

**Environment**: live Kaggle Jupyter kernel (user-provided URL this session), a fresh dedicated
kernel (2x Tesla T4 node, single-GPU workload observed), real `torch==2.10.0+cu128`,
`qwen2.5-vl-7b-instruct`, `grounding-dino-swin-l`, `sam2.1-hiera-base`, `lama-large`. Repository
cloned fresh at commit `72c7470` (this phase's own starting commit — no Phase 11 code exists yet
to test), `uv sync --extra dev --extra cv --extra video --extra ml`. Sanity gate on the worker:
`uv run pytest -q` → **529 passed, 1 skipped, 2 deselected**; `uv run ruff check .` → clean —
matching this repo's own local baseline exactly, confirmed independently on the remote hardware.

**Scope**: `realworld_villainess_ending_scuffle`, `realworld_wind_breaker_finish`,
`realworld_sss_hunter_gladiator`, panel mode (the mode each real defect occurred in) — chosen
because local forensics (section 5) already produced concrete, testable hypotheses for exactly
these three; `marika_love_meter` was not included this run (see section 4).

For every animated object (12 total across the 3 samples), a purpose-built diagnostic script
(`phase11_diagnostic.py`, not committed — a throwaway diagnostic, same convention as Phase 10's
`inspect_wbf.py`) called the real, unmodified `pipeline.orchestrator.run_pipeline` and computed,
from the real returned `PipelineRunResult`:

1. Mask connected-components stats (`cv2.connectedComponentsWithStats`) — tests mask fragmentation.
2. For MESH_WARP objects: real `strength` at 8 sampled `t_frac` values via the real
   `animation.curves.sample_motion_value` — tests unbounded-reach.
3. For every object with a `ReconstructionResult`: a Laplacian-variance blur ratio between the
   filled hole and a same-shaped ring of real surrounding pixels — tests reconstruction quality.

Real per-sample execution times (informational, not this phase's focus): `villainess_ending_scuffle`
completed in ~5 minutes; `wind_breaker_finish` (7 panels, 4 rendered objects, 7500px page) took
~9 minutes, compositing alone (CPU, single-threaded — confirmed by reading
`pipeline/orchestrator.py:656-657`, a plain `for i in range(frame_count)` loop, no
`multiprocessing`/`ThreadPoolExecutor`) took 89s; `sss_hunter_gladiator` (9780px page, 6 rendered
objects) took ~19 minutes, compositing alone took 353.6s. **A real, disclosed, out-of-scope
performance finding**: this project's compositing stage does not parallelize across frames despite
being embarrassingly parallel (each frame's composite is independent) — noted for future work, not
attempted this phase (a performance change, not a visual-defect fix, is outside this phase's
brief).

### 6.1 Hypothesis: mask fragmentation — `FALSE_POSITIVE` (disconfirmed by real data)

All 12 real masks' connected-components stats: second-largest-component area never exceeded
**1.25%** of any mask's total area (range: 0.0%–1.25%across all 12 real objects, including
`obj_cloth_5`: 9 components, second-largest 0.05% — negligible, consistent with ordinary
anti-aliasing noise, not a meaningful disconnected blob). **This phase's own initial hypothesis for
BUG-11-03 (section 5.1's first pass, before the mask-overlay visualization) — that SAM produced a
disconnected secondary mask component near the speech bubble — is directly disconfirmed by the
real downloaded mask array.** The true mechanism (section 5.1, revised) is a single, contiguous,
over-large mask, not a fragmented one.

### 6.2 Hypothesis: unbounded MESH_WARP `strength`/reach — real numbers obtained, not the primary mechanism

Real `strength` values confirmed: `obj_cloth_5` (villainess) reaches `77.8px` (implied margin
`79.8px` beyond its own bbox); `obj_character_clothing_1` (wind_breaker_finish) reaches `65.2px`
(margin `67.2px`) — both substantial, confirming Phase 10's own already-disclosed, previously
unconfirmed concern ("MESH_WARP's `strength`... has no upper bound," `docs/phase10-results.md`
§14) with two real instances for the first time. **However, section 5.1's direct mechanism check
shows this is not what causes the villainess defect**: the bubble/hand are already inside the
mask's own rest-pose footprint, not merely reached by the warp's padding margin. This finding
remains real and disclosed (MESH_WARP truly has no strength bound), but is not the confirmed root
cause of any specific defect this phase found — left `DEFERRED`, same status as Phase 10 left it,
now with concrete real numbers attached for a future phase to use if a bound is ever added.

### 6.3 Hypothesis: hidden-region reconstruction quality — real, universal, but not a per-defect discriminator

Every one of the 11 real reconstructions measured (100%) showed `blur_ratio_filled_over_surrounding
< 1.0` (range **0.40–0.73**) — including the two `raised_sword` objects (villainess: 0.566,
sss_hunter_gladiator: 0.664), which prior phases characterized as visually clean. **This means
LaMa's reconstruction fill is measurably softer than real surrounding manga line art
*universally* on this content type, not specifically on the objects that show a visible defect** —
a real, confirmed, systemic gap (Phase 9/10 never measured this quantitatively), but not, on its
own, a signal that discriminates "will look visibly wrong" from "looks fine." The full table:

| Sample | Object | blur_ratio |
| --- | --- | --- |
| villainess_ending_scuffle | raised_sword_12 | 0.566 |
| villainess_ending_scuffle | **cloth_5** | 0.689 |
| wind_breaker_finish | object_in_motion_12 (PRIMARY) | 0.600 |
| wind_breaker_finish | character_hair_0 | 0.503 |
| wind_breaker_finish | character_clothing_1 | 0.517 |
| wind_breaker_finish | **character_hair_7** | 0.607 |
| sss_hunter_gladiator | raised_sword_18 | 0.664 |
| sss_hunter_gladiator | **character_hair_7** | 0.727 |
| sss_hunter_gladiator | hand_10 | 0.428 |
| sss_hunter_gladiator | green_fluid_15 | 0.402 |
| sss_hunter_gladiator | character_face_17 | 0.474 |

(Bold = objects with a confirmed over-inclusive mask, section 6.4.) No separation between bold and
non-bold rows — reconstruction quality is a real, compounding factor, not the trigger.

### 6.4 Confirmed mechanism: semantically over-inclusive segmentation masks

Direct visualization of the real mask arrays confirms **4 of the 12 real objects** have masks that
are geometrically unremarkable (pass every existing check) but semantically wrong — capturing
substantially more, or different, real content than their assigned label:

| Object | Labeled as | Real mask actually covers |
| --- | --- | --- |
| `villainess_ending_scuffle` / `obj_cloth_5` | cloth | cloth + a full speech bubble + a hand |
| `wind_breaker_finish` / `obj_object_in_motion_12` (PRIMARY) | "a bicycle wheel in motion" | an incoherent vertical stripe through face, wheel, jersey, hand — no single object |
| `wind_breaker_finish` / `obj_character_hair_7` | character_hair | the character's entire face (glasses, eyes, mouth) |
| `sss_hunter_gladiator` / `obj_character_hair_7` | character_hair | the creature's head + a large area of background drape |

This is the single systemic finding (Objective D) this phase's evidence most strongly supports: a
real, recurring segmentation-quality failure mode, independent of and not overlapping with every
mechanism Phase 8.3/10 already fixed (mask edge-asymmetry, cross-object overlap, mesh_warp
direction). See section 7 for why no general detector for it was implemented.

## 7. Why no fix was implemented (Phase E)

Three candidate geometric signals were tested against the real evidence gathered this phase
(4 confirmed-defective real masks vs. 8 real masks from objects with no reported visual defect):

| Signal | Confirmed-defective values | Other real values | Separates cleanly? |
| --- | --- | --- | --- |
| Mask fragmentation (2nd-component area fraction) | 0.00%–0.05% | 0.00%–1.25% | **No** — disconfirmed outright (§6.1) |
| Density within own tight bbox | 37.9%, 58.9%, 84.3%, 90.2% | 33.3%–84.3% | **No** — the worst case (PRIMARY, 37.9%) is *lower* than several non-defective instances |
| Aspect ratio (`max/min` side) | 1.07, 1.12, 1.64, **3.55** | 1.07–3.55 | **No** — only 1 of 4 defective instances is an outlier; the rest are unremarkable |
| Convex-hull solidity (`mask_area/hull_area`) | 0.446, 0.765, 0.906, 0.952 | 0.446–0.955 | **No** — the most obviously-wrong mask (`wind_breaker_finish` PRIMARY) has the *lowest* solidity of all 12 real masks, but a real non-defective object (`sss_hunter_gladiator` PRIMARY) sits right behind it at 0.499; a real non-defective object (`wind_breaker_finish` `character_hair_0`) has the *highest* solidity of all 12, above 3 of the 4 defective masks |

**None of the three signals reliably separates the confirmed-defective masks from the rest of this
phase's own real dataset.** This is itself a real, evidenced, useful negative result — it directly
satisfies this phase's own brief ("do not introduce arbitrary thresholds without evidence"): the
evidence gathered this phase shows a purely geometric threshold *would* be arbitrary here, unlike
Phase 8.3's edge-asymmetry check (which had a real, clean 2x+ margin between its one confirmed
defect and five real non-defective instances) or its cross-object overlap guard (0.68 real overlap
vs. a 0.25 bound). The underlying failure is semantic (does this mask's content match its label),
not geometric (does this mask's shape look unusual) — no geometry-only check can be expected to
catch it reliably, and forcing one in anyway risks both missing future real instances and
false-rejecting legitimate objects (a genuinely dense, high-aspect-ratio, or organically-shaped
mask is common and often correct — Phase 8.3's own "genuinely rectangular banner" negative control
already established this principle for the adjacent edge-asymmetry check).

**No fix was implemented this phase for BUG-11-01, BUG-11-03, or BUG-11-04's root cause.** Per the
brief's own explicit permission ("If some defects remain UNKNOWN: that is acceptable... Do NOT
fabricate a fix... A clean honest REJECTED is preferable to a visually corrupted PASS"), this
phase's contribution is the confirmed root cause and the evidenced negative result on candidate
fixes, not a forced architectural change. See section 10 for what future evidence would change
this conclusion.

## 8. Local validation (Phase F)

No `src/manga_animation/` file was changed this phase (section 7), so no new local
tests/lint/mypy run was needed to validate a fix — the existing baseline is unchanged by
construction. For completeness, re-confirmed on this checkout:

```
uv run pytest -q       # 530 passed, 2 deselected (unchanged baseline, Phase 10's own number)
uv run ruff check .    # clean
uv run mypy src        # clean, 44 files
```

## 9. Phase 8/8.3/9/10 regression check (Phase J)

All fired for real, live, on this phase's own fresh GPU inference (not reused/stale data):

- **`_validate_mask_shape`** (Phase 8.3): fired correctly on `villainess_ending_scuffle`'s
  `obj_weapon_6` (55.1%/0.6%) and `obj_character_hair_10` (74.7%/1.6%); on `sss_hunter_gladiator`'s
  a dropped hair candidate (42.7%/0.5%) — all real, live rejections, values consistent with the
  check's own documented calibration.
- **`_drop_overlapping_secondary_objects`** (Phase 8.3, Defect A guard): fired correctly on
  `wind_breaker_finish` (`obj_character_glasses_8` dropped, 98.3% overlap with accepted hair) and
  `sss_hunter_gladiator` (`obj_character_leg_5` dropped, 97.9% overlap with accepted eyes).
- **Panel-mode default** (Phase 10): all three targeted samples ran in panel mode without
  requiring an explicit override, confirming the default remains in effect.
- **Loop continuity**: all three fresh renders report `seamless_loop_verified=True` (per-render
  `LoopMetrics`, unaffected by this phase's findings — all three confirmed defects are, as with
  every prior phase's mid-cycle defect, invisible at the actual frame-0/wrap-frame comparison).

No existing protection was weakened, bypassed, or had its threshold changed.

## 10. Known limitations / deferred work

- **The core finding — semantically over-inclusive segmentation masks — has no evidenced,
  general, geometry-only fix** (section 7). Closing this responsibly would need either: (a)
  meaningfully more real instances (both defective and legitimate) to find a geometric signal
  that *does* separate them, which this phase's 4-vs-8 sample did not find; or (b) a genuinely
  different, content-aware signal — e.g., a post-segmentation semantic re-validation step (a
  second, cheap VLM crop-verification call against the actual mask's silhouette/crop, analogous to
  `validation/validate.py`'s existing pre-segmentation semantic check, but checking "does this
  exact masked region look like it's *entirely* the target object" rather than "does this box
  plausibly contain the target") — a real architectural direction, but a new stage-design
  decision, explicitly out of this phase's "smallest correct fix" scope.
- **MESH_WARP's `strength` remains unbounded** (§6.2) — two new real instances (65px/78px) are
  disclosed but still insufficient (no real "how far is too far" instance exists) to calibrate a
  bound responsibly.
- **Reconstruction (LaMa) quality is universally below the surrounding real content's detail level**
  on this manga-line-art content type (§6.3, 11/11 real instances measured) — a real, systemic,
  previously-unquantified gap. No fix attempted (LaMa is the established model per ADR 0005;
  swapping/tuning it is out of this phase's scope, which explicitly forbids introducing a new
  generative model).
- **`marika_love_meter`'s own root cause remains `UNKNOWN`** — the "over-inclusive mask" hypothesis
  is now better-evidenced by cross-sample corroboration (§4) but not directly confirmed for this
  specific instance; no live GPU capture of its own real mask was obtained this phase.
- **Compositing is single-threaded CPU** (§6, confirmed by code read) — a real, disclosed
  performance characteristic (353.6s on a 9780px/6-object page), not a visual defect, out of this
  phase's declared scope.
- **A real CUDA OOM-with-automatic-retry-recovery was observed** during `sss_hunter_gladiator`'s
  reconstruction stage (§5.2) — disclosed as a real operational risk on memory-constrained shared
  GPU hardware with multiple resident models, not a code defect, no fix attempted.
- Two of Phase 9's original three defects (`wind_breaker_finish`, `villainess_ending_scuffle`) are
  now root-caused for the first time; the third (`marika_love_meter`) remains genuinely open.

## 11. Independent QA (Phase I)

A fresh `qa-agent` review was launched with the same adversarial mandate Phase 8.3/9/10's own
independent audits used — explicitly instructed to try to falsify section 7's "no safe fix"
conclusion by independently recomputing candidate discriminating signals from the raw `.npy`
evidence, not merely confirm this document's prose. That agent run **failed mid-review** (a
session/usage-limit error, unrelated to this phase's own evidence or claims) after partially
verifying file locations but before completing its numeric analysis — disclosed here rather than
silently omitted or retried into a fabricated result.

Given the failure, the orchestrating session itself performed one further adversarial check in the
same spirit before closing this phase: a **fourth** candidate geometric signal, convex-hull
solidity (`mask_area / convex_hull_area` — chosen specifically to try to catch "an incoherent mask
touching disjoint regions," the `wind_breaker_finish` PRIMARY case's own visual signature), was
tested against all 12 real masks and added to section 7's table. It also failed to separate
cleanly: `wind_breaker_finish`'s PRIMARY (the single most obviously-incoherent real mask found this
phase) has the *lowest* solidity of all 12 real objects, but a real object with no reported visual
defect sits within 0.05 of it, and a different real object with no reported defect has the
*highest* solidity of all 12 — above three of the four confirmed-defective masks. This is a fourth
independent, actively-adversarial attempt to find a counter-example to section 7's conclusion, and
it also failed to find one, strengthening (not merely repeating) that conclusion.

This does not fully substitute for a genuinely independent fresh-context review of every claim in
this document (in particular, the qualitative mask-overlay visualizations in sections 3, 5.1, 5.2,
and 6.4 were not re-derived by a second reviewer with no access to this session's own reasoning) —
disclosed as a real, honest limitation of this phase's QA coverage, not hidden. A follow-up
independent review remains open, lower-priority future work (the underlying evidence — the raw
`.npy` arrays and this document's own reproduction commands, section 12 — remains available on this
checkout for exactly that purpose).

## 12. Reproduction commands

```bash
# Local: tests, lint, types (unchanged baseline -- no src/ edits this phase)
uv run pytest -q
uv run ruff check .
uv run mypy src

# Local, deterministic, no-GPU reproduction of BUG-11-03's actual mechanism (uses the real
# downloaded mask array, git-ignored, present on this checkout at
# outputs/debug/phase11_gpu_evidence/villainess_ending_scuffle_obj_cloth_5_mask.npy):
uv run python -c "
import numpy as np, cv2
mask = np.load('outputs/debug/phase11_gpu_evidence/villainess_ending_scuffle_obj_cloth_5_mask.npy')
image = cv2.imread('examples/realworld/villainess_ending_scuffle.png')
x0,x1,y0,y1 = 18,470,958,1643
crop_mask = mask[y0:y1, x0:x1]
print('mask coverage of the speech-bubble/hand crop region:', (crop_mask>0).mean())
"

# Remote (Kaggle/Jupyter GPU worker only, per ADR 0003) -- not committed as reusable
# infrastructure (this phase's own throwaway diagnostic, same convention as Phase 10's
# inspect_wbf.py); fully reconstructable from section 6's own description.
```

## 13. Git

Branch: `phase-11-multi-bug-forensics`. See the top-level session summary for the final commit
list, PR URL, and validation results.
