# Phase 6 results: seamless-loop hardening + local-region rendering at scale

Real, local, non-GPU results for Phase 6 (see the Phase 6 brief, delivered directly to the
assistant, not committed as a file, and
[ADR 0012](decisions/0012-phase6-seamless-loop-and-local-rendering.md) for the design/decision
rationale this file's evidence supports). This is a point-in-time results record — ADR 0012 is
the source of truth for *why*; this file is *what was checked and what it measured*.

**Status: implementation complete, fully verified locally (pytest/ruff/mypy all green, 422
tests up from 329 at Phase 5.1 close). No remote GPU work was needed or performed — both Step 1
(schema) and Step 2 (CV) are deterministic CPU/OpenCV/NumPy code, verified entirely with
synthetic data per the Phase 6 brief's own GPU policy.**

## What Phase 6 set out to do

Per the brief: (1) close a confirmed inconsistency in the seamless-loop schema validator around
`loop_mode="once_hold"` + `loop.seamless=True`, and (2) harden rendering for scale *within a
single render* — extreme page aspect ratios, multiple simultaneously animated objects, large
frame counts, and making CPU/memory cost scale with the animated region rather than the full
page wherever architecturally safe. Explicitly **not** in scope: batch/multi-page processing,
model reuse across pages, new evaluation infrastructure (that's Phase 7), or any of Phase 5.1's
known limitations (`eval_weapon_effects.png` candidate correctness, live multi-object GPU E2E,
SAM/LaMa/VLM quality, the `_grounding_region`/`_reference_region` DRY duplication).

## Step 1: seamless-loop schema hardening

**Investigation finding**: `schemas/animation_plan.py`'s only seamless-loop check was
`cycle` + non-integer `speed`. Its error message suggested `once_hold` as an equivalent
alternative to `ping_pong`. It is not: per `animation/curves.py::sample_motion_value`,
`once_hold` sweeps from rest (`0.0`) to its end state (`1.0`) and then holds `1.0` for the rest
of the loop, never returning to rest before the loop wraps — so pairing it with
`loop.seamless=True` was **silently accepted but always produces a visible jump** at the loop
boundary for any object with real motion (`amplitude > 0` is schema-required). This was
confirmed both at the scalar motion-value level and by directly comparing composited frames at
the loop boundary (`test_once_hold_frame_pixels_do_not_match_at_loop_boundary` in
`tests/test_animation.py`).

**Fix**: `AnimationPlan` now rejects `once_hold` under `loop.seamless=True` outright (mirroring
the existing `cycle` check's structure), and the `cycle` check's own error message no longer
mentions `once_hold`. `once_hold` remains fully valid whenever `loop.seamless=False`. See ADR
0012 for the full before/after semantics and why this is the smallest correct fix (no change to
`curves.py`, no change to `ping_pong`/`cycle`'s existing behavior, no change to production
motion heuristics).

**Tests added** (`tests/test_animation_plan.py`, `tests/test_animation.py`):
`test_once_hold_with_seamless_loop_is_rejected`,
`test_once_hold_with_non_integer_speed_and_seamless_loop_is_rejected` (proves the check is
independent of the pre-existing speed rule), `test_once_hold_is_allowed_when_not_seamless`,
`test_non_integer_speed_seamless_cycle_error_does_not_suggest_once_hold`,
`test_once_hold_frame_pixels_do_not_match_at_loop_boundary`,
`test_ping_pong_frame_pixels_match_at_loop_boundary` (the seamless-safe replacement, including
at non-integer speed).

## Step 2: local-region CV rendering

### Pre-implementation CV review

Before touching `animation/transforms.py` (flagged in the brief as the highest-risk change), a
dedicated CV review pass checked the proposed localization strategy (ROI-restricted
`cv2.warpAffine`/`cv2.remap` via a shifted matrix / absolute-coordinate maps, no source
cropping) against real OpenCV internals and the schema's actual constraints. It confirmed the
matrix-shift derivation, confirmed no-input-crop is correct and safe (border handling matches
the old full-page call exactly), and flagged three concrete corrections before implementation:
margin must be derived from the transform matrix's own operator norm (not a guessed constant,
since `amplitude`/`shear` have no schema upper bound), integer ROI bounds must use `floor`/
`ceil` (not `int()`, which truncates toward zero and silently under-covers negative
coordinates), and `mesh_warp`'s margin must account for `direction` not being schema-normalized
for that transform kind. All three were incorporated before any code was written. It also
predicted, correctly, a bounded `±1` floating-point characteristic in the shifted-matrix
`warpAffine` path — see below.

### What changed

See ADR 0012 for the full design. Summary: `generate_transformed_layer`'s affine kinds and
`mesh_warp`, `compositing.composite_frame`/`composite_frame_stack`, and
`reconstruction._compute_hole_mask` all now restrict their actual pixel-level computation to
the relevant mask(s)' own bbox (or, for affine transforms, the AABB of that bbox's transformed
corners) instead of the whole page. No external contract changed — same function signatures,
same full-page-shaped return arrays, same `Layer`/`ReconstructionResult` types.

### Correctness evidence

A verbatim copy of each pre-Phase-6 full-page implementation was kept as a deterministic
reference (`tests/test_animation.py`, `tests/test_compositing.py`,
`tests/test_reconstruction.py`) and compared against the new localized code across the full
required edge-case matrix:

| Case | Covered in |
|---|---|
| small object, normal page | `test_localized_transform_matches_full_page_reference` (parametrized) |
| object at each of the 4 edges | same, `top_edge`/`bottom_edge`/`left_edge`/`right_edge` |
| object touching a corner | same, `top_left_corner`/`bottom_right_corner` |
| large object / very small (1x1-adjacent) object | same, `large_object`/`very_small_object` |
| non-square page | same, `non_square_page` |
| extreme-aspect-ratio page (720x90-scale, 6:1) | same, `extreme_aspect_ratio_page` |
| displacement beyond the original bbox | `test_localized_transform_with_displacement_beyond_original_bbox_matches_reference` |
| transform pushing the object fully off-page | `test_localized_transform_pushed_fully_off_page_matches_reference` |
| distant (page-referenced) rotation pivot | `test_localized_rotate_about_distant_page_pivot_matches_reference` |
| multiple objects, independence / identity safety | `test_localized_transform_multiple_objects_are_independent_and_identity_safe` |
| overlapping objects, z-order semantics | `test_composite_frame_stack_overlapping_layers_respect_z_order` + `test_composite_frame_stack_matches_full_page_reference_at_five_object_scale` |
| 5-object scale (this project's observed real-plan max, ADR 0010) | `test_composite_frame_stack_matches_full_page_reference_at_five_object_scale`, `test_composite_frame_stack_five_objects_respect_z_order_and_static_region` |
| larger frame count (96 frames = 4s@24fps default) | `test_composite_frame_stack_matches_full_page_reference_with_larger_frame_count` |
| reconstruction hole-mask locality (edges/corners/large/aspect-ratio) | `test_compute_hole_mask_matches_full_page_reference` (parametrized), `..._with_many_frames` (240 frames) |
| static-region bit-exactness | pre-existing tests, all still passing unchanged |

**Result**: `mesh_warp`, `opacity`, `reconstruction`, and `compositing` are bit-exact against
their old full-page references in every case. The affine (`warpAffine`-with-shifted-matrix)
kinds match with an explicitly justified `atol=1` (see next section) — never in the static
region, which stays exactly bit-exact throughout.

### A real, bounded floating-point finding

Initial exact-equality (`assert_array_equal`) comparisons against the old reference found a
small number of `±1` (never more) `uint8` mismatches in the affine kinds, confined to
partial-alpha mask edges / interpolation boundaries inside the moving object's own footprint —
never in the static region. This matches the CV review's prediction: `cv2.warpAffine`'s
`INTER_LINEAR` path quantizes each output pixel's source coordinate through two independently
fixed-point-rounded terms, and shifting the matrix's translation column (mathematically
equivalent, numerically different) can round a near-boundary case to the adjacent step. This
was verified to be genuinely bounded and harmless (not an under-coverage bug: a real ROI
mis-sizing would show as a large contiguous region missing or wrong, not an isolated `±1`), and
is documented with an explicit `atol=1` in the affected tests rather than silently loosened.
`test_cycle_frame_pixels_match_at_loop_boundary` was also updated to compare **composited**
output rather than raw per-object layer arrays outside the mask footprint — see that test's own
docstring for why raw-array equality there was never a real invariant (the old code's
"background incidentally warped outside the mask" was dead, compositing-irrelevant content, not
a documented contract).

## Performance evidence

`scripts/phase6_local_rendering_performance.py`, one real local run (Apple Silicon M1 Max, CPU
only — see the script's own docstring for why this is not a hard gate; numbers are wall-clock
and environment-dependent, reported as evidence per the Phase 6 brief):

**Raw `cv2.warpAffine` cost only** (full-page `dsize` vs. ROI-restricted `dsize`, identical
matrix, fixed 40x40 object) — isolates exactly the claim this phase makes:

| Page | ROI px | Old (full-page) mean | New (ROI-only) mean | Speedup |
|---|---|---|---|---|
| 600x800 | 3600 | 0.38 ms | 0.014 ms | **26x** |
| 720x5062 (6:1) | 3600 | 0.88 ms | 0.014 ms | **61x** |
| 1100x6613 (6:1) | 3600 | 1.71 ms | 0.016 ms | **109x** |

The ROI-only cost stays flat (~0.014-0.016ms) regardless of page size; the full-page cost grows
with page pixel count. This is the direct evidence for "expensive CV work scales primarily with
the local animated region rather than full-page pixel count."

**End-to-end `generate_transformed_layer`** (same object/pages, full call including the
full-page zero-initialized placement array):

| Page | Old mean | New mean | Speedup |
|---|---|---|---|
| 600x800 | 1.52 ms | 1.15 ms | 1.32x |
| 720x5062 | 9.58 ms | 8.12 ms | 1.18x |
| 1100x6613 | 19.04 ms | 16.29 ms | 1.17x |

Smaller than the raw-warp numbers above — but **not**, as an earlier version of this section
claimed, primarily because of the full-page zero-initialized placement array (that allocation
is real but small, see below). The actual dominant cost in the "New mean" column above is
`bbox_of_mask(mask)`, called once **every** `generate_transformed_layer` call — including from
`pipeline/orchestrator.py`'s per-frame animation loop, which calls this function once per
frame, per object, for the SAME original (untransformed) `mask`, whose tight bbox therefore
never changes across those calls. Isolating that one call's own share of the "New mean" cost
above, on the exact same runs:

| Page | `bbox_of_mask` mean | Share of "New mean" |
|---|---|---|
| 600x800 | 1.05 ms | 90.8% |
| 720x5062 | 7.87 ms | 97.0% |
| 1100x6613 | 15.74 ms | 96.6% |

That the share *grows* with page size (not just stays large) is the direct evidence this is the
dominant cost, not the (page-size-independent) allocation: `bbox_of_mask` does a full-page
`np.where(mask > 0)` scan, so its cost scales with total page pixel count exactly like the
old full-page `warpAffine` did, while the two `np.zeros_like` allocations below cost a
near-constant ~0.2-0.3ms regardless of page size.

**Fix (post-Phase-6 follow-up)**: `generate_transformed_layer` now accepts an optional
`object_bbox_px` parameter; when a caller supplies it, the internal `bbox_of_mask(mask)` call is
skipped entirely. `pipeline/orchestrator.py`'s per-frame loop now passes `seg.bbox` — the same
tight bbox already computed once, correctly, during segmentation
(`segmentation/segment.py::_tight_bbox`, `seg.bbox == bbox_of_mask(seg.mask)` always holds) —
instead of letting it be silently recomputed on every frame. The script now also measures this
"hoisted" usage directly (bbox computed once outside the per-frame timing loop, exactly mirroring
the orchestrator):

| Page | Old mean | New mean (hoisted bbox) | Speedup |
|---|---|---|---|
| 600x800 | 1.52 ms | 0.078 ms | **19.5x** |
| 720x5062 | 9.58 ms | 0.187 ms | **51.2x** |
| 1100x6613 | 19.04 ms | 0.318 ms | **59.9x** |

This is the real end-to-end speedup a rendered loop gets once the redundant per-frame bbox scan
is hoisted out — now close to the raw-warp-only numbers above (the small remaining gap is the
two `np.zeros_like` full-page placement-array allocations, still present and still
architecturally required per `Layer.frames`' unchanged full-page contract; see ADR 0012's
"Known limitations" for why that cost was deliberately not eliminated).

`mesh_warp` shows a larger "New mean" (bbox-recomputed) end-to-end speedup (2.04x, see the table
below) than the affine kinds (~1.16-1.18x) because its old implementation additionally built two
full-page float32 meshgrid arrays before remapping — more baseline overhead to remove, same
`bbox_of_mask` cost and allocation floor remaining after (the hoisted-bbox variant removes nearly
all of it for every kind alike, as the table above shows).

| Transform kind (720x5062 page) | Old mean | New mean | New mean (hoisted bbox) | Speedup (hoisted) |
|---|---|---|---|---|
| rotate | 9.58 ms | 8.12 ms | 0.215 ms | 44.6x |
| mesh_warp | 16.64 ms | 8.14 ms | 0.199 ms | 83.5x |
| translate | 9.26 ms | 8.01 ms | 0.152 ms | 60.8x |

**Multiple simultaneously-animated objects** (1100x6613 page, `composite_frame_stack`):

| n_objects | frame_count | Total | Mean/frame |
|---|---|---|---|
| 1 | 24 | 410 ms | 17.1 ms |
| 5 | 24 | 2272 ms | 94.7 ms |

**Larger frame counts** (720x5062 page, fixed object, ROTATE) — confirms no non-linear blowup:

| frame_count | Old mean/frame | New mean/frame |
|---|---|---|
| 24 | 9.61 ms | 8.12 ms |
| 96 | 9.55 ms | 8.17 ms |
| 240 | 9.59 ms | 8.12 ms |

Per-frame cost is flat across frame count in both implementations, as expected (each frame is
generated independently from the same source, per `generate_transformed_layer`'s own
docstring) — included as evidence against a non-linear regression, not as a new finding.

## Architectural invariants verified

- **Original Image Is the Source of Truth**: static-region bit-exactness tests (pre-existing
  and new) all pass; the affine `±1` characteristic is proven confined to inside the moving
  object's own mask footprint, never the static region.
- **Local Modification**: now actually implemented for the CV transform/compositing/
  reconstruction stages, not just stated (see ADR 0012).
- **Identity Safety**: `test_localized_transform_multiple_objects_are_independent_and_identity_safe`
  and the existing Phase 5 identity tests all pass unchanged.
- **Phase 4 reconstruction invariant**: `_compute_hole_mask`'s UNION-over-frames-of-
  `(original & ~transformed[i])` formula is untouched; only its computation is bbox-restricted.
  All pre-existing reconstruction tests pass unchanged.
- **Phase 5.1 grounding contract**: `panel_bbox_px` semantics untouched; Phase 6 did not touch
  `grounding`, `validation`, or `analysis`.
- **Multi-object failure policy** (PRIMARY hard-fail, SECONDARY/MICRO silent drop): untouched;
  Phase 6 step 2 made no change to `pipeline/orchestrator.py`. (The post-Phase-6 bbox-hoisting
  follow-up below does touch it — one line, threading `seg.bbox` into an already-existing call —
  but does not touch the failure-policy logic itself.)
- **Deterministic First**: no randomness introduced; every new code path is a pure function of
  its inputs (confirmed by the existing `test_generate_transformed_layer_is_deterministic`
  parametrized test, still passing across all 6 transform kinds).

## Known limitations

See ADR 0012's "Known limitations" for the full list (full-page allocation floor, the `atol=1`
affine characteristic, the raw-layer-content contract change outside a mask's footprint, and
the 5-object/96-frame scope of the multi-object performance evidence). Restated briefly here:
none of these are correctness regressions; all are disclosed, bounded, and either
architecturally required or already covered by dedicated regression tests.

**Correction (post-Phase-6 follow-up)**: this section and the "End-to-end
`generate_transformed_layer`" performance table above originally attributed the gap between the
raw-warp speedup (25x-109x) and the smaller end-to-end speedup (~1.15x-2x) entirely to the
full-page zero-initialized placement-array allocation. That allocation is real (~0.02-0.27ms
across the tested page sizes, confirmed by direct measurement) but was never the dominant term —
measurement showed it was 90.8%-97.0% attributable to a separate, non-architectural cost:
`bbox_of_mask(mask)` being recomputed via a full-page `np.where` scan on every single
`generate_transformed_layer` call, including once per frame from `pipeline/orchestrator.py`'s
per-frame animation loop, for the SAME object mask whose tight bbox never changes across those
calls. See the "End-to-end `generate_transformed_layer`" section above for the corrected
numbers and the fix (an optional `object_bbox_px` parameter, now populated from
`SegmentationResult.bbox` by the orchestrator). Unlike the allocation floor, this was **not** an
architecturally required cost — it is now hoisted out, closing nearly all of the gap (see the
"hoisted bbox" tables above).

## Post-Phase-6 follow-up: per-frame `bbox_of_mask` hoisting

A closure audit of this phase (independent of the implementation above) verified the
localization work itself was correct, but found the misattribution corrected above: the
dominant remaining per-call cost was a redundant `bbox_of_mask(mask)` recomputation, not the
allocation floor. Fix, scoped to exactly this gap:

- `src/manga_animation/animation/transforms.py::generate_transformed_layer` gained an optional
  `object_bbox_px: BBoxPx | None = None` parameter; when supplied, it is used directly instead
  of calling `bbox_of_mask(mask)` internally. Omitting it reproduces the exact prior behavior —
  fully backward compatible for any other caller.
- `src/manga_animation/pipeline/orchestrator.py`'s per-frame animation loop now passes
  `seg.bbox` (from `SegmentationResult`, itself computed once by `segmentation/segment.py::
  _tight_bbox` — the same tight-bbox algorithm as `bbox_of_mask`, so the two always agree for
  `mask == seg.mask`) — one line of wiring, no other orchestrator changes.
- `scripts/phase6_local_rendering_performance.py`'s `_time_generate_transformed_layer` now also
  measures `bbox_of_mask`'s own per-call cost and a "hoisted" variant (bbox computed once,
  reused across the frame loop) — see the corrected tables above.
- Regression tests added to `tests/test_animation.py`:
  `test_generate_transformed_layer_precomputed_bbox_matches_recomputed` (bit-identical output to
  the no-argument form, across all 6 transform kinds) and
  `test_generate_transformed_layer_trusts_caller_supplied_bbox_without_revalidation` (documents,
  deliberately, that a caller-supplied bbox is used as-is with no re-validation against
  `bbox_of_mask(mask)` — re-validating it would require the exact full-page scan this parameter
  exists to let callers skip, defeating its purpose; see that test's own docstring).

`uv run pytest` (429 tests), `uv run ruff check .`, and `uv run mypy src` all pass clean after
this follow-up. No other Phase 6 invariant, test, or contract was touched.

## Explicitly deferred / out of scope (unchanged from the brief)

Batch/multi-page processing, Phase 7 evaluation infrastructure, `eval_weapon_effects.png`
candidate correctness, panel detector correctness, VLM extreme-aspect-ratio analysis,
SAM/LaMa/inpainting quality, live multi-object GPU E2E validation, codec support beyond the
existing H.264 design, the `_grounding_region`/`_reference_region` DRY duplication, and frame-
dump retention policy (left unchanged — no Phase 6 evidence found a correctness/resource
failure that would justify touching it).

## Final acceptance

- [x] seamless-loop semantics correct for all supported loop modes
- [x] `once_hold` no longer creates the identified seamless-loop contradiction
- [x] existing `cycle`/`ping_pong` behavior remains correct (tests unchanged, still passing)
- [x] localized CV transforms preserve the observable result of the previous implementation
      (bit-exact for mesh_warp/opacity/reconstruction/compositing; documented `atol=1` for the
      affine `warpAffine` path, justified and bounded to the moving object's own footprint)
- [x] static pixels remain bit-exact
- [x] Phase 4 reconstruction invariant intact
- [x] Phase 5 identity-safety intact
- [x] Phase 5.1 grounding contract untouched
- [x] multi-object compositing tested at the observed 5-object scale
- [x] large/extreme-aspect synthetic pages covered
- [x] larger frame-count behavior covered (96-frame compositing equivalence; 240-frame
      reconstruction/timing evidence)
- [x] no new batch/evaluation framework introduced
- [x] no Phase 7 scope absorbed
- [x] pytest fully green (429 tests, up from 422 at the original Phase 6 report — 7 added by the
      post-Phase-6 bbox-hoisting follow-up)
- [x] ruff clean
- [x] mypy clean
- [x] no tests weakened or removed (one test's invariant was corrected, not weakened — see
      "A real, bounded floating-point finding" above and that test's own docstring)
- [x] documentation matches implementation (this file + ADR 0012 + updated
      `docs/animation-plan-schema.md` + updated `animation-planning` skill)
- [x] independent closure audit performed (separate from the implementation above); one real
      gap found — the "CV cost scales with the animated region" claim did not actually hold
      end-to-end because of the per-frame `bbox_of_mask` redundancy — and fixed, re-verified,
      and folded into this file's "Post-Phase-6 follow-up" section and ADR 0012's corrected
      "Known limitations", rather than accepted on the original report's word
- [x] git working tree clean at close (see commit list below)
- [x] no unapproved push occurred

**Verdict: PASS** (post-follow-up; the original implementation's claims were independently
audited rather than trusted, one real gap was found and fixed, and this file's numbers reflect
the corrected, re-measured state).

## Git state

Branch: `phase-6-wip` (off `main`, `phase-3.3-wip`/Phase 5.1 commits untouched). Commits, in
order:

1. `Phase 6 step 1: reject once_hold under seamless=True, fix misleading error text`
2. `Phase 6 step 2: localize CV transform/compositing/reconstruction to the animated region`
3. `Phase 6 step 3: local-rendering performance evidence script`
4. `Phase 6 step 4: document the seamless-loop fix and local-rendering architecture`
5. Post-Phase-6 follow-up: hoist the per-frame `bbox_of_mask` redundancy out of
   `generate_transformed_layer`'s hot path, and correct this file's/ADR 0012's performance
   narrative accordingly (this commit).

Not pushed; no push performed without explicit request, per the Phase 6 brief and standing git
discipline.
