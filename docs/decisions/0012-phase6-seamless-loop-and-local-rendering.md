# 12. Phase 6: seamless-loop schema correction and local-region CV rendering

Status: Accepted

## Context

Phase 6 was scoped as two things: (1) close a confirmed gap in the seamless-loop schema
validator, and (2) harden the existing rendering/CV pipeline for scale *within a single
render* (extreme page aspect ratios, multiple simultaneously animated objects, larger frame
counts) — explicitly not batch/multi-page processing, which stays Phase 7+ scope.

### Seamless-loop gap

`schemas/animation_plan.py`'s `AnimationPlan` validator only ever checked one seamless-loop
condition: a `loop_mode="cycle"` object under `loop.seamless=True` must have integer `speed`,
because a periodic `sin` curve only returns exactly to its start value after a whole number of
cycles. Its error message, written when only `cycle` was being validated, suggested three ways
out: "use an integer speed, switch loop_mode to 'once_hold'/'ping_pong', or set
loop.seamless=False". That message is wrong for `once_hold`. Per
`animation/curves.py::sample_motion_value`, `once_hold` sweeps from rest (`0.0`) to its end
state (`1.0`) over its active window and then **holds `1.0`** for the rest of the loop; nothing
ever resets it back to `0.0` before the loop wraps. Every fresh loop iteration restarts an
object at rest (`t < timing.delay_s` or `u = 0` both yield `0.0`), so a `once_hold` object's
held end state at the loop boundary structurally differs from its own rest state at frame 0,
for any object that actually has motion (`amplitude` is schema-required `> 0`). The schema
allowed `once_hold` + `loop.seamless=True` to construct successfully, so this was a silent,
always-triggered discontinuity at the loop boundary, not merely a possible one under specific
parameter choices (unlike the `cycle`/non-integer-speed case, which is parameter-dependent).
`ping_pong`, by contrast, is a genuine seamless-safe replacement: its triangular envelope
returns to `0.0` once its window closes, at any `speed` (its formula never uses `speed` at
all).

### Local rendering

`animation/transforms.py::generate_transformed_layer` is called once per rendered frame per
animated object, and — before this phase — always ran `cv2.warpAffine`/`cv2.remap` over the
**entire page**, even though the actually-animated object typically occupies a small fraction
of it (see the animation-planning skill's "minimal motion" framing: the system deliberately
finds one flag, one lock of hair, one weapon — not "the page"). `docs/architecture.md`'s "Local
Modification" principle ("every stage should touch the smallest region of the frame necessary
... warps should be local to the object being animated") had never actually been implemented
for this stage; Phase 3's `scripts/phase2_cv_feasibility.py` prototype used the same full-page
approach, and it carried straight through to production. `compositing/__init__.py` and
`reconstruction/__init__.py::_compute_hole_mask` had the same shape of gap: full-page-sized
boolean/blend operations regardless of how small the relevant mask actually was.

## Decision

### Seamless-loop: reject `once_hold` under `loop.seamless=True`

`AnimationPlan`'s validator now raises for any object whose `timing.loop_mode == "once_hold"`
while `loop.seamless` is `True` — the same treatment as the pre-existing `cycle`/non-integer-
speed check, added as a second, independent condition (not folded into the first; a `once_hold`
object always violates continuity regardless of `speed`). The `cycle` check's own error message
was corrected to only mention `ping_pong`/integer-speed/`loop.seamless=False` as ways out —
`once_hold` is no longer presented as an equivalent fix, because it never was one.
`once_hold` remains fully valid — and is the *documented, correct* choice for a genuinely
one-shot, non-repeating motion — whenever `loop.seamless=False`.

This is the smallest correct fix: no change to `curves.py`'s motion-value semantics (the
`once_hold` sweep-then-hold behavior itself is correct and used deliberately for non-seamless
loops), no change to production motion heuristics, and no change to `ping_pong`'s or `cycle`'s
existing validated behavior.

### Local rendering: ROI-restricted CV, unchanged external contracts

The target shape, common to all three modules touched:

```text
object bbox (+ transform's own reach, where the transform can move pixels)
    -> minimal local ROI, computed fresh per call from the actual matrix/motion parameters
    -> the local CV operation, restricted to that ROI
    -> the small result placed into a full-page-shaped array/output
    -> existing compositing/reconstruction/Layer contracts, entirely unchanged
```

No type in `pipeline/types.py` changed. `Layer.frames` still stores full-page `(ImageArray,
MaskArray)` pairs; `generate_transformed_layer`'s signature and return type are identical to
before. This was a deliberate boundary, not an oversight — Phase 6's own brief explicitly ends
the target architecture at "place transformed result into page coordinates", handing off to
"existing compositing/reconstruction semantics" unmodified. A deeper change (e.g. `Layer`
storing cropped regions plus an offset) would ripple into `compositing`, `reconstruction`,
`video-agent`'s frame assembly, and the orchestrator — out of scope for a hardening phase whose
explicit invariant list requires those stages' existing contracts and tests to keep passing
unchanged.

**`animation/transforms.py`** (the highest-cost stage — real `cv2.warpAffine`/`cv2.remap`
interpolation, not simple array arithmetic):

- **Affine kinds** (`translate`/`rotate`/`scale`/`shear`): the destination ROI is the AABB of
  the object bbox's four corners transformed through that frame's actual matrix (affine maps
  preserve convex containment, so this is a valid bound on where the transformed mask can be
  nonzero), expanded by a margin derived from the matrix's own **operator norm** (its largest
  singular value — how far this specific matrix can stretch a distance) rather than a guessed
  constant, since `amplitude`/`shear` have no schema-enforced upper bound. `cv2.warpAffine` is
  then called with the **full, uncropped** source array but a `dsize` restricted to the ROI and
  the matrix's translation column shifted to the ROI's own origin (`matrix_roi[:, 2] -=
  (roi.x0, roi.y0)`) — `warpAffine`'s cost is driven by the output canvas it iterates, not the
  input array size, so this alone makes the interpolation cost scale with the ROI. Not cropping
  the input sidesteps an entire class of "is the source margin wide enough for the interpolation
  kernel" bugs, since the full original array is always available to sample from at its real
  absolute coordinates.
- **`mesh_warp`**: the per-pixel displacement formula is unchanged, evaluated only for the ROI
  window's absolute page coordinates (`np.meshgrid(np.arange(roi.x0, roi.x1), ...)` instead of
  the whole page) and passed to `cv2.remap` against the full, uncropped image — since `remap`'s
  maps specify absolute sample coordinates directly, this needs no matrix-shift trick and is
  bit-exact by construction (same formula, evaluated at a subset of the same positions). ROI
  margin is `ceil(|strength| * max(|dir_x|, |dir_y|))` — not `ceil(|strength|)` alone, because
  `direction` is not schema-normalized for `mesh_warp` (only `translate`/`shear` require a unit
  vector).
- **`opacity`**: never moves pixels; only the mask-scaling arithmetic is restricted to the
  mask's own bbox (it is `0` everywhere else already, so nothing outside that bbox can differ).
- `bbox_of_mask` is computed exactly once per `generate_transformed_layer` call and threaded
  into `_mesh_warp_frame`/`_opacity_frame` as a parameter, removing a redundant second
  full-page `np.where` scan the old code performed for `mesh_warp` (and computed, unused, for
  `opacity`).

**`compositing/__init__.py`**: `composite_frame`'s hole-substitution and alpha blend, and
`composite_frame_stack`'s equivalents, are restricted to their relevant masks' own bboxes.
`composite_frame_stack` keeps its *exact* original numerical order of operations — every
active layer's contribution is accumulated in `float32` and rounded to `uint8` only once, after
the topmost layer, never re-rounded between layers — only the accumulator's *extent* shrinks
(to the union of every active layer's own bbox this frame, not the whole page). This distinction
matters: rounding to `uint8` between layers would change the observable result for two
overlapping partial-alpha layers, since the second layer would then blend against an
already-quantized value instead of the first layer's full-precision contribution — a real
regression the implementation was checked against directly (see `test_compositing.py`'s
overlapping-partial-alpha-layers cases).

**`reconstruction/__init__.py`**: `_compute_hole_mask`'s per-frame OR-accumulation loop is
restricted to `original_mask`'s own bbox — outside it, `original_mask > 0` is `False`
everywhere by construction, so every iteration there was already a guaranteed no-op.

### A real, bounded floating-point characteristic (not a bug)

Comparing the ROI-restricted affine path against a kept-verbatim copy of the old full-page
implementation across every required edge case (`tests/test_animation.py`) found `mesh_warp`,
`opacity`, `reconstruction`, and `compositing` are bit-exact, but the affine
(`warpAffine`-with-shifted-matrix) path occasionally differs by exactly `±1` on a `uint8`
channel value, at a small fraction of pixels, always at partial-alpha mask edges or interior
interpolation boundaries — never in the static region, never a magnitude greater than 1. An
independent CV review (see the implementation-report evidence in `docs/phase6-results.md`)
traced the mechanism: `cv2.warpAffine`'s `INTER_LINEAR` path quantizes each output pixel's
inverse-mapped source coordinate via two independently-rounded fixed-point terms; shifting the
matrix's translation column changes how the real-valued formula splits into those two terms,
and `round(a) + round(b) != round(a+b)` in general — a genuine, understood floating-point
characteristic of computing the *same* transform through a *differently parameterized* (but
mathematically identical) matrix, not an under-coverage or off-by-one bug. The regression tests
use an explicitly justified `atol=1` for the affine path's layer/mask pixel comparisons (never
for the static region, which stays exactly bit-exact); `mesh_warp`/`opacity` keep exact
equality, since their absolute-coordinate formulas don't go through this quantization path at
all.

## Known limitations

- The full-page zero-initialized array `generate_transformed_layer` still allocates every frame
  (to place the ROI result into, matching `Layer.frames`' unchanged full-page contract) is an
  **architecturally required, not eliminated** cost — see `docs/phase6-results.md`'s
  performance evidence: the raw interpolation compute now scales cleanly with the ROI (25x-109x
  faster in isolation, growing with page size, for a fixed small object), but the end-to-end
  per-call speedup is smaller (~1.15x-2x) because it is bounded by that allocation floor.
  Eliminating it would require `Layer`/`compositing` to stop consuming full-page arrays — an
  architecture change explicitly out of this phase's scope.
- The `atol=1` floating-point tolerance for the affine path (see above) is scoped to the
  moving object's own footprint by construction (compositing never reads a layer's pixels where
  its own mask is `0`), but it does mean `generate_transformed_layer`'s returned layer array is
  not byte-identical to the pre-Phase-6 implementation for every possible input — only
  observably identical once composited. This is disclosed, not hidden.
- The old full-page implementation's `warped_layer` return value incidentally contained the
  whole page warped (including background, wherever the transform's inverse mapping happened to
  sample it) outside the object's own transformed mask footprint. The new implementation returns
  zero there instead. No production code was found to depend on that content (compositing only
  ever reads it where the corresponding mask is nonzero), and one test
  (`test_cycle_frame_pixels_match_at_loop_boundary`) that incidentally checked raw-array equality
  including that region was updated to compare the actual pipeline-visible guarantee (composited
  output) instead — see that test's docstring for the full reasoning.
- Multi-object compositing performance evidence uses fixed, non-overlapping-by-design synthetic
  regions at up to 5 objects (this project's own observed real-plan maximum, per ADR 0010's
  Phase 5 audit) and 96 frames (the schema's real default 4s/24fps loop); it is not a claim about
  arbitrary/larger object counts or frame counts, which remain untested.

## Open questions

None new. Everything explicitly listed as out of scope in the Phase 6 brief (batch/multi-page
processing, Phase 7 evaluation infrastructure, SAM/LaMa/VLM quality, panel-detector accuracy,
frame-dump retention policy) was left untouched, per that brief.

## Acceptance

- `uv run pytest` (422 tests, up from 329 at Phase 5.1 close), `uv run ruff check .`, and
  `uv run mypy src` all pass clean.
- No existing test was weakened, deleted, or given a looser tolerance without the explicit,
  documented reasoning above; one test's *invariant* was corrected (not weakened) to check the
  actual pipeline-visible guarantee instead of an incidental implementation detail (see "Known
  limitations").
- Phase 4's reconstruction hole-mask formula, Phase 5's identity-safety tests, and Phase 5.1's
  `panel_bbox_px` grounding contract are all unmodified and still pass.
