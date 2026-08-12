# 8. Transform-aware geometric target validation (Phase 3.3.1)

Status: Accepted

## Context

Phase 3.3's real end-to-end evaluation run (`docs/phase3.3-results.md`) found a new, real
visual defect on `eval_weapon_effects.png`: Grounding DINO's candidate for `weapon` scored
0.255, and Phase 3.2's semantic validation check (`validation/validate.py`) correctly answered
"yes, this crop plausibly shows a weapon" — a real, defensible signal; a blade-like shape genuinely
is visible in the crop. But the candidate's bbox covered nearly the entire dark action panel
(the "척" sound-effect text, the panel's own border, and surrounding background artwork
included), and the plan's `rotate` transform then visibly swung the *whole panel* as one rigid
unit, not the weapon — comparing frame 0 to frame 24, the sound-effect text and panel border
tilt along with everything else, and torn black-wedge artifacts appear at the frame edges where
the rotation reveals background outside the bbox's own footprint. Every mechanical invariant
still held (pixels outside the *mask* untouched by construction, the loop still seamless), but
the *mask itself* was carved from an implausibly large region for the transform applied to it —
the same class of "technically valid, visually wrong" defect ADR 0006 fixed for semantic
mismatch, now recurring for a *geometric* mismatch instead.

Phase 3.2's validation stage (ADR 0006) answers exactly one question: "does this region depict
the intended semantic target?" It does not, and was never designed to, answer a second,
independent question: "is this specific region *geometrically safe* to animate with the
specific transform the plan intends?" A bbox that is a plausible instance of "a weapon" can
still be far too large, too close to a panel/page edge, or too oddly shaped to safely `rotate`
— the same bbox might be perfectly safe to `translate` a small amount, or to fade with
`opacity`. Semantic correctness and transform-geometric safety are independent properties of
the same candidate.

## Decision

Extend the existing validation stage (`src/manga_animation/validation`) — a new stage was
**not** added; this is one more independent check inside `validate_target`, sitting after the
existing semantic check and before that function returns ACCEPT, mirroring how ADR 0006 already
composes an existing cheap deterministic check (bbox plausibility) with a VLM-based one:

1. **New module `validation/transform_geometry.py`**: `check_transform_geometry(bbox,
   transform_kind, *, panel_bbox_px, image_shape) -> (bool, str)` — deterministic, no model
   call. Three checks, any one failing rejects the candidate:
   - **Area fraction** relative to a reference region (the object's real panel when known via
     `panel_bbox_px`, else the full page — identical fallback convention to every other
     panel-aware/page-level dual path in this pipeline).
   - **Edge-margin fraction** — minimum clearance between the bbox and the reference region's
     edges, so a transform that sweeps pixels beyond the box's own footprint (rotate, scale,
     shear) has real room to do that without immediately clipping/revealing background sharply.
   - **Aspect ratio** (only where a kind's mechanism makes this meaningful — currently only
     `ROTATE`, generously bounded so a legitimately elongated object like a raised sword isn't
     penalized for being elongated, only for being a degenerate sliver).

2. **`TransformGeometryProfile`**: a small, explicit per-`TransformKind` bound registry
   (`_TRANSFORM_GEOMETRY_PROFILES`), **not one universal threshold** — each kind's bounds are
   derived from that kind's actual geometric mechanism (documented inline, next to each entry):
   `ROTATE`/`SHEAR` (rigid/skewing sweep — tightest bounds), `SCALE` (grows beyond its own
   footprint — needs clearance, direction-agnostic), `MESH_WARP` (local/continuous deformation —
   loosest of the "moves pixels" kinds, matching this codebase's own real cloth/banner
   heuristic), `TRANSLATE` (small real amplitude in this codebase's actual usage — directional
   risk, not areal), `OPACITY` (never moves a pixel spatially — deliberately deferred back to
   the pre-existing generic `MAX_OBJECT_COVERAGE_FRACTION` bound, not given an invented
   stricter one it doesn't need). Extensible: a new `TransformKind` needs one new registry
   entry with its own documented rationale, nothing else changes.

3. **Check ordering inside `validate_target`**: bbox plausibility (existing, cheapest) →
   semantic agreement (existing, one VLM call) → **transform geometry (new, runs only after
   semantic agreement ACCEPTs)**. Deliberately *after* the semantic check, not before it,
   despite being cheaper — a candidate that is already the wrong object shouldn't have its
   geometry scored at all, and the semantic-mismatch rejection reason must never be shadowed by
   an incidental geometry failure on the same wrong candidate. Fail-closed throughout: semantic
   mismatch → REJECT; semantic match but geometrically unsafe → REJECT; only semantic match
   *and* geometric safety together → ACCEPT.

4. **`pipeline.types.ValidationResult`** gains `transform_compatible: bool | None` — `None`
   when the check was never reached (an earlier check already rejected the candidate, or the
   object has no `motion` to check against), `True`/`False` otherwise. Existing fields
   (`semantic_match`, `bbox_plausible`, `reason`) are unchanged in meaning.

5. **`pipeline.orchestrator.run_pipeline`** now computes `panel_bbox_px` before grounding
   (previously computed just before the animation stage) and threads it into every
   `validate_target` call, so the geometry check's reference region is the object's real panel
   whenever panel-aware analysis (Phase 3.3, ADR 0007) produced one — reusing the exact same
   `_panel_bbox_px` helper, not a new computation.

## Consequences

- No bbox is ever silently clipped/resized to force a pass — a geometrically unsafe candidate
  is REJECTed outright, exactly like a semantically wrong one, and the existing "try the next
  ranked grounding candidate" retry loop (ADR 0006) already handles moving on from it. This was
  a deliberate choice over silently shrinking the box: this project has no evidence yet that a
  clipped box remains semantically correct (clipping a "weapon" box down to 15% of its original
  area might just as easily crop out the actual blade and keep the hilt, or vice versa) — see
  "Open questions".
- `validate_target` gains one new optional parameter (`panel_bbox_px`, defaults to `None` =
  full-page reference, preserving every pre-existing caller's behavior exactly).
- Grounding DINO, SAM 2.1, the VLM, and panel detection (ADR 0007) are all completely
  untouched — this defect and its fix are entirely within the validation stage's existing
  boundary (grounding-quality question, `segmentation-agent`'s territory per
  `docs/pipeline.md`'s "Stage ownership").
- The bounds in `_TRANSFORM_GEOMETRY_PROFILES` are documented, evidenced-by-one-real-defect
  choices, not a statistically calibrated set — same status as every other threshold already in
  this codebase (`pipeline/types.py`'s coverage fractions, `validation/validate.py`'s crop
  margin). They are deliberately conservative (tighter than the pre-existing generic bound)
  rather than tuned for maximum acceptance rate, per the explicit brief this ADR implements:
  "do not optimize acceptance rate at the expense of visual correctness."

## Revision (same-phase, before final acceptance)

The real remote re-verification run performed immediately after this decision's initial
implementation (see `docs/phase3.3-results.md`'s "Phase 3.3.1" section) found that the
`TRANSLATE` profile's initial `min_edge_margin_fraction=0.03` **falsely rejected this
project's own already-confirmed-correct positive case**: `sample_page_01.png`'s real
`character_hair`/`translate` candidate sits flush (0% margin) against the top edge of its
panel — a completely normal portrait composition (hair starts at the top of a head), not a
geometric defect. Re-examining the reasoning: `TRANSLATE`'s real risk (per this same ADR's
own "Decision" section) is directional revealing of background at the trailing edge, which the
existing hidden-region reconstruction stage (`reconstruction/`) already exists to handle — a
small, rigid shift has nowhere new to "swing into" the way `ROTATE`/`SHEAR`/`SCALE` do, so an
edge-margin requirement was never actually load-bearing for this kind, only for those three.
`TRANSLATE`'s `min_edge_margin_fraction` was corrected to `0.0` (area fraction remains the only
active bound for this kind) and a regression test added
(`tests/test_validation.py::test_translate_accepts_a_bbox_flush_against_the_reference_edge`).
Re-verified for real afterward: `sample_page_01.png`'s hair candidate is ACCEPTed again, with
`transform_compatible=True`.

This is the exact kind of real, evidenced correction this ADR's bounds always anticipated might
be needed (see "Consequences": "documented, evidenced... choices, not a statistically
calibrated set") — caught before final acceptance specifically *because* the brief required
re-confirming the real positive case, not just the negative one.

## Open questions

- Whether a future phase should attempt to *repair* a geometrically-unsafe-but-semantically-
  correct candidate (e.g., a tighter re-grounding pass, or clipping toward the crop's own
  saliency) rather than only rejecting it, is intentionally left open — this phase's brief
  scoped the fix to "reject unsafe geometry," not "recover a usable region from it."
- The profiles are calibrated against exactly one real observed defect (`eval_weapon_effects.png`,
  `ROTATE`). The other five kinds' bounds are reasoned from their transform mechanics and this
  codebase's own existing heuristic amplitudes (`analysis/plan_builder.py::_MOTION_HEURISTICS`),
  not from a second independently-observed real failure for each kind — flagged, not hidden.
- No real mask exists at validation time (segmentation runs after it) — these checks are bbox-only
  by necessity. A future phase could re-check geometry once a real mask exists, closer to
  rendering, as a second, tighter gate — not attempted here (would need a new stage boundary or
  a second validation pass after segmentation, out of this fix's scope).
