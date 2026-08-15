# Phase 16 results: drawn-effect animation (RADIAL_EXPAND + effect-aware heuristics)

## Scope and question

The phase brief's goal 3 is "animate already-drawn effects" (speed lines, motion strokes,
impact effects, sparks, smoke, water, energy effects, glow) and goal 2 is "natural motion,
not simple geometric displacement." This phase's tested hypothesis:

> Every effect label the VLM produces (`rain`, `green_fluid`, `speed_lines`,
> `impact_effect`, `energy_effect`, `smoke`) falls through `_MOTION_HEURISTICS` to the same
> rigid `_DEFAULT_MOTION` (uniform translate, amplitude 0.02) -- exactly the "simple
> geometric displacement" the phase brief rejects. Adding an effect-specific motion model
> (`RADIAL_EXPAND` for the radial class: impact/energy/glow) plus effect-aware heuristics
> and an analysis prompt that asks the VLM to list drawn effects as animation targets will
> produce natural effect motion end-to-end.

## What was done

- **`RADIAL_EXPAND` transform kind** (`schemas/animation_plan.py`, `animation/transforms.py`):
  a spatially-varying radial pulse about the object's own pivot. The center stays
  effectively fixed while the rim breathes outward (`value > 0`) / inward (`value < 0`);
  displacement grows with distance from the pivot (`(r/r_max)^1.5` falloff), unlike uniform
  `SCALE` which moves the whole footprint rigidly. `amplitude` = peak rim displacement as a
  fraction of the object bbox's longest side. Identity at rest (`value == 0`), so a seamless
  `cycle` loop returns to its start state structurally (integer-speed rule already enforced
  by the schema). Local-ROI + `cv2.remap` implementation, same static-region-preservation
  contract as the other kinds.
- **`TransformGeometryProfile`** for `radial_expand` (`validation/transform_geometry.py`):
  MESH_WARP-like loose area bound (35%) plus a modest edge margin (2%) so the rim has room
  to breathe, no aspect-ratio check (elongated bursts are legitimate).
- **Effect-aware `_MOTION_HEURISTICS` entries** (`analysis/plan_builder.py`):
  impact/burst/explosion/shockwave/energy/glow/pulse/aura/radiat/flash -> `radial_expand`;
  smoke/steam -> `mesh_warp`; water/splash/fluid/liquid/wave/spray -> `mesh_warp`;
  rain -> translate-down (speed 2); spark/particle/debris/shard -> opacity flicker (speed 2);
  speed lines/streaks/slash -> `mesh_warp`. Object heuristics (cloth/hair/sword/eye) are
  ordered first, so effect keywords do not shadow ordinary objects (verified by test and by
  the real GPU signal: `weapon` still -> rotate, `character_hair` still -> translate).
- **Analysis prompt** now instructs the VLM to treat already-drawn effects as first-class
  animation targets with effect-specific `motion_description`s, while keeping speech
  bubbles/dialogue text/panel borders static.
- **Tests** (all pass): radial center-fixed/rim-moves, negative-value contraction,
  identity-at-rest, static-region preservation, seamless-loop boundary, determinism,
  off-center pivot resolution, single-pixel finiteness, effect-heuristic mapping,
  geometry-profile presence, prompt content.

## GPU evidence (short runs, real 2xT4 Kaggle worker, branch `phase-16-drawn-effect-animation` @ `5441e78`)

### Run 1: analysis-only signal (cheapest possible test of the hypothesis)

Pages: `eval_weapon_effects.png`, `wind_breaker_sprint.png`. Loads Qwen once, runs real
panel-aware analysis, prints every ObjectPlan's transform kind. No grounding/SAM/animation.

Result (the core hypothesis confirmation):

- `eval_weapon_effects`: `impact_burst` -> `radial_expand` (secondary), `speed_lines` ->
  rotate (secondary; its description contained a weapon mention, and the weapon entry is
  ordered before the effect entries -- the heuristic chose the object transform), `weapon`
  -> rotate (primary).
- `wind_breaker_sprint`: `speed_lines` -> **primary** `mesh_warp` (confidence 1.0),
  `impact_burst` -> `radial_expand` (secondary, twice), `rain` -> micro `mesh_warp`.
  Ordinary objects (`character_clothing` -> mesh_warp, `character_hair` -> translate,
  `bicycle` -> translate) unchanged by the effect track.

### Run 2: full end-to-end pipeline on `wind_breaker_sprint`

`run_phase16_gpu_effects.py --pages examples/realworld/wind_breaker_sprint.png`.

Panel statuses: `[PASS, REJECTED, REJECTED, REJECTED]`.

Panel_001 (PASS) object path:
- `obj_speed_lines_3` (PRIMARY, mesh_warp): grounding found it; semantic target
  validation ACCEPT ("dynamic lines typical of speed lines"); geometry ACCEPT for
  mesh_warp; SAM mask accepted; mask-semantics **ACCEPT** ("The bright region shows only
  speed lines without any other content", confidence 1.0); rendered.
- `obj_character_clothing_0` (secondary, mesh_warp): ACCEPT through validation and
  mask-semantics; rendered.
- `obj_character_hair_1` (secondary): all grounding candidates REJECTed semantically
  (crops showed clothing/background, not hair); dropped -- PRIMARY unaffected.
- `obj_impact_burst_2` (secondary, radial_expand): one candidate semantic ACCEPT but
  geometry REJECT (bbox within 0.5% of reference edge, under the 2% radial_expand margin);
  another candidate geometry REJECT (bbox 86.7% of reference region, over the 35% bound).
  All candidates dropped; PRIMARY unaffected. **Correct fail-closed behavior**: the burst
  touching the panel edge was not rendered rather than clipped.
- `obj_rider_2` (secondary, rotate): geometry REJECT (edge-touching / 17.1% area);
  dropped.

Numerical verification of the rendered PASS video (decoded source frames, panel_001):
seamless loop verified on the source frame sequence (wrap step 2.26 <= 2x ordinary step
2.19); 93.8% of pixels static across sampled frames; per-frame mean change over moving
pixels ~6.1. (The decoded H.264 stream shows the expected bounded codec noise, max diff
255 on mask-edge pixels, consistent with the project's known codec-noise contract.)

### Run 3: full end-to-end pipeline on `omniscient_reader_blade`

`run_phase16_gpu_effects.py --pages examples/realworld/omniscient_reader_blade.png`.

Panel status: `[PASS]`.

- `obj_raised_sword_4` (PRIMARY, rotate): grounding candidate rank 0 semantic ACCEPT but
  geometry REJECT (bbox flush to panel edge, 0% margin); rank 1 ACCEPT both; SAM mask
  accepted; mask-semantics **ACCEPT** ("The bright region shows only the raised sword
  without any additional content", confidence 1.0); rendered.
- `obj_speed_lines_3` (secondary, mesh_warp): ALL grounding candidates semantically REJECT
  because the grounded crops contained dialogue text and clothing, not speed lines. This is
  the artwork-preservation goal working on a real example: the effect was not grounded, so
  the dialogue text it pointed at was NOT animated. Dropped; PRIMARY unaffected.
- `obj_character_hair_0` (secondary, translate): geometry REJECT on a 63.1%-area candidate;
  a second candidate passed geometry but SAM produced an asymmetric edge-hugging mask
  (right edge 45.3% vs left 2.7%) caught by the Phase 8.3 edge-asymmetry gate. Dropped.
- `obj_character_clothing_1` (secondary, mesh_warp): geometry REJECT (63.4% area > 35%
  bound). Dropped.

Numerical verification of the rendered PASS video: seamless loop verified on the source
frame sequence (wrap step 0.27 <= 2x ordinary step 0.24); 94.5% of pixels static across
sampled frames; motion localized to the sword region.

### Run 4: full end-to-end pipeline on `angels_of_war_fleet`

`run_phase16_gpu_effects.py --pages examples/realworld/angels_of_war_fleet.png`.

Panel status: `[REJECTED]`.

- `obj_space_ship_impact_burst_6` (PRIMARY, radial_expand): one candidate semantic ACCEPT
  but geometry REJECT (bbox 41.0% of reference region > 35% radial_expand bound); the other
  candidate REJECTed pre-VLM (bbox 98.3% of the image, over the 90% generic bound).
  All candidates failed; the panel correctly fail-closed rather than animating a
  nearly-whole-panel burst.

### Run 5: full end-to-end pipeline on `space_monster_hypersenses`

`run_phase16_gpu_effects.py --pages examples/realworld/space_monster_hypersenses.png`.

Panel status: `[STATIC]` -- a valid, informative result: the VLM proposed no animated
object/effect on this page, so no video was produced and every model stage still released
correctly. No effect track was exercised.

### Run 6: RADIAL_EXPAND end-to-end on `angels_of_war_fleet` (after bound relaxation)

Real finding that blocked RADIAL_EXPAND: an effect's grounding bbox is a poor proxy for its
moved-pixel footprint. The effect-mask diagnostic (below) measured real speed_lines/impact_burst
boxes up to 98% of their panel producing SPARSE masks covering only 6-17% of it (density
0.28-0.50), because a burst is radiating lines, not a filled region. Based on that evidence:

- `transform_geometry.py`: radial_expand area bound 35% -> 60%, edge margin 2% -> 0 (an
  effect covering its panel legitimately has an edge-touching bbox while its sparse mask sits
  at the center).
- `segmentation/segment.py`: new opt-in `max_mask_density` (bound 0.70, documented from real
  defective masks at 0.84-0.90) rejects dense "select everything" masks post-segmentation;
  the orchestrator passes it for RADIAL_EXPAND objects only.

Re-run of `angels_of_war_fleet`: panel status `[PASS]`.

- `obj_space_ship_impact_burst_6` (chosen animated object, radial_expand): semantic ACCEPT
  -> geometry **ACCEPT** -> mask accepted -> mask-semantics **ACCEPT** ("The bright region
  shows only the space ship impact burst", confidence 1.0) -> rendered. **First end-to-end
  RADIAL_EXPAND render of a drawn impact burst.**
- `obj_space_ship_speed_lines_5` (secondary, mesh_warp): geometry REJECT (bbox 39.7% > 35%
  mesh_warp bound); dropped, PRIMARY unaffected.

Numerical verification of the rendered PASS video: seamless loop verified (wrap step 2.13 <=
2x ordinary step 2.00); 78.9% of pixels static across sampled frames (the other 21% is the
burst itself pulsing); motion present across the effect region.

### Effect-mask density diagnostic (evidence behind the bound relaxation)

`scripts/run_phase16_effect_mask_diagnostic.py` on `wind_breaker_sprint`,
`angels_of_war_fleet`, `omniscient_reader_blade` (real 2xT4 worker): for each effect label
(speed_lines, impact_burst, energy_effect, glow_effect), grounding + SAM measured bbox area
fraction, mask density, and mask area fraction per candidate. Key result: the TOP-ranked
candidates were sparse -- e.g. speed_lines on wind_breaker_sprint panel_01 cand0: bbox 17.4%
of panel, mask density 0.316, mask area 5.9%; impact_burst panel_03 cand0: density 0.661,
mask area 17.5%. Meanwhile dense "select everything" candidates (density 0.85-0.96, mask area
29-79%) were also returned by grounding and must be rejected -- which is exactly what the
post-segmentation `max_mask_density` check does.

### Run 0 (superseded observation): `eval_weapon_effects` full pipeline

`[REJECTED]`. The PRIMARY remained `weapon` (rotate), and its validated grounding
candidate was REJECTed by the transform-geometry gate (bbox 27.6% of reference region >
15% rotate bound) -- the same Phase 3.3 defect class (`eval_weapon_effects` weapon/panel
defect), still correctly fail-closed. This page did not exercise the effect track; the
analysis signal run above is the informative evidence for it.

## Classification and next steps

Classified **GOOD** for the drawn-effect track, based on seven real short runs:

- speed-lines PRIMARY rendered end-to-end (`wind_breaker_sprint`, seamless loop verified);
- **impact-burst PRIMARY rendered end-to-end with RADIAL_EXPAND** (`angels_of_war_fleet`
  after the evidence-based bound relaxation, seamless loop verified);
- a secondary speed-lines candidate whose grounding pointed at dialogue text was correctly
  fail-closed rather than animating the text (`omniscient_reader_blade`);
- ordinary objects still map to their pre-Phase-16 transform kinds on every page tried;
- real effect masks are sparse (diagnostic evidence), and the 35% bbox-area + 2% edge-margin
  bounds that fail-closed every effect were replaced, on evidence, with a 60% area + 0 margin
  pre-segmentation profile plus a post-segmentation mask-density gate (0.70).

Next steps (see `docs/current-status.md` Immediate Priorities):
1. Exercise RADIAL_EXPAND on a real impact/energy panel whose effect mask passes geometric
   validation (a burst not touching the panel edge and under 35% area).
2. Add more effect-heavy pages to the real evaluation set (impact/energy/glow/smoke/water
   render paths remain unexercised).
3. Consider effect-aware semantic labels for the mask-semantics gate prompt.

## Files changed

- `src/manga_animation/schemas/animation_plan.py` (TransformKind.RADIAL_EXPAND)
- `src/manga_animation/animation/transforms.py` (`_radial_expand_frame`)
- `src/manga_animation/validation/transform_geometry.py` (radial_expand profile)
- `src/manga_animation/analysis/plan_builder.py` (effect heuristics + prompt)
- `tests/test_animation.py`, `tests/test_analysis.py`, `tests/test_validation.py`
- `scripts/run_phase16_gpu_effects.py`, `scripts/run_phase16_analysis_signal.py`
- `docs/animation-plan-schema.md`, `docs/current-status.md`

Local gate: 608 tests pass, `ruff check .` clean, `mypy src` clean. The 8
`test_localized_transform_matches_full_page_reference` failures seen on the remote worker
also fail on `main` on that worker (OpenCV/numpy version skew on the pip-installed worker
environment) and are not caused by this phase.
