# 11. Panel-aware Grounding DINO cropping (Phase 5.1)

Status: Accepted

## Context

A live investigation (see the "Phase 5 grounding bottleneck investigation" report; no separate
results file was created for it, per that task's own "do not commit diagnostic artifacts"
instruction) established, with real GPU reproduction, that Grounding DINO's `"weapon."` failure
on `examples/phase3_action_page.png` (720x5062, an extreme ~7:1 aspect-ratio page) is **not**
weapon-specific: at the production `threshold=0.25`, the full page produced **0 candidates**
not only for `"weapon."` but also for `"hair."`, `"person."`, `"character."`, and `"face."` —
every tested category collapsed equally. The same investigation found that `analysis/panels.py`
already computes real per-panel geometry (ADR 0007, `detect_panels`) and that
`validation/transform_geometry.py` already consumes an object's real panel bbox (ADR 0008,
Phase 3.3.1) — but grounding itself was, and had always been, run against the **full page
image**, completely independent of `analysis_mode`. `docs/decisions/0007-panel-aware-analysis.md`
made this explicit and deliberate at the time: "grounding, validation, segmentation, animation,
compositing, and rendering are identical code paths either way" — Phase 3.3's own scope was
"does panel-aware *analysis* help", not grounding, so grounding was left full-page by default,
not by an evaluated decision against cropping it.

The investigation's controlled crop experiment (real `detect_panels()` output, no ground-truth
box used) measured a large, reproducible effect on this exact page:

| Region | `"weapon."` best score | `"character_hair."` best score |
| --- | --- | --- |
| Full page (720x5062) | 0 candidates (best raw signal 0.20, below the 0.25 threshold) | 0 candidates |
| panel_00 (0,0)-(720,619) | 0.4775 | 0.7627 |
| panel_01 (0,571)-(720,2441) | 0.4300 | 0.6089 |
| panel_02 (0,2385)-(720,3410) | 0.3356 | 0.5563 |
| panel_03 (0,3357)-(720,5062) | 0 candidates | 0.5622 |

Two independent semantic categories both went from "below threshold on the full page" to
"comfortably above threshold on 3-4 of 4 real panel crops" — this rules out a weapon-specific
lexical/model explanation and points at page-scale/preprocessing (the model's own image
preprocessing degrades a very tall page's legibility once resized to its working resolution).
This matches a limitation `docs/phase3.2-results.md`'s "Remaining limitations" had already
flagged but never quantified: "the extreme-aspect-ratio resolution/OOM tension found in Phase
3.1 (long-edge capping crushes the short edge on a 7:1 page) is unchanged".

## Decision

**Grounding DINO now runs against an object's real panel crop when one is known, instead of
always the full page.** No new type or abstraction was introduced — the change reuses
conventions this codebase already has:

1. `grounding/ground.py::ground_object_candidates` (and its `ground_object` wrapper) gain one
   new keyword parameter, `panel_bbox_px: BBoxPx | None = None`, mirroring **exactly** the
   convention `validation.validate_target` already uses for its own `panel_bbox_px` parameter
   (same type, same meaning, same default) — not a new pattern.
2. When given, the function crops `image` to that region (`image[y0:y1, x0:x1]`) before calling
   `client.detect()`. Every returned detection's box — which `client.detect()` returns in
   coordinates relative to *whatever image it was given*, crop-local in this case — is
   translated by the region's own `(x0, y0)` offset and clipped against the **full page's**
   bounds (not just the crop's own bounds — a local box can legitimately overshoot its own
   crop's edge, exactly like the pre-existing full-page code already had to clip) before a
   `GroundingResult` is ever constructed. `GroundingResult.bbox` is therefore always in
   full-page pixel coordinates; nothing downstream of grounding (validation, segmentation,
   reconstruction, animation, compositing) ever observes crop-local coordinates, or needs to
   change to accommodate this.
3. `pipeline/orchestrator.py::run_pipeline` passes the **already-computed**
   `panel_bbox_px_by_object[obj.object_id]` — the same dict Phase 3.3.1 already built before the
   grounding stage for the transform-geometry check — straight into the grounding call. No new
   computation, no new call to `_panel_bbox_px`.
4. **Fallback behavior requires no new semantic distinction**, because it already falls out of
   how `panel_bbox_px_by_object` is built:
   - Page-level analysis (`analyze_page`) assigns every object to a single synthetic
     `PanelPlan(bbox=BBox(x=0, y=0, width=1, height=1))` (`analysis/plan_builder.py`) — in pixel
     space this is exactly `(0, 0, image_width, image_height)`.
   - Panel-level analysis's own "no internal gutters" case
     (`analysis/panels.py::_whole_page_candidate`, `source="fallback_full_page"`) is *also*
     exactly `(0, 0, image_width, image_height)` in pixel space (confirmed by reading the
     function directly: `BBoxPx(x0=0, y0=0, x1=width, y1=height)`).
   - Both therefore crop to the **entire image**, byte-identical to omitting `panel_bbox_px`
     entirely (`_grounding_region`'s fallback in `grounding/ground.py`). Only a real, smaller
     `gutter_xy_cut` panel actually changes what Grounding DINO sees. This was verified directly,
     not assumed: `test_ground_object_candidates_full_page_panel_bbox_is_equivalent_to_none`
     (unit) and `test_run_pipeline_analysis_mode_panel_falls_back_to_full_page_when_no_real_panels_exist`
     (orchestrator-level, using the real panel detector against a real gutter-less page).

## Live evidence (real GPU validation, this phase)

Performed on the user's live Kaggle Jupyter session (2x Tesla T4), reached via the same
non-browser Jupyter REST/kernel-WebSocket transport as every prior real run in this project (no
`claude-in-chrome`). Real repository code at commit `ec50d37` (branch `phase-5.1-wip`, later
merged to `main`), real `grounding-dino-base` (float32), real `Qwen2.5-VL-7B-Instruct` (float16,
matching `configs/kaggle.yaml`), real `sam2.1-hiera-base-plus`, real LaMa.

**Regression check (`panel_bbox_px=None`, must match the pre-Phase-5.1 baseline exactly):**
byte-identical — `phase3_action_page.png` still raises the same 0-candidate
`PipelineStageError`; `eval_weapon_effects.png` still returns the same 3 candidates at the same
scores (0.2696 / 0.2575 / 0.2551) as every prior documented session.

**`phase3_action_page.png`, real panel crops, `"weapon."`, full `validate_target` (semantic +
transform-geometry), real Qwen2.5-VL:**

| Panel | Grounding score | Semantic match | Transform compatible | Accepted |
| --- | --- | --- | --- | --- |
| panel_00 | 0.4775 | **False** ("dialogue text but no weapon") | — (not reached) | No |
| panel_01, rank 0 | 0.4300 | **True** (0.95, "shows a sword") | **True** | **Yes** |
| panel_01, rank 1 | 0.2600 | True (0.95, "curved weapon, likely a scythe") | True | Yes |
| panel_02, rank 0 | 0.3356 | False ("character wearing a helmet") | — | No |
| panel_02, rank 1 | 0.2531 | False ("part of a character's headgear") | — | No |
| panel_03 | — | 0 candidates | — | No |

The highest-scoring candidate (panel_00, 0.4775) is still correctly rejected — recall improving
does not mean validation rubber-stamps everything; this is the same ADR 0006 lesson (grounding
score is not a correctness signal) reconfirmed under the new architecture, not bypassed by it.

**Real, complete end-to-end render, this phase, for the first time in this project's history**:
a controlled `AnimationPlan` (`panel_id` = the real `panel_01`, `semantic_label="weapon"`,
`transform_kind=ROTATE`) run through the full, unmodified `run_pipeline` with every real client
(Grounding DINO, Qwen2.5-VL, SAM 2.1, LaMa) produced `grounding.bbox=(117,1643,592,1939)
score=0.43`, `validation: accepted=True`, `segmentation.iou_score=0.82`, a real reconstruction
pass, and a real `output.mp4` (4 frames, 8 fps, `seamless_loop_verified=True`) on disk. Every
prior real session (Phase 3.2, Phase 3.3, Phase 3.3.1, the Phase 5 audit behind ADR 0010's
"Revision" section) failed this exact page/prompt at the grounding stage, before validation,
segmentation, reconstruction, or rendering were ever reached.

**`eval_weapon_effects.png`, real panel crop (`fallback_full_page`, mathematically identical to
the full page), full `validate_target`:** unchanged — 2 semantic REJECT (unrelated stylized
text/character-design crops), 1 geometric REJECT (27.6% of the reference region, exceeding
ROTATE's 15% bound) — byte-identical reasoning and outcome to every prior documented session.
**No safety check was bypassed or weakened to obtain the `phase3_action_page.png` improvement.**

**Non-weapon control, real panel crops, `"character hair."`:** 0 candidates full-page, 0.56-0.76
on 4/4 real panel crops of the same page — confirms the fix is generic (Step 8 of the brief this
ADR implements explicitly required this: no weapon-specific or prompt-specific logic anywhere in
`grounding/ground.py`; the diff is entirely region-shaped, not label-shaped).

## Consequences

- `grounding.ground_object_candidates`/`ground_object` gain one optional parameter each;
  every pre-existing caller (none of which pass `panel_bbox_px`) is unaffected —
  confirmed by the full pre-existing test suite (323 tests) passing unmodified, plus 15 new
  targeted tests (`tests/test_grounding.py`, `tests/test_pipeline.py`) covering coordinate
  correctness (normal crop, edge-flush crop, translation, page-boundary clipping), identity
  preservation across two different real panels, the `panel_bbox_px=None`/`fallback_full_page`
  fallback, page-mode regression, panel-mode actually cropping (verified via a spy on the real
  image shape handed to the grounding client, not just on the render succeeding), no
  local-coordinate leakage downstream, and multi-object safety (two objects on different panels
  cannot swap crops/boxes).
- `pipeline/orchestrator.py` changes by exactly one call site (the grounding call now threads
  `panel_bbox_px_by_object[obj.object_id]` through) plus a docstring update; no change to stage
  ordering, error handling, or the PRIMARY/SECONDARY/MICRO failure policy from ADR 0010.
- Grounding DINO's checkpoint, prompts, and thresholds (`threshold=0.25`, `text_threshold=0.2`)
  are completely unchanged — the recall improvement comes entirely from giving the model a
  smaller, more legible image, never from relaxing an acceptance criterion.
- `validation/validate.py` and `validation/transform_geometry.py` are untouched — they already
  accepted `panel_bbox_px` since Phase 3.3.1 and needed no change.

## Known limitations

- This does **not** mean every grounding failure is now solved. `eval_weapon_effects.png`'s own
  PRIMARY failure is completely unaffected by this ADR, by construction: its panel detector
  already returns `fallback_full_page` (no internal gutters on that page), so its grounding crop
  is, and remains, mathematically identical to the full page both before and after this change.
  This ADR fixes an evidenced *preprocessing/scale* failure mode; it does not, and was never
  claimed to, fix *candidate-correctness* failures (wrong region semantically, or too large for
  its transform) — those remain exactly what `validation/` already exists to catch.
- A future page could still fail to ground for reasons this ADR doesn't touch: no real internal
  gutter structure at all (falls back to the pre-existing full-page behavior, unchanged), the
  target genuinely absent from the artwork, or a real detected panel itself being too large or
  low-quality (`analysis/panels.py`'s gutter detector has its own documented, unrelated
  limitations — see ADR 0007's "Open questions").
- This ADR makes an object's already-assigned panel crop available to grounding; it does not
  validate that the panel assignment itself (which `panel_id` the VLM/analysis stage picked for
  an object) is correct. A wrong panel assignment would still produce a wrong, if now smaller,
  crop — an analysis-stage question, out of this ADR's scope.
- Only two real pages and two real semantic categories (`weapon`, `character_hair`) were used as
  live evidence this phase — the same "small honest dataset" caveat every prior phase's real
  evaluation already carries (see `docs/phase3.3-results.md`).
- **The real end-to-end render described above animated a single PRIMARY object only** (a
  controlled one-object `AnimationPlan`). It demonstrates that this ADR's grounding fix unblocks
  a real PRIMARY render on `phase3_action_page.png`; it does **not** demonstrate a real,
  simultaneous multi-object (PRIMARY + SECONDARY/MICRO, each potentially on a different panel)
  render against live GPU models — that combination is currently verified only by deterministic
  fake-client tests (`tests/test_pipeline.py`'s identity/multi-object-safety tests), not by a
  live run. Not claimed as demonstrated; real future work if needed.
- The live E2E run confirmed `reconstruction` *ran* (not `None`) but its actual inpainted pixel
  content was not visually inspected this phase (unlike Phase 4's own real-data hole-mask visual
  check) — this ADR's evidence covers that reconstruction executes, not that its output quality
  is correct; that remains the same open, model-quality question ADR 0010's "Revision" section
  already flagged.
- **Minor DRY duplication, not a correctness issue**: `grounding/ground.py::_grounding_region`
  and `validation/transform_geometry.py::_reference_region` independently implement the same
  "`panel_bbox_px` when given, else the whole page/image as a `BBoxPx`" fallback. Both are
  correct and independently tested, but a future change to one's edge-case handling could drift
  from the other undetected. Worth consolidating into one shared `pipeline.types` helper in a
  future pass; not done here (no correctness bug to justify touching either module further).
- No test exercises a caller passing `panel_bbox_px` that is inconsistent with the actual
  `image` array's dimensions (e.g. computed against a different page). Every current production
  call site derives both from the same `page_shape`, so this is structurally unreachable today,
  not a live bug — but `ground_object_candidates` itself has no defensive check for it, only
  numpy's silent-truncation behavior on an out-of-range slice.

## Open questions

- Whether panel-cropping should also feed into the *analysis* stage's own image (not just
  grounding) for `analysis_mode="page"` is out of scope here — `analyze_page_panels` already
  exists and does this for VLM analysis specifically (ADR 0007); this ADR only changes what
  Grounding DINO receives, unconditionally, once an object (from either analysis mode) already
  has a `panel_id`.
- Whether a *second*, tighter crop (e.g. a saliency-based sub-region within a panel) would help
  further on a page where even the panel-level crop still fails is not investigated here — no
  real page in this project's evidence base needed it yet.

## Acceptance

An independent acceptance audit (separate pass, after this ADR's initial "Accepted" status and
live evidence above were recorded) re-verified every claim in this document against the
repository, the test suite, and git history rather than trusting this ADR's own account —
**verdict: PASS**. Confirmed independently: coordinate translation/clipping correctness, the
`None`/page-mode/`fallback_full_page` fallback equivalence, no existing test weakened or
deleted, 323/323 tests green with ruff/mypy clean, and every live-evidence number above matches
the real session output it was transcribed from. The audit is also the source of this section's
"Known limitations" additions (single-object E2E scope, reconstruction not visually inspected,
the `_grounding_region`/`_reference_region` duplication, and the untested
`panel_bbox_px`/`image` consistency assumption) — surfaced, not hidden, and explicitly left
unfixed as out of Phase 5.1's scope.
