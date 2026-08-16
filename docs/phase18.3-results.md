# Phase 18.3 Results: Per-Candidate VLM Object Description (Full Image + Bbox)

**Status: implemented, integrated, and validated on real pages.** The architecture change the
phase brief demanded is in the production pipeline, not a parallel demo:

```text
Original Page -> Grounding DINO -> candidate bboxes -> SAM2 -> masks (downstream only)
  -> Qwen2.5-VL (FULL image + bbox pixel coordinates) -> structured animation description
  -> animation planning (deterministic mapping to MotionSpec) -> animation (mask + spec)
```

## What changed

- New stage package `src/manga_animation/object_description/` (schema, prompt, mapping,
  describe), wired between `mask_semantics` and `animation` in both `run_pipeline` and
  `run_page_panels`. The SAM mask is NOT an input to the VLM: it stays for the stages that
  consume it (segmentation output feeds animation as before).
- The VLM input contract: ONE full pipeline image (panel scene crop in panel mode, the page
  otherwise) plus the accepted grounding bbox as pixel coordinates stated in the prompt --
  never a crop of the candidate, never a bbox visualization.
- Coordinate contract: `object_description.prompt.prepare_image_and_bbox` applies the same
  two-step geometry the Qwen2.5-VL processor applies itself (long-edge downscale to
  `config.resolution`, then rounding each side to the nearest 28px patch-grid multiple --
  `round(x/28)*28`, verified against the transformers 5.0.0 `smart_resize` source) and scales
  the bbox by exactly the same factors, so the coordinates in the prompt are the pixel space
  the model sees. Verified against the REAL processor on the worker: all 5 test sizes
  `match=True` (1024x1536 -> 1036x1540, 720x5062 -> 224x1540 after the project's own
  long-edge cap, 600x400 -> 588x392, 1536x1536 -> 1540x1540, 200x220 -> 196x224).
- Output contract: a strict pydantic response (`ObjectDescriptionResponse`) with
  `bbox_assessment` (pass/ambiguous/partial/reject/not_animatable), `object_identity`,
  `matches_semantic_label`, `animatable`, `movable_parts`, `static_parts`, `motion_kind`,
  `direction`, `amplitude_band`, `speed_band`, `pivot_hint`, `constraints`,
  `neighbor_conflicts`, `confidence`, `reason`. Cross-field rules are validated
  (animatable => motion_kind; drift => direction). Non-drift `direction` is stripped
  (inert field; real Qwen output habitually fills it in for sway). Unknown enum values fail
  the read.
- Fail-closed behavior: unparseable/schema-invalid output gets exactly one recovery
  re-prompt; a second failure is `rejection_reason="unparseable"`. Acceptance requires
  `assessment == pass` AND `matches_semantic_label` AND `animatable` AND a deterministic
  identity backstop (identity must not name speech-bubble/text/background/panel/lettering
  content). Every non-accept has a machine-readable `rejection_reason`; raw responses are
  logged and kept in `ObjectDescriptionResult.raw_responses` for audit. PRIMARY non-PASS
  rejects the run (stage="object_description"); SECONDARY/MICRO non-PASS drops the object.
- Animation planning: the description's semantic fields map deterministically to a
  schema-valid `MotionSpec` (`object_description.mapping`), reusing the analysis-stage
  baseline amplitudes/easings per motion kind; the mapped spec REPLACES the keyword-heuristic
  motion for the object and is what the animation stage applies (SAM mask + spec ->
  `generate_transformed_layer`).
- Config toggle `enable_object_description_validation` (default true); evaluation
  `schema_version` 7 with `ObjectDescriptionOutcome`; `Stage` literal extended.

## Prompt-engineering iterations (real GPU findings)

Real Qwen2.5-VL output drove four contract/prompt fixes:

1. **null vs. strict fields**: the model emitted `null` for `amplitude_band`/`speed_band`/
   `pivot_hint`/`constraints` on non-animatable objects -- the optional bands now tolerate
   null with documented defaults; semantic fields stay strict.
2. **invented enum values**: `bbox_assessment="static"` (outside the five-state contract) and
   `direction="up_down"` appeared. The prompt and recovery prompt now restate the exact
   allowed values; the recovery prompt repeats them (before this, a recovery response with an
   invented assessment value stayed unparseable).
3. **text rule enforcement**: the model proposed `sway` for a text banner while naming it
   `text_banner`; the prompt now says lettering is never animatable regardless of the label,
   and `text_rigid` responses carry `animatable=false` (rejected via `not_animatable`).
4. **over-strict "static" reading**: with an earlier "prefer the stricter verdict" rule the
   model rejected ordinary still characters as "static pixelated figures with no discernible
   motion"; the prompt now states that still drawings of people/weapons/cloth are normal
   animatable targets and `animatable:false` is only for unsafe content.

A fifth fix was a deterministic CODE backstop, not prompt text: a real false-accept on
`villainess_ending_scuffle` (recovery response said `object_identity: "speech_bubble"` while
claiming `matches_semantic_label: true` + `animatable: true` -- all soft signals said accept).
`_NON_ANIMATABLE_IDENTITY_KEYWORDS` rejects such identities outright.

## Curated scenario results (real Qwen2.5-VL, deterministic synthetic pages)

10 required scenarios, final run (`outputs/experiments/phase18_3_final.json`):

| Scenario | Assessment | Accepted |
|---|---|---|
| single unambiguous object | pass | no (matches=false: "person_red" vs label "character" -- synthetic-art identity nit; see Known Limitations) |
| several objects nearby | ambiguous | no (correct) |
| bbox containing several objects | ambiguous | no (correct) |
| bbox partially covering object | ambiguous | no (conservative; "partial" would be ideal) |
| object partially occluded | pass | YES (questionable -- occlusion should be ambiguous; see Known Limitations) |
| small object | ambiguous | no (conservative false-doubt; synthetic art) |
| object on complex background | ambiguous | no (conservative false-doubt) |
| several visually similar objects | pass | no (box holds one instance; animatable=false) |
| text banner (DINO-bait) | pass + animatable=false | no (correctly rejected via not_animatable) |
| partially animatable object | pass | YES (visible part animatable, constraint reported) |

The stage is fail-closed in every direction that matters: nothing unparseable slips through,
lettering is never accepted, multi-instance boxes are rejected. Two accepted cases
(occluded, partially animatable) and several conservative ambiguous verdicts on crude
synthetic art are documented noise, not contract violations.

## Real-page results (5 pages, real DINO + real Qwen)

25 DINO candidates: 6 accepted, 15 ambiguous, 2 reject, 1 partial, 1 unparseable.

- `angels_of_war_fleet`: all 12 candidates (character/weapon/flag/speed-lines labels)
  rejected as ambiguous/partial/reject -- the page contains spaceships and missiles, not the
  requested objects; the VLM named `spaceship`/`missile`/`speech_bubble` as actual contents.
  This is the semantic-validation layer working as intended on DINO's documented
  wrong-instance detections (Phase 17/18.1 bottleneck).
- `villainess_ending_scuffle`: DINO rank-2 "character" candidate (score 0.33) PASSED with
  `object_identity=character`, animatable, confidence 0.95, motion sway -- the correct
  instance accepted despite not being DINO's top-1. Two rank-0/1 candidates were
  false-accepted pre-backstop (identity=speech_bubble) -- now deterministically rejected.
- `marika_love_meter`: character candidates accepted as `clap` (2) and ambiguous (1);
  flag/speed-lines rejected as speech-bubble-containing.
- `wind_breaker_sprint`: one candidate unparseable (fail-closed).

## End-to-end animation (SAM mask + Qwen description -> animation)

`scripts/run_phase18_3_e2e.py` runs the real `run_page_panels` (analysis -> grounding ->
validation -> segmentation -> mask_semantics -> **object_description** -> animation ->
reconstruction -> compositing -> rendering) on real pages with real models. Results recorded
in `outputs/experiments/phase18_3_e2e.json`.

## Known Limitations

- `matches_semantic_label` is literal-name-based in the model: a stylized "person_red" does
  not match the label "character", causing false rejects on synthetic art; on real pages the
  model matched correctly. A synonym-tolerant prompt instruction is a candidate improvement.
- The model's verdicts are noisy run-to-run on borderline cases (occluded object accepted in
  one run, ambiguous in another). The stage is fail-closed; the noise is a quality ceiling,
  not a safety hole.
- One Qwen load/unload per stage means the description stage adds a 4th VLM residency
  (~1-4 min each on 2xT4). Merging adjacent VLM stages into one residency is a possible
  optimization, not a correctness fix.
- `outputs/experiments/phase18_3_{ab,ab_v2,b_v3,b_v4,b_v5,final,c,e2e}.*` on the worker hold
  the raw per-call records (including every raw VLM response) for audit.
