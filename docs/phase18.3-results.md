# Phase 18.3 Results: Per-Candidate VLM Object Description (Full Image + All Bboxes, One Call)

**Status: implemented, integrated, and validated end-to-end on real manga pages.**

## Final architecture (the pipeline's ONLY VLM stage)

```text
Original Page -> deterministic panel detection -> scene crops
  -> GROUNDING (DINO, labels from the caller) -> candidate bboxes
  -> SEGMENTATION (SAM2) -> masks (kept for the animation stage; never sent to the VLM)
  -> OBJECT_DESCRIPTION (Qwen2.5-VL, loaded ONCE per page):
       input = ONE full panel image + ALL its candidate bboxes as pixel coordinates
       (native <|box_start|> tokens + plain numbers + image dims), one generate() call
       output = ONE JSON array, one strict description per box (box_index-mapped)
  -> ANIMATION PLANNING (deterministic): highest-confidence accepted -> PRIMARY, rest
     SECONDARY; transform-geometry gate; cross-panel gate
  -> ANIMATION (SAM mask + description-mapped MotionSpec) -> RECONSTRUCTION (LaMa)
  -> COMPOSITING -> RENDERING (H.264, loop metrics)
```

There is no analysis stage, no crop-based VLM validation and no mask-semantics stage: Qwen
is called exactly once per panel image (per page residency), and every animation decision
comes from its structured description. Labels are supplied by the caller
(`DEFAULT_ANIMATION_LABELS` when omitted); the pipeline never invents them with a VLM.

## Key contracts implemented

- **Coordinate contract**: `prepare_image_and_bbox` applies the exact geometry the
  Qwen2.5-VL processor applies (long-edge downscale to `config.resolution`, then
  `round(x/28)*28` patch-grid rounding) and scales every bbox by the same factors. Verified
  against the REAL processor on the worker: all 5 test sizes match exactly
  (1024x1536 -> 1036x1540; 720x5062 -> 224x1540 after the project's long-edge cap;
  600x400 -> 588x392; 1536x1536 -> 1540x1540; 200x220 -> 196x224).
- **One call per image**: `describe_objects` sends the image and ALL candidate bboxes in one
  prompt; the model answers a JSON array with `box_index` per entry. Missing/duplicate
  indices fail that candidate closed. `max_new_tokens` raised to 4096 (a 10-box batch needs
  ~2-3k tokens; the earlier 512-token default truncated answers and failed the batch).
- **Strict schema + fail-closed**: `ObjectDescriptionResponse` validates every entry
  (assessment enum, matches, animatable, motion_kind/direction cross-rules). One recovery
  re-prompt restates the allowed values. Non-pass assessment, label mismatch, non-animatable
  content, or an identity-conflict (speech_bubble/text/background/panel/lettering -- the
  observed real false-accept) all reject the candidate with a machine-readable reason. Raw
  responses are logged and kept in `ObjectDescriptionResult.raw_responses`.
- **Action-driven prompt**: the model first reads what is HAPPENING in the scene (STEP 1:
  READ THE ACTION), then judges the candidate with the action in mind, then derives the
  motion from the action. Real responses: "The character is in motion, with visible sweat
  indicating exertion", "The hair is flowing due to the character's movement", "The flag is
  waving due to the wind", "The speed lines indicate movement and can be animated", "The
  impact burst indicates a forceful impact and can be animated".

## Real end-to-end result (wind_breaker_sprint, real Qwen+DINO+SAM+LaMa)

`scripts/run_phase18_3_e2e.py`, outputs/experiments/phase18_3_e2e_v4.json:

- **All 4 panels PASS**, each a real 96-frame (4s @ 24fps) H.264 video.
- **VLM calls: 2, boxes per call: [10, 9]** -- exactly ONE Qwen call per panel, each
  covering ALL of that panel's segmented candidates at once (the input contract).
- Animation planning per panel: PRIMARY=flag_banner + 5 secondaries (panel 2);
  PRIMARY=speed_lines + 6 secondaries (panel 4) -- the accepted descriptions drove the
  transforms.
- Decoded-video verification: 96 frames each; 79-87 of 95 adjacent-frame pairs moved
  (real motion); mean per-frame change 1.0-10.6; wrap-step 2.3-17.8 (multi-object loops
  with integer speeds).
- The deterministic gates kept working on real data: 25 mask-asymmetry drops at
  segmentation, semantic-accept + geometry-reject for edge-touching hair/flags (mesh_warp
  edge margin), and `not_animatable` reads for some effect candidates in an earlier run.

## Notes

- `scripts/run_phase18_3_gpu_benchmark.py` holds the coordinate-contract check (part A),
  the curated scenario pages (part B) and the real-page DINO-candidate batch (part C);
  raw per-call records including every VLM response are under
  outputs/experiments/phase18_3_*.json on the worker.
- Known quality ceiling (documented, not a safety hole): the model's verdicts are noisy
  run-to-run on borderline candidates, and it can read drawn effects (speed lines/impact
  burst) as not_animatable in some runs. Everything stays fail-closed either way.
