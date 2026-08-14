# The Animation Plan schema

Implementation: [`src/manga_animation/schemas/animation_plan.py`](../src/manga_animation/schemas/animation_plan.py)
(pydantic v2 models — this document explains the design; the code is the source of truth
for exact field constraints).

## What this is for

The Animation Plan is the contract between the *semantic* half of the pipeline (VLM/panel
analysis: "what's happening, what should move, why") and the *mechanical* half (grounding,
segmentation, CV animation, compositing, rendering: "where is it in the image, and how do
pixels move"). See [pipeline.md](pipeline.md) for where it sits.

Two consequences follow from that split:

1. **It is deliberately pixel-free.** No bounding boxes or masks on objects — those are
   *produced from* the plan by the grounding/segmentation stages, not stored on it.
   `PanelPlan.bbox` is the one exception: panel layout is a property of the page itself,
   decided by the analysis stage, not something later stages derive.
2. **It is fully resolution-independent.** Spatial values (`BBox`, `PivotSpec` and direction
   vectors) are normalized to `[0, 1]` or use a unit vector. `MotionSpec.amplitude` is
   transform-dependent (for example, degrees for rotation), so it is not universally normalized.
   The same plan is
   valid whether the image was analyzed at a 1024px local debug resolution or a 2048px
   remote GPU resolution (`configs/local.yaml` vs. `configs/kaggle.yaml`).

## Top-level shape

```text
AnimationPlan
├── plan_id, schema_version, created_at
├── source: SourceImage            (path, width, height, checksum)
├── panels: [PanelPlan]            (>= 1; one panel for a splash page is fine)
│     ├── panel_id
│     ├── bbox: BBox               (normalized, relative to the page)
│     └── description
├── objects: [ObjectPlan]          (may be empty — an all-STATIC semantic plan is valid)
│     ├── object_id, panel_id, semantic_label, confidence
│     ├── motion_type: STATIC | PRIMARY | SECONDARY | MICRO
│     ├── parent_id, children_ids  (kinematic hierarchy)
│     └── motion: MotionSpec | None
│           ├── transform_kind: translate | rotate | scale | shear | mesh_warp | opacity | radial_expand
│           ├── direction: Vector2 | None   (unit vector; required for translate/shear)
│           ├── amplitude                   (meaning depends on transform_kind — see below)
│           ├── phase, speed, easing
│           ├── pivot: PivotSpec
│           └── timing: TimingSpec
└── loop: LoopSpec                 (duration_s, fps, seamless, crossfade_frames)
```

## `MotionType`: STATIC / PRIMARY / SECONDARY / MICRO

- **STATIC** — no motion. This is the default and the preferred outcome (see "Static Is a
  Valid Result" in [architecture.md](architecture.md)). A STATIC `ObjectPlan` must not
  carry a `motion` spec at all — the schema rejects one if present, so "static but has
  leftover motion params" can't silently exist.
- **PRIMARY** — motion that carries the page's action: the thing the reader's eye should
  read as moving on purpose (a swung weapon, a jumping figure).
- **SECONDARY** — motion that follows from a primary mover rather than acting on its own
  (cloth, hair, a cape) — typically a child of a PRIMARY or STATIC anchor object.
- **MICRO** — small motion that adds life without carrying narrative weight (a blink, a
  subtle sway) — usually low amplitude, often independent of any PRIMARY motion.

Every non-STATIC object *must* carry a `motion` spec — the schema rejects `PRIMARY` /
`SECONDARY` / `MICRO` objects with no motion just as it rejects STATIC objects that have
one. Motion presence and `motion_type` can't drift apart.

An `AnimationPlan` may contain at most one `PRIMARY` object. Multiple independent movers
must use `SECONDARY`/`MICRO` for the additional objects; this keeps the orchestrator and
evaluation contract unambiguous.

## `transform_kind` and what `amplitude` means

`MotionSpec.amplitude` is a single scalar whose unit depends on `transform_kind`, rather
than a separate typed field per transform (kept deliberately simple — see "Do not
over-engineer" in the project brief):

| `transform_kind` | `amplitude` means | `direction` |
|---|---|---|
| `translate` | fraction of the panel diagonal | **required**, normalized to a unit vector |
| `shear` | shear factor | **required**, normalized to a unit vector |
| `rotate` | degrees (sign = direction: +CW/−CCW) | unused |
| `scale` | fractional size delta (e.g. `0.05` = ±5% at peak) | optional (axis hint; omitted = uniform) |
| `opacity` | fractional opacity delta | unused |
| `mesh_warp` | normalized warp strength | optional (flow hint) |
| `radial_expand` | peak rim displacement as a fraction of the object bbox's longest side | unused |

`amplitude` must be `> 0` — a "motion" with zero amplitude isn't motion, it's a STATIC
object with extra steps, so the schema pushes that case toward `motion_type=STATIC`
instead.

`radial_expand` (Phase 16, Drawn Effect Track) is the drawn-effect motion model for the
radial class of manga effects — impact bursts, energy fields, radiating focus lines, glow.
It is a spatially-varying radial pulse about the object's own pivot: the center stays
effectively fixed while the rim breathes outward/inward, unlike uniform `scale` (which
moves the whole footprint as one rigid block). `amplitude` is the peak rim displacement as
a fraction of the object bbox's longest side (e.g. `0.08` = the rim moves 8% of that side
at peak). It requires no `direction`; the radial axis is derived from the pivot geometry.

## Kinematic hierarchy: `parent_id` / `children_ids`

Objects can declare a kinematic parent (`parent_id`) and optionally cache child IDs
(`children_ids`). `parent_id` is the source of truth: `AnimationPlan.children_of(object_id)`
derives the effective child list from it. When `children_ids` is supplied, it is checked
against the corresponding children's `parent_id`; omitting it is valid. This deliberately
avoids requiring two independently generated fields to be kept synchronized. Self-parenting
and parent cycles are rejected. Automatic parent-transform inheritance is not implemented.

The current implementation validates these links but does not automatically inherit or
compose a parent's transform during animation. Callers must provide a separate motion spec
for each object until coordinated motion is implemented.

## `pivot`, `phase`, `speed`, `easing`, `timing`

- **`pivot`** — the anchor point a rotate/scale/shear is applied around, normalized to a
  `reference` frame (`object_bbox` by default; `panel`/`page` for motion anchored outside
  the object itself). E.g. `(0.5, 0.0)` on `object_bbox` = top-center — the natural anchor
  for hair swaying from the scalp.
- **`phase`** — where in its cycle the motion starts, as a fraction (`[0, 1)`). Lets two
  objects share a `speed` but move out of sync.
- **`speed`** — cycles completed per full loop duration. See the seamless-loop constraint
  below — this is not an arbitrary free float.
- **`easing`** — `linear` / `ease_in` / `ease_out` / `ease_in_out` / `sine`.
- **`timing`** — `delay_s` (when motion starts within the loop), `duration_s` (`None` =
  spans the rest of the loop), and `loop_mode`: `cycle` (repeats), `once_hold` (plays once,
  holds the end state — only valid when `loop.seamless=False`, see below), or `ping_pong`
  (plays forward then reverse).

## The seamless-loop constraint on `speed` and `loop_mode`

`LoopSpec.seamless` (default `True`) means the plan promises frame 0 and the frame after
the last one are visually identical. Two `loop_mode`s are checked against that promise:

- **`cycle`**: the underlying motion curve is periodic (e.g.
  `sin(2π·(speed·t/duration + phase))`); that curve only returns exactly to its start value
  at `t = duration` when `speed` is a **whole number** of cycles. So under a seamless loop,
  a `cycle` object's `speed` must be integer-valued (`1`, `2`, `3`, ...) — the schema
  validates this and raises with a message explaining the three ways out (use an integer
  speed, switch `loop_mode` to `ping_pong`, or set `loop.seamless = False`) rather than just
  rejecting the value.
- **`once_hold`**: by construction (see `animation/curves.py::sample_motion_value`) this
  mode sweeps away from rest (`0.0`) once and then holds its end state (`1.0`) for the rest
  of the loop — it never returns to `0.0` on its own. Every fresh loop iteration restarts the
  object at rest, so pairing `once_hold` with a seamless loop always produces a visible jump
  at the boundary (held end state -> rest), regardless of `speed`, `easing`, or how long the
  motion window is. The schema therefore rejects `once_hold` outright whenever
  `loop.seamless=True`, for any object that carries motion. `once_hold` remains valid — and
  is the correct choice for a genuinely one-shot, non-repeating motion — whenever
  `loop.seamless=False`.
- **`ping_pong`** returns to rest (`0.0`) on its own once its window closes (see the same
  module), so it is compatible with a seamless loop at any `speed`, integer or not — it is
  the recommended replacement for both of the cases above when the loop must stay seamless.

`AnimationPlan` also validates that every object's `delay_s + duration_s` fits inside
`loop.duration_s` — a motion window that runs past the end of the loop is rejected at plan
construction time, not discovered later at render time. `duration_s * fps` must also round to
at least one output frame.

## Confidence

`ObjectPlan.confidence` (`[0, 1]`) is the analysis stage's confidence in *both* the
STATIC/ANIMATED decision and, where applicable, the motion parameters attached to it. It's
intended for downstream QA (e.g. flagging low-confidence PRIMARY motion for review) and for
future re-planning heuristics, not currently enforced against any threshold in the schema
itself — thresholding is a policy decision for the stage that consumes the plan, not a
schema-level constraint.

## Serialization

`AnimationPlan` is a pydantic model: `model_dump_json()` / `model_validate_json()` give
JSON round-tripping for free, and `to_json_file(path)` / `from_json_file(path)` wrap those
for the common case of persisting a plan next to its source image. See
`tests/test_animation_plan.py` for round-trip behavior under test.

## Full example

Generated directly from the schema (four objects: a static flag pole with a mesh-warped
cloth child, and a static head with a translating hair child):

```json
{
  "plan_id": "602a35573c594ab4ba7d5fac6c6a6622",
  "schema_version": "1.0",
  "source": {
    "path": "examples/page_014.png",
    "width": 1600,
    "height": 2400,
    "checksum": "sha256:..."
  },
  "created_at": "2026-08-11T19:17:00.458999Z",
  "panels": [
    {
      "panel_id": "panel_1",
      "bbox": { "x": 0.05, "y": 0.05, "width": 0.9, "height": 0.4 },
      "description": "Character raises a hand-held flag on a windy rooftop."
    }
  ],
  "objects": [
    {
      "object_id": "obj_flag_pole",
      "panel_id": "panel_1",
      "semantic_label": "flag_pole",
      "confidence": 0.97,
      "motion_type": "static",
      "parent_id": null,
      "children_ids": ["obj_flag_cloth"],
      "motion": null
    },
    {
      "object_id": "obj_flag_cloth",
      "panel_id": "panel_1",
      "semantic_label": "flag_cloth",
      "confidence": 0.93,
      "motion_type": "primary",
      "parent_id": "obj_flag_pole",
      "children_ids": [],
      "motion": {
        "transform_kind": "mesh_warp",
        "direction": null,
        "amplitude": 0.12,
        "phase": 0.0,
        "speed": 2.0,
        "easing": "sine",
        "pivot": { "x": 0.0, "y": 0.5, "reference": "object_bbox" },
        "timing": { "delay_s": 0.0, "duration_s": null, "loop_mode": "cycle" }
      }
    },
    {
      "object_id": "obj_head",
      "panel_id": "panel_1",
      "semantic_label": "character_head",
      "confidence": 0.99,
      "motion_type": "static",
      "parent_id": null,
      "children_ids": ["obj_hair"],
      "motion": null
    },
    {
      "object_id": "obj_hair",
      "panel_id": "panel_1",
      "semantic_label": "character_hair",
      "confidence": 0.85,
      "motion_type": "secondary",
      "parent_id": "obj_head",
      "children_ids": [],
      "motion": {
        "transform_kind": "translate",
        "direction": { "x": 1.0, "y": 0.0 },
        "amplitude": 0.02,
        "phase": 0.15,
        "speed": 1.0,
        "easing": "ease_in_out",
        "pivot": { "x": 0.5, "y": 0.0, "reference": "object_bbox" },
        "timing": { "delay_s": 0.0, "duration_s": null, "loop_mode": "cycle" }
      }
    }
  ],
  "loop": { "duration_s": 4.0, "fps": 24, "seamless": true, "crossfade_frames": 0 }
}
```

Note `obj_flag_cloth.speed = 2.0` (a whole number, valid under the default seamless loop)
versus `obj_hair.speed = 1.0` with a `phase` offset — two independently-timed motions that
still satisfy the seamless-loop constraint.
