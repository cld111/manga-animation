---
name: animation-planning
description: How to fill in an Animation Plan's MotionSpec — choosing amplitude/phase/speed/easing/pivot values that are kinematically sensible and satisfy the seamless-loop constraint, plus parent/child and confidence conventions. Load when drafting or reviewing MotionSpec values for an AnimationPlan.
---

# Animation planning

Practical guidance for turning "this object should move, roughly like this" into concrete
`MotionSpec` field values. Schema reference: `docs/animation-plan-schema.md` and
`src/manga_animation/schemas/animation_plan.py` — this skill is about *how to choose good
values*, not the field definitions themselves.

## Choosing `transform_kind` and `amplitude`

Pick the transform that matches the drawn motion cue as directly as possible — don't reach
for `mesh_warp` when `translate` or `rotate` would read correctly, since simpler transforms
are cheaper and easier to keep artifact-free:

| Drawn cue | Likely `transform_kind` | Typical `amplitude` range |
|---|---|---|
| Object drifting/swaying in place | `translate` | 0.01–0.05 (fraction of panel diagonal) — small |
| A held/hinged object swinging | `rotate` | 3–15° for subtle sway; higher only for the actual action's peak motion |
| A pulse/breathing/emphasis effect | `scale` | 0.02–0.08 fractional delta |
| Cloth/hair/flag rippling | `mesh_warp` | 0.05–0.2 normalized strength |
| A glint, a fade, a light flicker | `opacity` | 0.1–0.4 fractional delta |

These are starting points, not hard limits — the drawn motion's apparent scale should drive
the actual value. When unsure, prefer the smaller end of the range: see "Minimal Motion
Principle" in `docs/architecture.md` — subtle and readable beats large and distracting.

## `PRIMARY` / `SECONDARY` / `MICRO` amplitude and speed relationships

- `SECONDARY` motion should generally have **smaller amplitude and equal-or-lower speed**
  than the `PRIMARY` motion it follows from — physically, a follower's motion is damped
  relative to its driver's.
- `MICRO` motion should be barely-there: low amplitude, and ideally `phase`-offset from any
  `PRIMARY`/`SECONDARY` motion in the same panel so it doesn't read as synchronized (which
  would make it look intentional/meaningful rather than incidental).

## The seamless-loop `speed` rule

Under the default `loop.seamless=True` with `loop_mode="cycle"`, `speed` **must be a whole
number** — it's the count of full oscillation cycles completed over the loop's duration.
Practical defaults:

- `speed=1` for slow, single-sway motion across the whole loop (most hair/cloth sway).
- `speed=2` or `3` for faster repeated motion (a flickering effect, a fast flag flap) within
  the same 3–5s loop — higher `speed` reads as faster motion without changing loop length.
- If a motion should only happen once and hold (not repeat), use `loop_mode="once_hold"` —
  but only when `loop.seamless=False`. `once_hold` holds its end state indefinitely and
  never returns to rest, so the schema rejects it outright under a seamless loop (it would
  always produce a visible jump at the loop boundary); do not reach for it on a plan that
  needs `loop.seamless=True`.
- Use `loop_mode="ping_pong"` for a motion that should read as "there and back" without
  necessarily being a clean sinusoid (arbitrary `speed` is fine here) — this is the
  seamless-safe replacement for a one-shot motion when the loop must stay seamless.

## `phase` for multiple related objects

When several objects share similar motion (e.g. multiple strands of hair, several flags),
give each a distinct `phase` (e.g. spread across `[0, 1)`) rather than identical values —
identical phase across many objects reads as mechanical/synchronized rather than natural.

## `pivot` conventions

Default to `object_bbox` reference and anchor at the physically attached point: top-center
`(0.5, 0.0)` for things hanging/swaying from above (hair from a scalp, a flag from a pole
attachment point), the relevant edge-center for a hinged object. Get the pivot wrong and an
otherwise-correct amplitude/speed will still look physically implausible.

## `confidence` calibration

- High confidence (`>0.85`): the motion cue is explicitly drawn (motion lines, visible
  deformation) and the object/attachment is unambiguous.
- Medium (`0.5–0.85`): motion is plausible and common for this kind of object, but not
  explicitly drawn — an inference, not an observation.
- Low (`<0.5`): still worth recording (useful for QA/review) but should generally push
  toward `STATIC` instead, per the manga-analysis skill's checklist.
