---
name: manga-analysis
description: Domain knowledge for reading a manga page like a manga reader would — panel/reading order, composition conventions, and the STATIC vs. ANIMATED motion-selection heuristics used before drafting an Animation Plan. Load when interpreting a manga page's panels, characters, objects, or action.
---

# Manga analysis

Practical heuristics for turning a manga page into the semantic understanding an Animation
Plan needs (see `docs/animation-plan-schema.md`). This is knowledge, not code — it informs
what `vision-agent` (or whoever is analyzing a page) should look for.

## Reading order and panel structure

- Traditional manga reads **right-to-left, top-to-bottom**; confirm orientation before
  assuming panel order — a mis-ordered read can misattribute cause/effect between panels
  (which matters for motion: what caused what).
- A panel's *gutters* (the space between panels) and panel *borders* (or their absence —
  bleed panels) are part of the composition. A borderless/bleeding panel is often used
  specifically to convey large or unconstrained motion — a signal worth weighting.
- A splash page (single full-page panel) should still be modeled as one `PanelPlan`
  covering the page, not skipped.

## Reading manga's motion vocabulary

Manga artists draw *implied* motion with specific conventions — these are strong signals
for which objects are motion-relevant, and even for a first guess at direction/amplitude:

- **Speed lines / motion lines**: streaks parallel to an object's implied trajectory —
  strong signal for `PRIMARY` motion in that direction.
- **Impact/focus lines** (radiating from a point): mark the moment of an action (a hit, a
  landing) — the page is often depicting the *peak* of a motion, not its middle, which
  matters for phase/amplitude choices later.
- **Wavy/flowing linework** on hair, cloth, or flags: drawn to depict continuous motion
  (wind, movement) even in a static image — a strong candidate for `SECONDARY` motion.
- **Small repeated marks** (sweat drops, dust motes, floating petals): usually `MICRO`
  motion candidates, low narrative weight but adds life.
- **Deliberately static, heavily-inked backgrounds** behind a dynamic foreground figure:
  a signal the background should stay `STATIC` even though the foreground is dynamic — don't
  let one dynamic panel push you toward animating everything in it.

## STATIC vs. ANIMATED checklist

Before marking anything other than `STATIC`, be able to answer:

1. **What in the drawing itself justifies motion here?** (a motion line, implied
   trajectory, drawn deformation, physical attachment to something that must move) — not
   "this is a character, characters can move."
2. **Is this the thing carrying the action (`PRIMARY`), something that would follow from a
   `PRIMARY` mover (`SECONDARY`), or independent subtle life (`MICRO`)?**
3. **Would a reader specifically notice if this did *not* move?** If not, default to
   `STATIC` — see "Static Is a Valid Result" in `docs/architecture.md`.
4. **Am I confident, or guessing?** Reflect that honestly in `ObjectPlan.confidence` rather
   than picking a motion type you're not sure about and hiding the uncertainty.

## Common over-animation traps

- Animating an entire background "for atmosphere" — background stays `STATIC` unless a
  specific drawn element (a banner, smoke, rain) justifies motion.
- Animating a character's whole body when only one part (hair, a held object) is the
  actual motion cue.
- Treating every character in an action panel as equally dynamic — usually only the
  actor and the immediately affected object/character are `PRIMARY`; bystanders are
  `STATIC` or at most `MICRO`.
