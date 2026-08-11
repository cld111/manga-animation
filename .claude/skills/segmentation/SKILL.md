---
name: segmentation
description: Practical workflow and quality checklist for grounding Animation Plan objects into pixel regions and producing clean segmentation masks — bounding boxes, SAM-style prompting, object-part segmentation, overlap/occlusion handling, mask validation. Load when doing grounding or segmentation work.
---

# Segmentation

Practical workflow for turning `AnimationPlan` objects (semantic labels) into precise pixel
regions. Schema/architecture reference: `docs/animation-plan-schema.md`,
`docs/architecture.md` ("Local Modification"). This skill is the *how*, not model selection
(that's a Phase 2 benchmarking decision, recorded in `PipelineConfig.model_variants`).

## Grounding workflow

1. **Start from the plan, not the image.** For each non-`STATIC` `ObjectPlan` (and any
   `STATIC` object that's a parent of one — its region may still be needed as a spatial
   reference even though it isn't itself animated), resolve `semantic_label` to a region.
2. **Use panel context to disambiguate.** `ObjectPlan.panel_id` scopes the search to one
   panel's `bbox` — don't let a grounding model search the whole page when the plan already
   says which panel the object is in.
3. **Prefer a bounding box pass before a mask pass.** A coarse box grounding step
   (detection/grounding model) followed by a box-prompted precise segmentation
   (SAM-style) is cheaper and more controllable than asking a segmentation model to find
   the object unprompted.
4. **Only segment what the plan needs.** If the plan's `transform_kind` is `opacity` or
   uniform `scale`, a tight bounding region may be sufficient — a full precise mask is
   worth its cost mainly for `mesh_warp`, `translate`, and `rotate`, where mask edges are
   visible against the background during the transform.

## Overlap and occlusion

- When two planned objects' regions overlap (e.g. a hand in front of a held object), decide
  and record which one is "in front" using the drawn depth cues (overlap, size, panel
  composition) — this ordering matters later for compositing.
- An occluded object's mask should represent only its *visible* extent, not an inferred
  full shape, unless hidden-region reconstruction (a separate, later stage) is explicitly
  filling in the occluded part.
- Don't let one object's mask "leak" into an adjacent, unrelated object's region —
  over-inclusive masks are a common source of compositing artifacts later.

## Object-part segmentation

Only split an object into parts when the Animation Plan's motion actually requires
independent movement of the parts (e.g. individual hair strands with slightly different
`phase`). If the plan treats "hair" as one `ObjectPlan`, one mask is correct — don't
over-segment into strands nobody asked to animate independently (see "Local Modification"
in `docs/architecture.md`).

## Mask validation checklist

Before handing a mask off to `cv-agent`/compositing:

- **No stray disconnected components** unless the object is genuinely disjoint in the
  drawing (e.g. two separated cloth pieces) — a stray few pixels elsewhere is almost always
  an error, not a real part of the object.
- **No unintended holes** inside a mask that should be solid (check especially where the
  object overlaps heavy linework or screentone).
- **Edges follow inked linework**, not a soft/approximate boundary — manga line art has
  crisp edges, and a mask that doesn't hug them will show a halo/seam once transformed.
- **Mask extent is consistent with the object's declared `panel_id` bbox** — a mask that
  extends outside its panel likely grounded to the wrong region.
- **Coverage sanity check**: mask area should be a plausible fraction of its bounding box
  for the object type (e.g. hair masks are rarely near-100% of their bbox; solid props
  often are) — a wildly implausible fill ratio is worth a second look before proceeding.
