"""Transformation-aware geometric compatibility: does a semantically-correct grounding

candidate's bbox actually make sense for the SPECIFIC transform the `AnimationPlan` intends to
apply to it? See docs/decisions/0008-transform-aware-target-validation.md for the full design
rationale; this module is the deterministic, no-model-call check that decision introduces.

Real motivating defect (Phase 3.3 real E2E run, `eval_weapon_effects.png`): Grounding DINO's
candidate for "weapon" scored 0.255, and the VLM's semantic crop-check correctly answered "yes,
this crop plausibly shows a weapon" — both real, individually defensible signals (see
`validation/validate.py`'s semantic check). But the bbox itself covered nearly the entire dark
action panel (sound-effect text, panel border, and surrounding background artwork included),
and the plan's `rotate` transform then visibly swung the WHOLE panel, not the weapon, producing
torn black-wedge artifacts at the frame edges where the rotation revealed background outside
the bbox's own footprint. "Is this a weapon" and "is this specific box safe to rotate" are
independent questions — `validate_target` answers the first with the VLM; this module answers
the second deterministically, using only the geometry already available (no mask exists yet at
this point in the pipeline — segmentation runs after validation).

Only `check_transform_geometry` is the public entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

from manga_animation.pipeline.types import MAX_OBJECT_COVERAGE_FRACTION, BBoxPx
from manga_animation.schemas.animation_plan import TransformKind


@dataclass(frozen=True, slots=True)
class TransformGeometryProfile:
    """How much geometric risk one `TransformKind` poses to pixels outside the intended object,

    and the concrete bounds that follow from that risk (see the per-kind rationale next to
    `_TRANSFORM_GEOMETRY_PROFILES` below). Every field is a maximum tolerance / minimum
    clearance for THAT transform kind specifically — there is deliberately no single shared
    number reused across kinds (see the Phase 3.3.1 brief's explicit "do not use one universal
    threshold" instruction); a kind whose real mechanism poses no extra risk beyond the
    existing generic bbox check (`OPACITY`) is simply given that generic bound back, not an
    invented stricter one.
    """

    max_area_fraction: float
    """Max plausible bbox area as a fraction of its reference region's area (the object's
    panel, or the full page when panel-level analysis wasn't used — see `_reference_region`)."""

    min_edge_margin_fraction: float
    """Minimum required clearance between the bbox and its reference region's edges, as a
    fraction of the reference region's shorter side. `0.0` = no clearance required."""

    max_aspect_ratio: float | None
    """Reject a bbox whose long/short side ratio exceeds this. `None` = not checked (not every
    kind has a meaningful aspect-ratio risk — see below)."""


# Every bound is a documented, evidenced-but-not-statistically-calibrated choice — the same
# status as this codebase's other deterministic thresholds (pipeline/types.py's coverage
# fractions, validation/validate.py's crop margin fraction). Each is grounded in that
# transform's actual geometric mechanism, not copied from a neighboring kind:
_TRANSFORM_GEOMETRY_PROFILES: dict[TransformKind, TransformGeometryProfile] = {
    # ROTATE swings every pixel inside the bbox rigidly around a pivot -- the real defect this
    # module exists to catch. A box big enough to be "most of the panel" WILL visibly rotate
    # panel furniture (borders, sound-effect text, background) along with the object, and a box
    # close to its reference region's edge has nowhere to swing into without clipping/revealing
    # background sharply (the real observed black-wedge artifacts). Tightest bounds of any
    # kind. `max_aspect_ratio` stays generous (12:1) because a legitimately elongated object
    # (a raised sword, a spear) is a common, real ROTATE target — the risk here is area and
    # edge proximity, not elongation.
    TransformKind.ROTATE: TransformGeometryProfile(
        max_area_fraction=0.15, min_edge_margin_fraction=0.05, max_aspect_ratio=12.0
    ),
    # SHEAR skews the bbox's rectangle into a parallelogram -- structurally the same "reveals a
    # wedge of background at the box's own corners" risk as ROTATE, slightly less violent for a
    # small shear angle, so slightly more permissive.
    TransformKind.SHEAR: TransformGeometryProfile(
        max_area_fraction=0.20, min_edge_margin_fraction=0.05, max_aspect_ratio=None
    ),
    # SCALE grows or shrinks the bbox's content around its pivot -- growing pushes content
    # beyond the box's own original footprint, so it also needs real clearance to do that
    # safely; shrinking doesn't, but this check can't know the sign in advance, so it stays
    # conservative for both directions.
    TransformKind.SCALE: TransformGeometryProfile(
        max_area_fraction=0.40, min_edge_margin_fraction=0.05, max_aspect_ratio=None
    ),
    # MESH_WARP deforms the region locally/continuously rather than rigidly sweeping it as one
    # unit -- real cloth/banner objects (this codebase's own flag/banner motion heuristic,
    # analysis/plan_builder.py's _MOTION_HEURISTICS) are legitimately large and elongated, so
    # this is deliberately looser than ROTATE/SHEAR on both area and aspect ratio.
    TransformKind.MESH_WARP: TransformGeometryProfile(
        max_area_fraction=0.35, min_edge_margin_fraction=0.02, max_aspect_ratio=None
    ),
    # TRANSLATE moves the bbox's content rigidly, but only by a small amplitude in this
    # codebase's real usage (amplitude = fraction of the panel diagonal; e.g. the hair
    # heuristic's amplitude=0.03, see _MOTION_HEURISTICS). No edge-margin requirement:
    # real evidence (Phase 3.3.1 remote re-verification, see ADR 0008's "Revision") showed an
    # initial 3% margin bound falsely rejected this project's own real, already-confirmed-
    # correct hair candidate (sample_page_01.png) -- hair naturally starts flush against a
    # portrait panel's top edge, which is normal composition, not a defect. Any real "hole"
    # a translate reveals at its trailing edge is already the hidden-region reconstruction
    # stage's job to fill (`reconstruction/`), not this check's -- unlike ROTATE/SHEAR/SCALE,
    # a small rigid shift has nowhere new to "swing into" that reconstruction doesn't already
    # handle, so area alone is the meaningful bound here.
    TransformKind.TRANSLATE: TransformGeometryProfile(
        max_area_fraction=0.50, min_edge_margin_fraction=0.0, max_aspect_ratio=None
    ),
    # OPACITY never moves a single pixel spatially -- it blends alpha in place, so it cannot
    # produce the "wrong pixels move" class of defect this module exists to catch. Deliberately
    # deferred back to the existing generic MAX_OBJECT_COVERAGE_FRACTION bound (no EXTRA
    # restriction) rather than inventing a number this transform's real mechanism doesn't need.
    TransformKind.OPACITY: TransformGeometryProfile(
        max_area_fraction=MAX_OBJECT_COVERAGE_FRACTION,
        min_edge_margin_fraction=0.0,
        max_aspect_ratio=None,
    ),
}


def _reference_region(panel_bbox_px: BBoxPx | None, image_shape: tuple[int, int]) -> BBoxPx:
    """The region bbox area/edge-margin are measured against: the object's real panel when

    known, else the full page — identical fallback behavior to how a page-level `AnimationPlan`
    already represents "no real panel" as a single `(0, 0, 1, 1)` `PanelPlan` elsewhere in this
    pipeline (see `pipeline/orchestrator.py::_panel_bbox_px`), so this isn't a new convention.
    """
    if panel_bbox_px is not None:
        return panel_bbox_px
    h, w = image_shape
    return BBoxPx(x0=0, y0=0, x1=w, y1=h)


def check_transform_geometry(
    bbox: BBoxPx,
    transform_kind: TransformKind,
    *,
    panel_bbox_px: BBoxPx | None,
    image_shape: tuple[int, int],
) -> tuple[bool, str]:
    """Deterministic, no-model-call check: is `bbox` geometrically safe to animate with

    `transform_kind`? Returns `(compatible, reason)` — `reason` is always a human-readable
    sentence, on both outcomes (never just on rejection), so an ACCEPT is equally explainable.

    Three checks, any one failing rejects the candidate (conservative — a candidate must pass
    all three): bbox area relative to its reference region (panel, else page), bbox proximity
    to that region's edges, and bbox aspect ratio where the transform kind makes that
    meaningful. See `_TRANSFORM_GEOMETRY_PROFILES` for the per-kind bounds and their rationale.
    """
    profile = _TRANSFORM_GEOMETRY_PROFILES[transform_kind]
    region = _reference_region(panel_bbox_px, image_shape)

    region_area = region.width * region.height
    bbox_area = bbox.width * bbox.height
    area_fraction = bbox_area / region_area if region_area else 1.0
    if area_fraction > profile.max_area_fraction:
        return False, (
            f"bbox covers {area_fraction:.1%} of its reference region, exceeding the "
            f"{profile.max_area_fraction:.0%} bound a {transform_kind.value} target allows -- "
            "too large to safely animate without moving pixels outside the intended object"
        )

    region_short_side = min(region.width, region.height)
    margins = (
        (bbox.x0 - region.x0) / region_short_side,
        (region.x1 - bbox.x1) / region_short_side,
        (bbox.y0 - region.y0) / region_short_side,
        (region.y1 - bbox.y1) / region_short_side,
    )
    min_margin = min(margins)
    if min_margin < profile.min_edge_margin_fraction:
        return False, (
            f"bbox sits within {min_margin:.1%} of its reference region's edge, closer than "
            f"the {profile.min_edge_margin_fraction:.0%} margin a {transform_kind.value} "
            "target needs to move without clipping against the boundary"
        )

    if profile.max_aspect_ratio is not None:
        aspect = max(bbox.width, bbox.height) / max(1, min(bbox.width, bbox.height))
        if aspect > profile.max_aspect_ratio:
            return False, (
                f"bbox aspect ratio {aspect:.1f}:1 exceeds the {profile.max_aspect_ratio:.0f}:1 "
                f"bound plausible for a {transform_kind.value} target"
            )

    return True, f"bbox geometry is compatible with a {transform_kind.value} transform"
