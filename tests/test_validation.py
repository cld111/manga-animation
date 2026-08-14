from __future__ import annotations

import json

import numpy as np
import pytest

from manga_animation.pipeline.types import BBoxPx, GroundingResult
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PivotSpec,
    TransformKind,
    Vector2,
)
from manga_animation.validation.validate import validate_target


class FakeVLMClient:
    """A `VLMClient` double for the validation stage: returns a canned response string,

    records every (image, prompt) call it received.
    """

    def __init__(self, response: str):
        self._response = response
        self.prompts: list[str] = []
        self.images: list = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        self.images.append(image)
        return self._response


def _verification_json(matches: bool, confidence: float = 0.9, reason: str = "fake reason") -> str:
    return json.dumps({"matches": matches, "confidence": confidence, "reason": reason})


def make_image(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_object_plan(
    semantic_label: str = "flag_banner", transform_kind: TransformKind = TransformKind.MESH_WARP
) -> ObjectPlan:
    return ObjectPlan(
        object_id="obj_1",
        panel_id="panel_1",
        semantic_label=semantic_label,
        confidence=0.8,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=transform_kind,
            direction=Vector2(x=1.0, y=0.0) if transform_kind == TransformKind.TRANSLATE else None,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    )


def make_grounding(bbox: BBoxPx | None = None, score: float = 0.5) -> GroundingResult:
    resolved = bbox or BBoxPx(x0=50, y0=50, x1=100, y1=100, score=score)
    return GroundingResult(object_id="obj_1", bbox=resolved, model_id="fake-grounding")


# --- accept / reject on semantic agreement ------------------------------------------------


def test_validate_target_accepts_when_bbox_plausible_and_vlm_agrees():
    client = FakeVLMClient(_verification_json(True, confidence=0.92, reason="clearly a banner"))
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is True
    assert result.semantic_match is True
    assert result.semantic_confidence == pytest.approx(0.92)
    assert result.bbox_plausible is True
    assert result.reason == "clearly a banner"


def test_validate_target_rejects_when_vlm_disagrees_even_at_high_grounding_score():
    """Real Phase 3.1 finding this stage exists to catch: a technically valid, high-scoring

    detection is not automatically correct.
    """
    client = FakeVLMClient(_verification_json(False, confidence=0.85, reason="this is a face"))
    high_score_grounding = make_grounding(score=0.9)

    result = validate_target(make_image(), make_object_plan(), high_score_grounding, client)

    assert result.accepted is False
    assert result.semantic_match is False
    assert result.grounding_score == pytest.approx(0.9)
    assert "face" in result.reason


# --- bbox plausibility pre-filter (no model call needed) ----------------------------------


def test_validate_target_rejects_implausible_bbox_without_calling_the_vlm():
    client = FakeVLMClient(_verification_json(True))
    # ~99% of a 200x200 image -- a real grounded object practically never covers this much
    huge_box = BBoxPx(x0=0, y0=0, x1=199, y1=199, score=0.5)

    result = validate_target(make_image(), make_object_plan(), make_grounding(huge_box), client)

    assert result.accepted is False
    assert result.bbox_plausible is False
    assert result.semantic_match is None
    assert result.semantic_confidence is None
    assert client.prompts == []  # never spent a VLM call on an already-implausible box


def test_validate_target_accepts_a_small_plausible_bbox_when_vlm_agrees():
    client = FakeVLMClient(_verification_json(True))
    small_box = BBoxPx(x0=90, y0=90, x1=110, y1=110, score=0.4)  # 1% of a 200x200 image

    result = validate_target(make_image(), make_object_plan(), make_grounding(small_box), client)

    assert result.accepted is True
    assert result.bbox_plausible is True


# --- fail-closed on an unparseable VLM response --------------------------------------------


def test_validate_target_rejects_unparseable_vlm_response():
    client = FakeVLMClient("this is not json at all {{{")
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is False
    assert result.semantic_match is None
    assert result.bbox_plausible is True  # got past the pre-filter; the VLM call is what failed


def test_validate_target_accepts_json_wrapped_in_markdown_fences():
    wrapped = f"Sure, here you go:\n```json\n{_verification_json(True)}\n```\n"
    client = FakeVLMClient(wrapped)
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)

    assert result.accepted is True


# --- crop construction ----------------------------------------------------------------------


def test_validate_target_crops_with_margin_around_the_bbox():
    client = FakeVLMClient(_verification_json(True))
    box = BBoxPx(x0=50, y0=50, x1=100, y1=100, score=0.5)

    validate_target(make_image(), make_object_plan(), make_grounding(box), client)

    seen_w, seen_h = client.images[0].size  # PIL Image.size == (width, height)
    assert seen_w > box.x1 - box.x0
    assert seen_h > box.y1 - box.y0


# --- diagnostics -----------------------------------------------------------------------------


def test_validate_target_records_the_candidate_rank_it_was_given():
    client = FakeVLMClient(_verification_json(True))
    result = validate_target(
        make_image(), make_object_plan(), make_grounding(), client, candidate_rank=2
    )
    assert result.candidate_rank == 2


def test_validate_target_never_raises_on_reject():
    """A REJECT is a normal outcome, not an exception -- see the Phase 3.2 acceptance

    criterion ("a correct REJECT is a successful result").
    """
    client = FakeVLMClient(_verification_json(False))
    result = validate_target(make_image(), make_object_plan(), make_grounding(), client)
    assert result.accepted is False  # returned, not raised


# --- Phase 3.3.1: transform-aware geometric validation --------------------------------------
#
# Real motivating defect (Phase 3.3 real E2E run, eval_weapon_effects.png): a candidate scored
# semantically correct ("yes, this crop shows a weapon") but its bbox covered nearly the whole
# action panel; the plan's `rotate` transform then visibly swung the whole panel, not the
# weapon. See validation/transform_geometry.py and
# docs/decisions/0008-transform-aware-target-validation.md.


def test_validate_target_accepts_a_geometrically_valid_rotate_candidate():
    """1. Semantically correct + geometrically valid (small, well-clear-of-the-edge) candidate

    -> ACCEPT, with transform_compatible recorded True.
    """
    client = FakeVLMClient(_verification_json(True, reason="a real sword"))
    small_centered_box = BBoxPx(x0=80, y0=80, x1=120, y1=120, score=0.5)  # 4% of a 200x200 image
    plan = make_object_plan("sword", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(small_centered_box), client)

    assert result.accepted is True
    assert result.semantic_match is True
    assert result.transform_compatible is True


def test_validate_target_rejects_oversized_bbox_for_rotate():
    """2. The real Phase 3.3 E2E defect, reproduced directly: a semantically-correct candidate

    whose bbox is too large for ROTATE must be REJECTed, not animated. The semantic check must
    still have run (and passed) -- this is a geometry rejection, not a semantic one.
    """
    client = FakeVLMClient(_verification_json(True, reason="a real weapon"))
    # 81% of a 200x200 image (180x180) -- comfortably clears the generic <=90% bbox-plausibility
    # bound (so this reaches the VLM/geometry checks at all) but far exceeds ROTATE's 15% bound
    # -- the same shape as the real defect (grounding's box covered nearly the whole panel).
    huge_box = BBoxPx(x0=10, y0=10, x1=190, y1=190, score=0.255)
    plan = make_object_plan("weapon", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(huge_box), client)

    assert result.accepted is False
    assert result.semantic_match is True  # semantic check DID pass -- geometry is what failed
    assert result.transform_compatible is False
    assert "rotate" in result.reason.lower()
    assert client.prompts  # the VLM call happened -- semantic check ran before geometry


def test_validate_target_rejects_boundary_risk_bbox_for_rotate():
    """3. A bbox small enough by area but flush against the image edge -- no room to safely

    rotate without clipping/revealing background sharply at the swing's far side.
    """
    client = FakeVLMClient(_verification_json(True, reason="a real weapon"))
    # 30x30 box (2.25% of a 200x200 image -- comfortably under ROTATE's 15% area bound) but
    # flush against the top-left corner -- zero clearance to swing into.
    edge_box = BBoxPx(x0=0, y0=0, x1=30, y1=30, score=0.5)
    plan = make_object_plan("weapon", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(edge_box), client)

    assert result.accepted is False
    assert result.semantic_match is True
    assert result.transform_compatible is False
    assert any(word in result.reason.lower() for word in ("edge", "margin", "boundary", "clip"))


def test_validate_target_rejects_extreme_aspect_ratio_for_rotate():
    """A pathologically thin sliver is implausible even for a legitimately elongated ROTATE

    target (a sword/spear) -- ROTATE's max_aspect_ratio bound exists for this degenerate case.
    """
    client = FakeVLMClient(_verification_json(True, reason="a real weapon"))
    sliver = BBoxPx(x0=95, y0=20, x1=105, y1=180, score=0.5)  # 10px wide, 160px tall: 16:1
    plan = make_object_plan("weapon", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(sliver), client)

    assert result.accepted is False
    assert result.transform_compatible is False
    assert "aspect" in result.reason.lower()


def test_geometry_check_is_transform_specific():
    """4. The same oversized bbox must be evaluated against its OWN transform_kind's profile,

    not a generic one -- verified directly against check_transform_geometry.
    """
    from manga_animation.validation.transform_geometry import check_transform_geometry

    bbox = BBoxPx(x0=10, y0=10, x1=190, y1=190)  # 81% of a 200x200 region
    compatible, reason = check_transform_geometry(
        bbox, TransformKind.ROTATE, panel_bbox_px=None, image_shape=(200, 200)
    )
    assert compatible is False
    assert "rotate" in reason.lower()


def test_different_transform_kinds_have_different_geometry_bounds():
    """5. The identical bbox must be evaluated differently depending on transform_kind -- no

    single universal area/margin threshold governs every transform (see the Phase 3.3.1
    brief's explicit "do not use one universal threshold" instruction).
    """
    from manga_animation.validation.transform_geometry import check_transform_geometry

    # 39% of a 200x200 region -- fails ROTATE's 15% bound, passes TRANSLATE's 50% bound.
    bbox = BBoxPx(x0=20, y0=20, x1=145, y1=145)
    rotate_ok, _ = check_transform_geometry(
        bbox, TransformKind.ROTATE, panel_bbox_px=None, image_shape=(200, 200)
    )
    translate_ok, _ = check_transform_geometry(
        bbox, TransformKind.TRANSLATE, panel_bbox_px=None, image_shape=(200, 200)
    )
    assert rotate_ok is False
    assert translate_ok is True


def test_radial_expand_has_a_transform_specific_geometry_profile():
    """Phase 16: RADIAL_EXPAND (drawn-effect pulse) must have its OWN registered geometry
    profile -- no KeyError, and the same bbox is evaluated with the radial model's looser
    MESH_WARP-like bounds rather than ROTATE's strict ones."""
    from manga_animation.validation.transform_geometry import check_transform_geometry

    # 25% of a 200x200 region: fails ROTATE's 15% bound, passes RADIAL_EXPAND's 35% bound.
    bbox = BBoxPx(x0=10, y0=10, x1=110, y1=110)
    rotate_ok, _ = check_transform_geometry(
        bbox, TransformKind.ROTATE, panel_bbox_px=None, image_shape=(200, 200)
    )
    radial_ok, radial_reason = check_transform_geometry(
        bbox, TransformKind.RADIAL_EXPAND, panel_bbox_px=None, image_shape=(200, 200)
    )
    assert rotate_ok is False
    assert radial_ok is True
    assert "radial_expand" in radial_reason.lower()


def test_translate_accepts_a_bbox_flush_against_the_reference_edge():
    """Real regression guard: a TRANSLATE candidate flush against its reference region's top

    edge (0% margin) must still be ACCEPTed -- real evidence (Phase 3.3.1 remote
    re-verification against this project's own already-confirmed-correct hair candidate,
    sample_page_01.png) showed an earlier version of this check applied a nonzero edge-margin
    requirement to TRANSLATE and falsely rejected it: hair naturally starts flush against a
    portrait panel's top edge, which is normal composition, not a geometric defect. See
    docs/decisions/0008-transform-aware-target-validation.md's "Revision".
    """
    from manga_animation.validation.transform_geometry import check_transform_geometry

    # Flush against the top edge (y0=0), small enough by area to be a real object, not a
    # boundary risk for a rigid, small-amplitude shift.
    hairline_box = BBoxPx(x0=60, y0=0, x1=140, y1=90)  # 12.5% of a 200x200 region

    compatible, reason = check_transform_geometry(
        hairline_box, TransformKind.TRANSLATE, panel_bbox_px=None, image_shape=(200, 200)
    )

    assert compatible is True, reason


def test_geometry_check_uses_the_panel_as_its_reference_region_when_known():
    """A bbox that's a small fraction of the full PAGE can still be a large, unsafe fraction of

    its own PANEL -- the reference region must be the real panel when panel-aware analysis
    provided one (see validate_target's `panel_bbox_px` parameter): the SAME bbox must be able
    to pass against the page and fail against its own (smaller) panel.
    """
    from manga_animation.validation.transform_geometry import check_transform_geometry

    # A small panel positioned well away from the page's own edges, so only the AREA
    # comparison differs between the two reference regions (not edge-margin too).
    small_panel = BBoxPx(x0=400, y0=400, x1=500, y1=500)  # 100x100, panel area 10_000px^2
    bbox = BBoxPx(x0=410, y0=410, x1=490, y1=490)  # 80x80 = 6400px^2, well inside the panel

    ok_against_page, _ = check_transform_geometry(
        bbox, TransformKind.ROTATE, panel_bbox_px=None, image_shape=(1000, 1000)
    )  # 6400 / 1_000_000 = 0.64% of the page, huge margins -- passes ROTATE's bounds easily
    ok_against_panel, _ = check_transform_geometry(
        bbox, TransformKind.ROTATE, panel_bbox_px=small_panel, image_shape=(1000, 1000)
    )  # 6400 / 10_000 = 64% of the panel -- fails ROTATE's 15% area bound

    assert ok_against_page is True
    assert ok_against_panel is False


def test_semantic_rejection_happens_before_geometry_check_runs():
    """6. A candidate that fails the SEMANTIC check must be rejected for that reason -- the

    geometry check must never even run (and never overwrite the real rejection reason), even
    when the bbox is ALSO geometrically bad for the plan's transform_kind.
    """
    client = FakeVLMClient(_verification_json(False, reason="this is a face, not a weapon"))
    huge_box = BBoxPx(x0=10, y0=10, x1=190, y1=190, score=0.9)  # also geometrically bad for rotate
    plan = make_object_plan("weapon", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(huge_box), client)

    assert result.accepted is False
    assert result.semantic_match is False
    assert result.transform_compatible is None  # geometry check never reached
    assert "face" in result.reason  # the real (semantic) rejection reason, not a geometry one


def test_transform_compatible_stays_none_when_bbox_implausibility_short_circuits():
    """The geometry check must not run (and `transform_compatible` must stay `None`) when the

    cheap generic bbox-plausibility pre-filter already rejected the candidate -- consistent
    with the existing "cheapest first" short-circuit for the semantic check.
    """
    client = FakeVLMClient(_verification_json(True))
    huge_box = BBoxPx(x0=0, y0=0, x1=199, y1=199, score=0.5)  # ~99% -- fails the generic bound
    plan = make_object_plan("weapon", TransformKind.ROTATE)

    result = validate_target(make_image(), plan, make_grounding(huge_box), client)

    assert result.accepted is False
    assert result.bbox_plausible is False
    assert result.transform_compatible is None
    assert client.prompts == []
