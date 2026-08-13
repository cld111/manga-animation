from __future__ import annotations

import json

import numpy as np
import pytest

from manga_animation.pipeline.types import BBoxPx
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    PivotSpec,
    TransformKind,
)
from manga_animation.validation.mask_semantics import verify_mask_semantics


class FakeVLMClient:
    """A `VLMClient` double for the mask_semantics stage: returns a canned response string,

    records every (image, prompt) call it received -- same style as
    `tests/test_validation.py::FakeVLMClient`.
    """

    def __init__(self, response: str):
        self._response = response
        self.prompts: list[str] = []
        self.images: list = []

    def generate(self, image, prompt: str) -> str:
        self.prompts.append(prompt)
        self.images.append(image)
        return self._response


def _mask_response(
    matches: bool,
    confidence: float = 0.9,
    reason: str = "fake reason",
    unexpected_content: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "mask_matches_object": matches,
            "confidence": confidence,
            "unexpected_content": unexpected_content or [],
            "reason": reason,
        }
    )


def make_image(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def make_object_plan(semantic_label: str = "cloth") -> ObjectPlan:
    return ObjectPlan(
        object_id="obj_1",
        panel_id="panel_1",
        semantic_label=semantic_label,
        confidence=0.8,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind.MESH_WARP,
            amplitude=0.1,
            speed=1.0,
            easing=Easing.SINE,
            pivot=PivotSpec(x=0.5, y=0.0, reference="object_bbox"),
        ),
    )


def make_diamond_mask(h: int, w: int, bbox: BBoxPx) -> np.ndarray:
    """A plausible SAM-shaped mask (diamond inscribed in `bbox`) -- same construction

    `tests/test_pipeline.py::_region_mask` uses, so geometric-signal computation exercises a
    realistic silhouette rather than a degenerate solid rectangle.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (bbox.x0 + bbox.x1 - 1) / 2.0, (bbox.y0 + bbox.y1 - 1) / 2.0
    ax, ay = max((bbox.x1 - bbox.x0) / 2.0, 1e-9), max((bbox.y1 - bbox.y0) / 2.0, 1e-9)
    yy, xx = np.mgrid[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    local = (np.abs(xx - cx) / ax + np.abs(yy - cy) / ay) <= 1.0
    mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1] = local.astype(np.uint8) * 255
    return mask


# --- accept / reject on the VLM's mask-content read -----------------------------------------


def test_verify_mask_semantics_accepts_a_confident_match():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(True, confidence=0.95, reason="clearly just cloth"))

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "accept"
    assert result.accepted is True
    assert result.vlm_matches is True
    assert result.vlm_confidence == pytest.approx(0.95)
    assert result.reason == "clearly just cloth"
    assert result.method == "vlm_mask_crop_v1"


def test_verify_mask_semantics_rejects_the_real_phase11_cloth_defect_shape():
    """Real Phase 11 defect this stage exists to catch (docs/phase11-results.md section 6.4):

    a "cloth" mask whose real content also included a full speech bubble and a hand. The VLM
    read is faked here (no live model in a unit test), but the resulting REJECT + populated
    `unexpected_content` is exactly the structured record that real defect needs.
    """
    bbox = BBoxPx(x0=18, y0=958, x1=470, y1=1643)
    mask = make_diamond_mask(2000, 600, bbox)
    client = FakeVLMClient(
        _mask_response(
            False,
            confidence=0.88,
            reason="the highlighted region includes a speech bubble and a hand, not just cloth",
            unexpected_content=["speech bubble", "hand"],
        )
    )

    result = verify_mask_semantics(
        make_image(2000, 600), make_object_plan("cloth"), mask, bbox, client
    )

    assert result.verdict == "reject"
    assert result.accepted is False
    assert result.vlm_matches is False
    assert result.unexpected_content == ("speech bubble", "hand")
    assert "speech bubble" in result.reason


# --- ABSTAIN on low-confidence evidence -------------------------------------------------------


def test_verify_mask_semantics_abstains_on_near_coin_flip_confidence():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(True, confidence=0.5, reason="genuinely unclear"))

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "abstain"
    assert result.accepted is False
    assert "coin-flip" in result.reason or "0.5" in result.reason


def test_verify_mask_semantics_does_not_abstain_just_outside_the_band():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(True, confidence=0.61, reason="fairly confident"))

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "accept"


# --- fail-closed on an unparseable VLM response ------------------------------------------------


def test_verify_mask_semantics_rejects_unparseable_vlm_response():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient("not json at all {{{")

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "reject"
    assert result.vlm_matches is None
    assert result.vlm_confidence is None


def test_verify_mask_semantics_accepts_json_wrapped_in_markdown_fences():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    wrapped = f"Sure:\n```json\n{_mask_response(True)}\n```\n"
    client = FakeVLMClient(wrapped)

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "accept"


def test_verify_mask_semantics_never_raises():
    """A REJECT/ABSTAIN is a normal outcome, never an exception -- mirrors

    test_validation.py::test_validate_target_never_raises_on_reject.
    """
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(False))
    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)
    assert result.verdict == "reject"  # returned, not raised


# --- crop construction: mask overlay dims everything outside the mask -------------------------


def test_verify_mask_semantics_dims_pixels_outside_the_mask_in_the_sent_crop():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)  # a diamond -- its own bbox corners are OUTSIDE it
    client = FakeVLMClient(_mask_response(True))

    verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    sent = np.asarray(client.images[0])
    # The bbox's inscribed diamond touches the midpoint of each edge but not the corners --
    # a corner pixel of the (margin-padded) crop must be dimmed relative to the flat source
    # image value (128), while the crop's own center (inside the mask) must not be.
    corner_value = sent[0, 0].astype(int).mean()
    center_value = sent[sent.shape[0] // 2, sent.shape[1] // 2].astype(int).mean()
    assert corner_value < 128
    assert center_value == pytest.approx(128, abs=1)


def test_verify_mask_semantics_crops_with_margin_around_the_bbox():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(True))

    verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    seen_w, seen_h = client.images[0].size  # PIL Image.size == (width, height)
    assert seen_w > bbox.x1 - bbox.x0
    assert seen_h > bbox.y1 - bbox.y0


# --- geometric signals: computed and attached, but never gating -------------------------------


def test_verify_mask_semantics_attaches_geometric_signals_regardless_of_verdict():
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)

    accept_client = FakeVLMClient(_mask_response(True))
    reject_client = FakeVLMClient(_mask_response(False))

    accepted = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, accept_client)
    rejected = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, reject_client)

    for result in (accepted, rejected):
        assert set(result.geometric_signals) == {
            "second_component_area_fraction",
            "bbox_density",
            "aspect_ratio",
            "convex_hull_solidity",
        }
        # A single, real, connected diamond -- no meaningful second component.
        assert result.geometric_signals["second_component_area_fraction"] == pytest.approx(0.0)


def test_geometric_signals_do_not_change_the_verdict():
    """The real Phase 11 finding this module's design is built on: geometric signals alone

    (an obviously "unremarkable" diamond here) must never override what the VLM actually says
    about the mask's content -- a geometrically pristine mask can still be semantically wrong.
    """
    bbox = BBoxPx(x0=50, y0=50, x1=100, y1=100)
    mask = make_diamond_mask(200, 200, bbox)
    client = FakeVLMClient(_mask_response(False, reason="wrong object entirely"))

    result = verify_mask_semantics(make_image(), make_object_plan(), mask, bbox, client)

    assert result.verdict == "reject"  # geometry looked fine; the VLM read is what mattered
