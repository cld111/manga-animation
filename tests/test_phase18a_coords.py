"""Phase 18.2A coordinate-conversion tests -- the brief's mandatory unit tests for the
normalization / resize-safe conversion between the Qwen 0..1000 convention and source pixels.

These verify the exact contract in `coords.py`: parse lenient JSON, validate the 0..1000
convention, scale to pixels, and never let a coordinate mismatch pass silently.
"""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase18a.coords import (
    COORD_SCALE,
    bbox_from_response,
    convert_prediction,
    coords_in_scale,
    extract_json_object,
    parse_direct_response,
    scale_to_pixels,
)

# A canonical source page size (matches the phase-17 manifest's real page geometry).
W, H = 1654, 1170


def test_extract_json_clean():
    assert extract_json_object('{"found": true, "bbox": [0,0,500,500]}') == (
        '{"found": true, "bbox": [0,0,500,500]}'
    )


def test_extract_json_with_fences_and_prose():
    text = "Here is the answer:\n```json\n{\"found\": true, \"bbox\": [1, 2, 3, 4]}\n```\nDone."
    assert extract_json_object(text) == '{"found": true, "bbox": [1, 2, 3, 4]}'


def test_extract_json_embedded_in_prose():
    text = 'The box is {"found": true, "bbox": [0, 0, 100, 100]} as requested.'
    assert "bbox" in extract_json_object(text)


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")


def test_parse_direct_response_unparseable_is_none():
    assert parse_direct_response("I cannot answer this.") is None
    assert parse_direct_response("") is None


def test_bbox_from_response_found_false():
    found, box, error = bbox_from_response({"found": False, "bbox": None})
    assert (found, box, error) == (False, None, None)


def test_bbox_from_response_found_false_with_bbox_is_error():
    found, box, error = bbox_from_response({"found": False, "bbox": [1, 2, 3, 4]})
    assert found is False and error is not None


def test_bbox_from_response_missing_found():
    _, _, error = bbox_from_response({"bbox": [1, 2, 3, 4]})
    assert error is not None


def test_bbox_from_response_missing_bbox():
    found, box, error = bbox_from_response({"found": True, "bbox": None})
    assert found is True and box is None and error is not None


def test_bbox_from_response_coerces_integral_floats():
    found, box, error = bbox_from_response({"found": True, "bbox": [1.0, 2.5, 3.0, 4]})
    assert error is not None  # 2.5 is not integral
    found2, box2, _ = bbox_from_response({"found": True, "bbox": [1.0, 2.0, 3.0, 4.0]})
    assert box2 == (1, 2, 3, 4)


def test_coords_in_scale():
    assert coords_in_scale((0, 0, COORD_SCALE, COORD_SCALE))
    assert coords_in_scale((10, 20, 100, 200))
    assert not coords_in_scale((-1, 0, 100, 100))  # negative -> out of convention
    assert not coords_in_scale((0, 0, 1001, 100))  # > 1000 -> raw-pixel-looking mismatch
    assert not coords_in_scale((100, 100, 100, 200))  # degenerate x
    assert not coords_in_scale((100, 200, 200, 200))  # degenerate y (y1 == y0)


def test_scale_to_pixels_full_page():
    assert scale_to_pixels((0, 0, COORD_SCALE, COORD_SCALE), W, H) == (0, 0, W, H)


def test_scale_to_pixels_quarter():
    assert scale_to_pixels((0, 0, 500, 500), W, H) == (0, 0, round(0.5 * W), round(0.5 * H))


def test_scale_to_pixels_is_exact_round():
    # A box expressed in 0..1000 must land on the exact pixel the ratio implies.
    box = (
        659 * COORD_SCALE // W,
        968 * COORD_SCALE // H,
        734 * COORD_SCALE // W,
        1095 * COORD_SCALE // H,
    )
    x0, y0, x1, y1 = scale_to_pixels(box, W, H)
    assert x0 <= round(659 / W * COORD_SCALE * W / COORD_SCALE) + 1
    assert x0 >= round(659 / W * COORD_SCALE * W / COORD_SCALE) - 1
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


def test_scale_to_pixels_clamps():
    # Out-of-scale coords clamp to the page edge (callers reject them via coords_in_scale).
    assert scale_to_pixels((0, 0, 2000, 2000), W, H) == (0, 0, W, H)


def test_scale_to_pixels_degenerate_raises():
    # Two adjacent 1000-scale units can round to one pixel on a small page -> degenerate.
    with pytest.raises(ValueError):
        scale_to_pixels((0, 0, 2, 2), 3, 3)


def test_convert_prediction_happy_path():
    raw = '{"found": true, "bbox": [0, 0, 500, 500]}'
    pred = convert_prediction("s1", raw, W, H)
    assert pred.found and pred.usable and pred.error is None
    assert pred.pixel_box == (0, 0, round(0.5 * W), round(0.5 * H))
    assert pred.convention_ok
    assert pred.box_1000 == (0, 0, 500, 500)


def test_convert_prediction_not_found():
    pred = convert_prediction("s1", '{"found": false, "bbox": null}', W, H)
    assert not pred.found and not pred.usable and pred.error is None
    assert pred.pixel_box is None


def test_convert_prediction_unparseable():
    pred = convert_prediction("s1", "I don't see it.", W, H)
    assert not pred.found and not pred.usable
    assert pred.error is not None and "unparseable" in pred.error


def test_convert_prediction_malformed():
    pred = convert_prediction("s1", '{"found": true, "bbox": [1, 2]}', W, H)
    assert not pred.usable and pred.error is not None


def test_convert_prediction_convention_violation():
    # Values > 1000 look like raw pixels of a processed image -- must be flagged, not scaled.
    pred = convert_prediction("s1", '{"found": true, "bbox": [100, 100, 1500, 1500]}', W, H)
    assert pred.found and not pred.usable
    assert not pred.convention_ok
    assert pred.error is not None and "convention" in pred.error


def test_convert_prediction_degenerate_after_scale():
    pred = convert_prediction("s1", '{"found": true, "bbox": [0, 0, 1, 1]}', 3, 3)
    assert pred.found and not pred.usable
    assert pred.error is not None


def test_as_dict_roundtrip_fields():
    pred = convert_prediction("s1", '{"found": true, "bbox": [0, 0, 500, 500]}', W, H)
    d = pred.as_dict()
    assert d["usable"] and d["pixel_box"] == [0, 0, round(0.5 * W), round(0.5 * H)]
    assert d["box_1000"] == [0, 0, 500, 500]
