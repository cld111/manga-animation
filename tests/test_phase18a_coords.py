"""Phase 18.2A coordinate-conversion tests -- the brief's mandatory unit tests for the
normalization / resize-safe conversion between the Qwen coordinate output and source pixels.

These verify the exact measured contract in `coords.py`: Qwen2.5-VL reports SOURCE-PIXEL
coordinates (established on a real GPU smoke run: values > 1000 on a 1654-wide page can only be
source pixels), so conversion is identity up to an edge-tolerance clamp, and any response
outside that convention (a 0..1000-relative or normalized value, reversed/degenerate box,
garbage) is flagged -- never silently scaled or swapped.
"""

from __future__ import annotations

import pytest

from manga_animation.benchmarking.phase18a.coords import (
    EDGE_TOLERANCE_FRACTION,
    bbox_from_response,
    clamp_box,
    convert_prediction,
    extract_json_object,
    parse_direct_response,
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
    _, _, error = bbox_from_response({"found": True, "bbox": [1.0, 2.5, 3.0, 4]})
    assert error is not None  # 2.5 is not integral
    _, box2, _ = bbox_from_response({"found": True, "bbox": [1.0, 2.0, 3.0, 4.0]})
    assert box2 == (1, 2, 3, 4)


def test_clamp_box_identity_inside_bounds():
    assert clamp_box((100, 200, 300, 400), W, H) == (100, 200, 300, 400)


def test_clamp_box_small_overshoot():
    # y1 = 1174 on a 1170-tall page is a normal model overshoot (measured) -> clamped, kept.
    assert clamp_box((368, 435, 759, 1174), W, H) == (368, 435, 759, 1170)


def test_convert_prediction_negative_coord_fails_convention():
    # Negative coordinates violate the [0, dim] pixel convention at the conversion level.
    pred = convert_prediction("s1", '{"found": true, "bbox": [-5, 435, 759, 1170]}', W, H)
    assert pred.found and not pred.usable
    assert not pred.convention_ok
    assert pred.error is not None and "convention" in pred.error


def test_clamp_box_fully_outside_raises():
    with pytest.raises(ValueError):
        clamp_box((2000, 2000, 3000, 3000), W, H)


def test_convert_prediction_source_pixels_happy_path():
    raw = '{"found": true, "bbox": [368, 435, 759, 1174]}'
    pred = convert_prediction("s1", raw, W, H)
    assert pred.found and pred.usable and pred.error is None
    assert pred.pixel_box == (368, 435, 759, 1170)  # edge-tolerance clamp
    assert pred.convention_ok and pred.clamped


def test_convert_prediction_values_over_page_flag_failure():
    # x=1254 fits the 1654-wide page; but values beyond the 5% tolerance must fail.
    raw = '{"found": true, "bbox": [100, 100, 2000, 1200]}'
    pred = convert_prediction("s1", raw, W, H)
    assert pred.found and not pred.usable
    assert not pred.convention_ok
    assert pred.error is not None and "convention" in pred.error


def test_convert_prediction_normalized_1000_like_values_are_tiny_but_convention_ok():
    # A model returning a 0..1000-relative box would still look like source pixels on a
    # 1654-wide page -- the response is kept (raw text recorded for forensics), never silently
    # rescaled; only out-of-tolerance values are failures.
    pred = convert_prediction("s1", '{"found": true, "bbox": [100, 100, 800, 800]}', W, H)
    assert pred.found and pred.usable
    assert pred.pixel_box == (100, 100, 800, 800)


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


def test_convert_prediction_reversed_corners_fails():
    pred = convert_prediction("s1", '{"found": true, "bbox": [759, 1174, 368, 435]}', W, H)
    assert pred.found and not pred.usable
    assert pred.error is not None  # degenerate after clamp -> conversion failure, no silent swap


def test_as_dict_roundtrip_fields():
    pred = convert_prediction("s1", '{"found": true, "bbox": [100, 100, 500, 500]}', W, H)
    d = pred.as_dict()
    assert d["usable"] and d["pixel_box"] == [100, 100, 500, 500]
    assert d["box_raw"] == [100, 100, 500, 500]
    assert d["width"] == W and d["height"] == H


def test_edge_tolerance_constant_is_small():
    assert 0.0 < EDGE_TOLERANCE_FRACTION < 0.10
