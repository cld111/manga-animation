"""Phase 18.2A classification tests: the mechanical error-category mapping (brief items 1-9)."""

from __future__ import annotations

from manga_animation.benchmarking.phase18a.classify import ERROR_CATEGORY_NAMES, classify

# GT: a 200x200 box at (100, 100)-(300, 300) on a 1000x1000 page.
GT = (100, 100, 300, 300)
PAGE = (1000, 1000)


def test_good_bbox():
    c = classify(GT, (100, 100, 300, 300), True, None, *PAGE)
    assert c.name == "good" and c.category == 0


def test_imprecise_but_correct():
    # Loose box (80,80,320,320): IoU = 40000/57600 = 0.694 -> category 1.
    c = classify(GT, (80, 80, 320, 320), True, None, *PAGE)
    assert c.name == "correct_object_imprecise_bbox" and c.category == 1


def test_wrong_instance():
    c = classify(GT, (700, 700, 900, 900), True, None, *PAGE)
    assert c.name == "wrong_instance" and c.category == 2


def test_bbox_too_large():
    # 400x400 around the target: IoU 0.25, area ratio 4x -> too large (3).
    c = classify(GT, (100, 100, 500, 500), True, None, *PAGE)
    assert c.name == "bbox_too_large" and c.category == 3


def test_bbox_too_small():
    # Tiny box inside the GT: low coverage, small ratio -> too small (4).
    c = classify(GT, (150, 150, 190, 190), True, None, *PAGE)
    assert c.name == "bbox_too_small" and c.category == 4


def test_page_grab():
    c = classify(GT, (0, 0, 900, 900), True, None, *PAGE)
    assert c.name == "target_outside_panel_or_page_grab" and c.category == 7


def test_not_found():
    c = classify(GT, None, False, None, *PAGE)
    assert c.name == "not_found" and c.category == 8


def test_conversion_failure_takes_precedence():
    c = classify(GT, None, True, "coordinates outside the 0..1000 convention", *PAGE)
    assert c.name == "coordinate_conversion_failure" and c.category == 9


def test_names_cover_brief_taxonomy():
    for cat in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        assert cat in ERROR_CATEGORY_NAMES
    assert ERROR_CATEGORY_NAMES[5] == "partially_hidden_object"
    assert ERROR_CATEGORY_NAMES[6] == "multiple_similar_objects"
