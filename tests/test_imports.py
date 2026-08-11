from __future__ import annotations

import importlib

import pytest


def test_top_level_package_imports_and_has_version():
    import manga_animation

    assert manga_animation.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module_name",
    [
        "manga_animation.core",
        "manga_animation.core.config",
        "manga_animation.core.logging",
        "manga_animation.core.seed",
        "manga_animation.schemas",
        "manga_animation.schemas.animation_plan",
        "manga_animation.analysis",
        "manga_animation.grounding",
        "manga_animation.segmentation",
        "manga_animation.layers",
        "manga_animation.reconstruction",
        "manga_animation.animation",
        "manga_animation.compositing",
        "manga_animation.rendering",
        "manga_animation.pipeline",
    ],
)
def test_all_stage_packages_are_importable(module_name):
    importlib.import_module(module_name)
