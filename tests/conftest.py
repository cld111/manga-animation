from __future__ import annotations

import pytest

from manga_animation.schemas import AnimationPlan
from manga_animation.schemas.animation_plan import PanelPlan, SourceImage


def make_plan(**overrides) -> AnimationPlan:
    """A minimal valid AnimationPlan, for tests to extend via overrides."""
    defaults: dict = dict(
        source=SourceImage(path="examples/page_001.png", width=1600, height=2400),
        panels=[
            PanelPlan(panel_id="panel_1", bbox={"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0})
        ],
    )
    defaults.update(overrides)
    return AnimationPlan(**defaults)


@pytest.fixture
def base_plan_kwargs() -> dict:
    return dict(
        source=SourceImage(path="examples/page_001.png", width=1600, height=2400),
        panels=[
            PanelPlan(panel_id="panel_1", bbox={"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0})
        ],
    )
