"""Per-candidate VLM object description stage (Phase 18.3).

The VLM sees the full pipeline image plus a candidate bounding box given as pixel
coordinates, judges the candidate itself, and produces a structured animation description
that the animation stage actually applies. See `describe.py` for the stage contract and
`docs/pipeline.md` for its place in the pipeline.
"""

from manga_animation.object_description.describe import (
    METHOD_ID,
    PROMPT_MARKER,
    describe_object,
)
from manga_animation.object_description.mapping import motion_spec_from_description
from manga_animation.object_description.prompt import (
    PreparedVlmInput,
    build_prompt,
    prepare_image_and_bbox,
)
from manga_animation.object_description.schema import (
    AmplitudeBand,
    BBoxAssessment,
    DirectionWord,
    MotionKind,
    ObjectDescriptionResponse,
    PivotHint,
    SpeedBand,
)

__all__ = [
    "AmplitudeBand",
    "BBoxAssessment",
    "DirectionWord",
    "METHOD_ID",
    "MotionKind",
    "ObjectDescriptionResponse",
    "PROMPT_MARKER",
    "PreparedVlmInput",
    "PivotHint",
    "SpeedBand",
    "build_prompt",
    "describe_object",
    "motion_spec_from_description",
    "prepare_image_and_bbox",
]
