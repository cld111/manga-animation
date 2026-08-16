"""Per-candidate VLM object description stage (Phase 18.3).

The VLM sees the full pipeline image plus a candidate bounding box given as pixel
coordinates, judges the candidate itself, and produces a structured animation description
that the animation stage actually applies. See `describe.py` for the stage contract and
`docs/pipeline.md` for its place in the pipeline.
"""

from manga_animation.object_description.describe import (
    METHOD_ID,
    PROMPT_MARKER,
    CandidateBox,
    describe_object,
    describe_objects,
)
from manga_animation.object_description.mapping import motion_spec_from_description
from manga_animation.object_description.prompt import (
    PreparedVlmInput,
    PromptCandidate,
    build_multi_prompt,
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
    "CandidateBox",
    "DirectionWord",
    "METHOD_ID",
    "MotionKind",
    "ObjectDescriptionResponse",
    "PROMPT_MARKER",
    "PreparedVlmInput",
    "PromptCandidate",
    "PivotHint",
    "SpeedBand",
    "build_multi_prompt",
    "build_prompt",
    "describe_object",
    "describe_objects",
    "motion_spec_from_description",
    "prepare_image_and_bbox",
]
