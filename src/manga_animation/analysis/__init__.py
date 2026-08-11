"""Panel/scene analysis and VLM-based semantic understanding of manga pages.

Turns a real manga page into a schema-valid `AnimationPlan` (see
`docs/animation-plan-schema.md`) via a vision-language model. `client.VLMClient` is the seam
that keeps `torch`/`transformers` out of this package's import graph except inside the real
client's methods (see `client.py`'s docstring) -- `plan_builder` is fully unit-testable with a
fake client.
"""

from manga_animation.analysis.client import Qwen25VLClient, VLMClient
from manga_animation.analysis.plan_builder import ANALYSIS_PROMPT, analyze_page, build_plan

__all__ = [
    "ANALYSIS_PROMPT",
    "Qwen25VLClient",
    "VLMClient",
    "analyze_page",
    "build_plan",
]
