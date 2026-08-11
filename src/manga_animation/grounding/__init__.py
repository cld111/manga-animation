"""Object grounding — mapping semantic labels from the Animation Plan to image regions."""

from manga_animation.grounding.client import Detection, GroundingClient, GroundingDinoClient
from manga_animation.grounding.ground import ground_object

__all__ = [
    "Detection",
    "GroundingClient",
    "GroundingDinoClient",
    "ground_object",
]
