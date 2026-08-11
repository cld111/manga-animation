"""Cross-cutting foundations: configuration and logging."""

from manga_animation.core.config import PipelineConfig, load_config
from manga_animation.core.logging import StageTimer, get_gpu_memory_mb, get_logger, setup_logging
from manga_animation.core.seed import set_global_seed

__all__ = [
    "PipelineConfig",
    "StageTimer",
    "get_gpu_memory_mb",
    "get_logger",
    "load_config",
    "set_global_seed",
    "setup_logging",
]
