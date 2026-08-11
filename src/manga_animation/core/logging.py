"""Practical, structured-enough logging for pipeline stages.

Deliberately just stdlib `logging` plus a small context manager — see
docs/architecture.md. Every pipeline stage (once implemented) should wrap its work in
`StageTimer` so stage name, timing, device, model, shapes, confidence, and GPU memory are
reported consistently without every stage hand-rolling its own logging calls.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(debug: bool = False, *, stream: Any = None) -> None:
    """Configure root logging once, e.g. at process/script entry point."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=stream or sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_gpu_memory_mb() -> float | None:
    """Currently allocated GPU/accelerator memory in MB, or None if unavailable.

    Never raises: torch may not be installed (Phase 1), or no accelerator may be present
    (local CPU-only, or a CUDA-only code path running on MPS).
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.mps.current_allocated_memory() / (1024**2)
    except Exception:
        return None
    return None


@dataclass
class StageTimer:
    """Context manager that logs a pipeline stage's start, timing, and outcome.

    Usage:
        with StageTimer("segmentation", logger, device="mps", model="sam2-base") as t:
            masks = run_segmentation(...)
            t.input_shape = image.shape
            t.output_shape = masks.shape
            t.confidence = float(masks.scores.mean())

    On exception, logs an error (with the exception) and re-raises — it does not swallow
    failures.
    """

    stage: str
    logger: logging.Logger
    device: str | None = None
    model: str | None = None
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    confidence: float | None = None
    _start: float = field(default=0.0, init=False, repr=False)

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        self.logger.info(
            "stage=%s status=start device=%s model=%s", self.stage, self.device, self.model
        )
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        elapsed = time.perf_counter() - self._start
        if exc_type is not None:
            self.logger.error(
                "stage=%s status=error elapsed_s=%.3f device=%s model=%s error=%s",
                self.stage,
                elapsed,
                self.device,
                self.model,
                exc,
            )
            return

        self.logger.info(
            "stage=%s status=done elapsed_s=%.3f device=%s model=%s input_shape=%s output_shape=%s "
            "confidence=%s gpu_mem_mb=%s",
            self.stage,
            elapsed,
            self.device,
            self.model,
            self.input_shape,
            self.output_shape,
            self.confidence,
            get_gpu_memory_mb(),
        )
