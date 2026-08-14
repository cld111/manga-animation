"""Deterministic GPU model lifecycle: `ModelStage` owns one model client's residency.

Why this exists (Phase 14, evidence from a real Kaggle 2xT4 run, see docs/phase14-results.md):
`torch.cuda.empty_cache()` only releases blocks the CUDA caching allocator no longer considers
allocated. A model whose Python references are still alive -- including references held inside
cyclic garbage that `gc` has not yet collected -- keeps its blocks allocated no matter how many
times `empty_cache()` is called. A `device_map="auto"` model (the Qwen2.5-VL 7B client's load
path, ADR 0005) demonstrably leaves such cycles: measuring a real Qwen load/unload on the GPU,
`model = None; torch.cuda.empty_cache()` left ~16 GiB allocated, and only
`gc.collect(); torch.cuda.empty_cache()` returned it to zero. The old pipeline called
`empty_cache()` without `gc.collect()`, so every Qwen load OOM'd once two generations were alive
at once.

`ModelStage` makes that release deterministic and exception-safe: entering the stage optionally
loads the client, exiting always drops the client's references, collects cyclic garbage, and
flushes the caching allocator -- whether the stage body succeeded or raised. Every model-backed
pipeline stage (analysis, validation, semantic mask validation, grounding, segmentation,
reconstruction) runs inside one, so a failed panel or a mid-stage exception can never leave a
model resident and poison the next panel.
"""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager
from typing import Any

from manga_animation.core.logging import get_logger

logger = get_logger(__name__)


def release_device_memory(device: str | None = None) -> None:
    """Deterministically release a model's CUDA blocks, in the only order that works for
    `device_map="auto"` transformers models (Phase 14 evidence): collect cyclic Python garbage
    first (the model's tensors are unreachable-but-alive until then), then flush the caching
    allocator, then `ipc_collect()` for any shared-memory references. Safe to call on a CPU-only
    machine (no-op) and when no model is resident (also a no-op beyond collecting garbage).
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return  # local/dev machines without the `ml` extra must stay import-safe (ADR 0003)
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _release_memory_log(before_mb: float, after_mb: float, name: str) -> None:
    released = before_mb - after_mb
    if released > 16.0:  # only log a materially resident stage, not per-tiny-stage churn
        logger.info(
            "model stage %r released %.0f MiB of CUDA allocator memory after completion",
            name,
            released,
        )


class ModelStage(AbstractContextManager["ModelStage"]):
    """Owns one model client's GPU residency for the duration of a pipeline stage.

    Use as a context manager: the client is loaded on entry (when it has a `load()` method;
    the VLM client lazy-loads inside its own `generate()` instead) and deterministically
    unloaded on exit -- on normal completion AND on any exception. Explicit ownership means
    no stage accidentally retains another stage's model, a failed panel still releases its
    models, and a stage that terminates early (via exception) cannot leave the GPU in an
    unusable state for the next stage.

    `name` is only for logging. When `auto_load` is `True` (default) the client's `load()`
    method is called on entry if it has one; the client's own `load()` must be idempotent
    (the production clients guard on `self.model is not None`).
    """

    def __init__(
        self, client: Any, *, name: str, device: str | None = None, auto_load: bool = True
    ) -> None:
        self.client = client
        self.name = name
        self.device = device
        self.auto_load = auto_load
        self._active = False

    def __enter__(self) -> ModelStage:
        if self._active:
            raise RuntimeError(f"ModelStage {self.name!r} entered while already active")
        self._active = True
        if self.auto_load:
            loader = getattr(self.client, "load", None)
            if callable(loader):
                try:
                    loader()
                except BaseException:
                    # Python never calls `__exit__` when `__enter__` raises, so a failed
                    # load() would otherwise leave `_active=True` (poisoning this stage object
                    # against any later reuse) and any CUDA caching-allocator blocks allocated
                    # before the failure un-released. Reset the guard and attempt the same
                    # deterministic release the normal exit path uses before the load error
                    # propagates.
                    self._active = False
                    self._unload()
                    raise
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        unload_error: BaseException | None = None
        try:
            self._unload()
        except Exception as err:  # noqa: BLE001 -- cleanup must never mask the stage result
            unload_error = err
        finally:
            self._active = False
        if unload_error is not None:
            if exc_type is None:
                # The stage body completed; a cleanup failure still fails the run (fail-closed).
                raise unload_error
            # The stage body already raised: log the cleanup failure but never replace the
            # stage's own exception -- a broken unload must not mask the real failure.
            logger.warning(
                "model stage %r: cleanup failed while unwinding a stage exception: %s",
                self.name,
                unload_error,
            )

    def _unload(self) -> None:
        before_mb = self._allocated_mb()
        unloader = getattr(self.client, "unload", None)
        try:
            if callable(unloader):
                unloader()
        finally:
            # The Phase 14 leak fix (gc.collect() -> empty_cache() -> ipc_collect()) must run
            # even when the client's own unload() raises (e.g. a broken CUDA context after a
            # real OOM can make empty_cache() fail): a failed unload must not silently skip the
            # deterministic release of the previous stage's model.
            release_device_memory(self.device)
            _release_memory_log(before_mb, self._allocated_mb(), self.name)

    def _allocated_mb(self) -> float:
        try:
            import torch
        except ImportError:
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        try:
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.memory_allocated(i)
            return total / 2**20
        except Exception:  # noqa: BLE001 -- memory accounting must never break a stage exit
            return 0.0
