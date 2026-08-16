"""The VLM client seam: `plan_builder` depends only on `VLMClient`, never on `torch`/

`transformers` directly, so it stays importable and unit-testable without the `ml` extra
installed (see ADR 0003 — heavy ML libs are a remote-GPU-worker-only dependency).
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image


class VLMClient(Protocol):
    """What `plan_builder` needs from a vision-language model."""

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Run one image+prompt -> text generation call."""
        ...

    def unload(self) -> None:
        """Release model and processor memory after the analysis/validation stages."""
        ...


class Qwen3VLClient:
    """Real `qwen3-vl-8b-instruct` client (Phase 20), with explicit `load()` for co-residency.

    Phase 20 moves every model family to run-level co-residency: all clients are loaded at the
    start of a `run_pages` invocation and stay resident until it finishes (ADR 0021), instead
    of the stage-by-stage load/unload of ADR 0020. This client therefore exposes an explicit
    idempotent `load()` (in addition to the lazy `_ensure_loaded()` used by `generate()`) so
    `ModelStage` can bring Qwen up together with DINO/SAM/LaMa on entry.

    `device_map="auto"` is required, same as ADR 0005's Qwen2.5-VL path: Qwen3-VL-8B's float16
    weights alone exceed one T4, so the session's 2xT4 must share the model. Qwen3-VL adds
    native thinking mode; the structured JSON object description needs it disabled, so the chat
    template is applied with `enable_thinking=False`.
    """

    def __init__(self, source: str, dtype: str, max_new_tokens: int = 4096) -> None:
        self.source = source
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Explicitly load the model and processor (idempotent). Phase 20 co-residency:
        the run-level `ModelStage` calls this once, at the start of the whole run."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.source)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype), device_map="auto"
        )
        self._model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Inputs go on the sharded model's first device, not a fixed device string -- with
        # device_map="auto" the model spans multiple devices (see ADR 0005).
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        return str(self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0])

    def unload(self) -> None:
        # Same deterministic release path as the Qwen2.5-VL client (Phase 14 evidence): a
        # `device_map="auto"` model keeps cyclic Python references alive, so the CUDA
        # allocator still counts its tensors until `gc.collect()` runs. Order matters:
        # collect cyclic garbage first, then flush the caching allocator.
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen25VLClient:
    """Real `qwen2.5-vl-7b-instruct` client, per ADR 0005's confirmed-working call.

    `device_map="auto"` (not `.to(device)`) is required: a single T4 OOMs on this model's
    float16 weights alone (14.15 GiB on a 14.56 GiB-usable card) — sharding across the
    session's 2xT4 is what actually worked on the real Kaggle run this mirrors, see
    `scripts/phase2_kaggle_benchmark.py`'s `Qwen25VLAdapter`.
    """

    def __init__(self, source: str, dtype: str, max_new_tokens: int = 4096) -> None:
        self.source = source
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.source)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.source, torch_dtype=getattr(torch, self.dtype), device_map="auto"
        )
        self._model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Inputs go on the sharded model's first device, not a fixed device string — with
        # device_map="auto" the model spans multiple devices (see ADR 0005).
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        return str(self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0])

    def unload(self) -> None:
        # Phase 14 (docs/phase14-results.md): this client is the one model in the pipeline
        # whose tensors survive `self._model = None; empty_cache()` -- a `device_map="auto"`
        # model (ADR 0005's Qwen sharding path) keeps cyclic Python references alive, so the
        # CUDA allocator still counts its ~16 GiB until `gc.collect()` runs. Without it, a
        # second Qwen load races the first unreleased instance and OOMs (reproduced on a real
        # 2xT4 Kaggle run). The order matters: collect cyclic garbage first, then release the
        # caching allocator's now-unreachable blocks.
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
