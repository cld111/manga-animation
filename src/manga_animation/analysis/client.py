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

    `device_map="auto"` is required for Qwen3-VL-8B's float16: the weights alone exceed one
    T4, so the session's 2xT4 must share the model (Phase 20 baseline, ADR 0021). A SMALLER
    fp16 model (e.g. Qwen3-VL-4B, ~8.5 GiB) can instead run as ONE instance per GPU
    (`device` given -> `device_map={"": device}`, Phase 22 A/B, ADR 0023 per-GPU scheme):
    the pipeline then creates one instance per card and the description stage splits panels
    between them, with no cross-GPU token traffic. Qwen3-VL's thinking mode is a model-side
    generation flag; the repo's chat template (transformers 5.0.0) renders no thinking block
    by default, so structured JSON output is unaffected.
    """

    def __init__(
        self,
        source: str,
        dtype: str,
        max_new_tokens: int = 4096,
        device: str | None = None,
    ) -> None:
        self.source = source
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.device = device  # None -> device_map="auto" (shard); "cuda:N" -> one instance per GPU
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
            self.source,
            torch_dtype=getattr(torch, self.dtype),
            device_map={"": self.device} if self.device else "auto",
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


class Qwen3VLInt8Client:
    """Per-GPU `qwen3-vl-8b` int8 client (Phase 22, ADR 0023): ONE instance per GPU.

    Phase 20/21's fp16 client sharded one Qwen across both T4s (`device_map="auto"`), so
    every decoded token crossed the GPU boundary and both cards were only partially
    utilized. This client instead loads a bitsandbytes int8 quantized copy of the model
    ONTO A SINGLE GPU (~9.5 GiB): the pipeline creates one instance per card and the
    description stage runs them as a parallel worker pool, splitting panels between the
    cards.

    The weights are pre-quantized ONCE (the fp16 repo id -> int8 conversion, saved with
    `BitsAndBytesConfig(load_in_8bit=True, pre_quantized=True)`) so loading never
    materializes the 16 GiB fp16 checkpoint on the card: with fp16-on-disk, transformers
    5.0.0 materializes the whole fp16 checkpoint before quantizing (OOM on one T4), while
    the pre-quantized safetensors load straight into the bnb int8 layout (~3 s).
    """

    def __init__(
        self,
        source: str,
        device: str,
        max_new_tokens: int = 4096,
    ) -> None:
        self.source = source
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        """Load the int8 model onto this instance's GPU (idempotent)."""
        if self._model is not None:
            return
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        self._processor = AutoProcessor.from_pretrained(self.source)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.source,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True, pre_quantized=True),
            device_map={"": self.device},
        )
        self._model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        assert self._model is not None and self._processor is not None, (
            "Qwen3VLInt8Client.generate() requires load() first (the pipeline's run-level "
            "ModelStage calls it on entry)"
        )

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
        )
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, prompt_len:]
        return str(self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0])

    def unload(self) -> None:
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
