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


class Qwen25VLClient:
    """Real `qwen2.5-vl-7b-instruct` client, per ADR 0005's confirmed-working call.

    `device_map="auto"` (not `.to(device)`) is required: a single T4 OOMs on this model's
    float16 weights alone (14.15 GiB on a 14.56 GiB-usable card) — sharding across the
    session's 2xT4 is what actually worked on the real Kaggle run this mirrors, see
    `scripts/phase2_kaggle_benchmark.py`'s `Qwen25VLAdapter`.
    """

    def __init__(self, source: str, dtype: str, max_new_tokens: int = 512) -> None:
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
        import torch

        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
