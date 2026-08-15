"""Phase 19 adapter tests (torch-free): the module imports cleanly on the local dev machine
(no torch/xtuner/mmcv), the official prompt template assembly is exact, and the adapter object
can be constructed without the GPU stack."""

from __future__ import annotations

import importlib

from manga_animation.benchmarking.phase19.adapter import (
    OMGLLavaAdapter,
    prompt_text,
)


def test_module_imports_without_torch():
    # Module import must not pull torch/transformers/xtuner (ADR 0003).
    module = importlib.import_module("manga_animation.benchmarking.phase19.adapter")
    assert "torch" not in {m for m in module.__dict__ if m.startswith("torch")}
    assert module.OMGLLavaAdapter


def test_prompt_text_official_internlm2_template():
    text = prompt_text("Can you please segment the character body in the given image")
    assert text.startswith("<|im_start|>user\n")
    assert "<image>\nCan you please segment the character body in the given image" in text
    assert text.endswith("<|im_start|>assistant\n")
    assert "<|im_end|>\n" in text
    assert text.count("<image>") == 1


def test_adapter_construction_without_gpu_stack(tmp_path):
    adapter = OMGLLavaAdapter(
        config_path=str(tmp_path / "config.py"),
        pth_path=str(tmp_path / "model.pth"),
        device="cuda",
        llm_bits="4",
        max_new_tokens=256,
    )
    assert adapter.max_new_tokens == 256
    assert adapter.llm_bits == "4"
    assert not adapter._model  # nothing loaded
