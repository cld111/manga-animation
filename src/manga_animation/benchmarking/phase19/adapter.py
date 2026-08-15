"""OMG-LLaVA inference adapter reproducing the OFFICIAL `chat_omg_llava.py` flow.

This adapter is a faithful re-implementation of the official inference entry point (verified
from `lxtGH/OMG-Seg/omg_llava/omg_llava/tools/chat_omg_llava.py`), with the exact same steps:

    page -> expand2square (mean-color pad) -> CLIP preprocess 1024x1024
      -> visual_encoder(image, output_hidden_states=True) -> projector(visual_outputs)
      -> llm.generate(inputs_embeds=..., output_hidden_states=True, return_dict_in_generate=True)
      -> hidden states at [SEG] positions -> projector_text2vision
      -> visual_encoder.forward_llm_seg -> masks (sigmoid > 0.5, bilinear to padded canvas)

Everything heavy is lazy-imported inside methods (the local dev machine has no torch/xtuner/
mmcv, ADR 0003), so this module is importable and its pure helpers are unit-testable locally.
The official mask-extraction logic is copied verbatim (`get_seg_hidden_states`, `_show_mask`)
so alignment behavior matches the official tool exactly -- including its quirks.

Hardware strategy (phase brief section 15): the official README recommends >= 32 GB for the 7B
model. On 2xT4-16GB the practical path is the config's bitsandbytes 4-bit LLM (the finetune
config already carries `quantization_config`), or fp16 with `device_map="auto"` sharding. The
strategy is a documented load-time choice selected by the smoke test, never silently assumed.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from manga_animation.benchmarking.phase19.masks import SquarePad

# The internlm2_chat template (official default in both chat and gradio tools). Reproduced
# from xtuner's PROMPT_TEMPLATE.internlm2_chat.
_TEMPLATE_INSTRUCTION = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
_TEMPLATE_STOP_WORDS = ["<|im_end|>", "<|im_start|>"]

_DEFAULT_IMAGE_TOKEN = "<image>"
_IMAGE_TOKEN_INDEX = -200  # xtuner's IMAGE_TOKEN_INDEX (placeholder replaced in embedding space)


@dataclass
class OMGLLavaOutput:
    """One prediction: the decoded text and the emitted masks.

    `masks` are boolean arrays on the PADDED-SQUARE canvas (the official output convention);
    the caller maps them to original page coordinates with `masks.mask_from_canvas` /
    `SquarePad`. `latency_seconds` is the generation time only (model already loaded).
    """

    text: str
    masks: list[np.ndarray]  # (S, S) boolean, one per [SEG] token, in [SEG] order
    latency_seconds: float
    seg_ids: list[int] = field(default_factory=list)  # positions in the generated ids

    @property
    def n_masks(self) -> int:
        return len(self.masks)


def get_seg_hidden_states(hidden_states, output_ids, seg_id: int):
    """Verbatim copy of the official `chat_omg_llava.py::get_seg_hidden_states`.

    Copied unchanged (including the `[-n_out:]` slice and boolean mask) so our extraction
    matches the official tool's alignment behavior exactly. `hidden_states` is the concatenated
    per-step last-layer hidden states; `output_ids` is `generate_output.sequences[0][:-1]`.
    """
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


def prompt_text(user_text: str) -> str:
    """The exact first-turn prompt the official chat tool builds: `<image>\\n<user text>`
    wrapped in the internlm2_chat instruction template. Pure string logic (testable locally)."""
    text = f"{_DEFAULT_IMAGE_TOKEN}\n{user_text}"
    return _TEMPLATE_INSTRUCTION.format(input=text, round=1, bot_name="BOT")


def _show_mask_threshold(
    masks: Any,
    canvas_size: int,
) -> list[np.ndarray]:
    """The mask post-processing half of the official `show_mask_pred`: bilinear-interpolate the
    logits to the padded-square canvas, apply sigmoid > 0.5, return boolean (N, S, S) arrays.

    Operates on a torch tensor (the worker env) and returns CPU numpy booleans. This is the
    faithful reproduction of `F.interpolate(masks, size=image.size, mode='bilinear',
    align_corners=False); masks.sigmoid() > 0.5` with `image.size == (S, S)`.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = F.interpolate(
            masks, size=(canvas_size, canvas_size), mode="bilinear", align_corners=False
        )
        binary = (logits.sigmoid() > 0.5).to(torch.uint8).cpu().numpy()
    return [binary[i, 0] for i in range(binary.shape[0])]


class OMGLLavaAdapter:
    """Loads the official OMG-LLaVA checkpoint and runs one referring/autonomous prediction.

    `config_path` must be the official finetune config (or a copy with paths pointed at the
    worker's weights); `pth_path` the `omg_llava_7b_finetune_8gpus.pth` checkpoint. The
    adapter builds the model through mmengine's `BUILDER` exactly as the official tool does
    and loads the checkpoint with `strict=False`.
    """

    def __init__(
        self,
        config_path: str,
        pth_path: str,
        *,
        device: str = "cuda",
        llm_bits: str | None = None,
        shard_two_gpus: bool = False,
        resolution: int = 1024,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.75,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        offload_folder: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.pth_path = Path(pth_path)
        self.device = device
        self.llm_bits = llm_bits  # None=official config, "4", "8", or "fp16"
        self.shard_two_gpus = shard_two_gpus
        self.resolution = resolution
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.offload_folder = offload_folder
        self._model: Any = None
        self._tokenizer: Any = None
        self._image_processor: Any = None

    # --- lifecycle -------------------------------------------------------------------------
    def load(self) -> None:
        """Build the model from the official config, apply the documented LLM-bits strategy,
        and load the finetune checkpoint. Raises RuntimeError with a clear message when the
        chosen strategy cannot be built (the smoke test must pick a working one first)."""
        if self._model is not None:
            return  # idempotent
        from mmengine.config import Config
        from xtuner.model.utils import guess_load_checkpoint
        from xtuner.registry import BUILDER

        cfg = Config.fromfile(str(self.config_path))
        # The official chat tool nulls the config's pretrain checkpoint before building --
        # the finetune pth passed on the CLI is loaded separately via load_state_dict.
        cfg.model.pretrained_pth = None
        if self.resolution != 1024:
            # Non-official resolution: patch the CLIP processor. The official config is 1024;
            # on 2xT4-16GB the official 1024x1024 forward OOMs (measured), so the benchmark
            # documents the highest resolution that fits (phase brief section 14).
            for key in ("size", "crop_size"):
                if key in cfg.image_processor:
                    cfg.image_processor[key] = self.resolution
        self._apply_llm_strategy(cfg)
        model = BUILDER.build(cfg.model)

        state_dict = guess_load_checkpoint(str(self.pth_path))
        model.load_state_dict(state_dict, strict=False)

        tokenizer_cfg = dict(cfg.tokenizer)
        tokenizer_type = tokenizer_cfg.pop("type")
        tokenizer = tokenizer_type(**tokenizer_cfg)

        ip_cfg = dict(cfg.image_processor)
        ip_type = ip_cfg.pop("type")
        image_processor = ip_type(**ip_cfg)

        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._model = model
        self._place_model()

        self._model.eval()
        for part in (self._model.llm, self._model.visual_encoder):
            part.eval()

    def _apply_llm_strategy(self, cfg: Any) -> None:
        """Override the config's LLM build for the documented hardware strategies.

        - llm_bits is None: leave the official config untouched (it already carries the 4-bit
          bitsandbytes quantization_config).
        - llm_bits == "4"/"8": keep/force the corresponding quantization.
        - llm_bits == "fp16": drop quantization, run float16.
        - shard_two_gpus: `device_map="auto"` for the LLM so its weights split across the two
          T4s (fp16 7B alone is ~14 GB, too large for one 16 GB card with activations).
        """
        import torch

        llm_cfg = cfg.model.llm
        bits_cfg = llm_cfg.get("quantization_config", {})
        has_quant = bool(bits_cfg)
        if self.llm_bits is None:
            return  # official config as-is
        if self.llm_bits in ("4", "8"):
            if self.llm_bits == "4":
                llm_cfg.setdefault("quantization_config", {})["load_in_4bit"] = True
                llm_cfg["quantization_config"]["load_in_8bit"] = False
            else:
                llm_cfg["quantization_config"] = {"load_in_8bit": True}
            llm_cfg["torch_dtype"] = torch.float16
        elif self.llm_bits == "fp16":
            if has_quant:
                llm_cfg["quantization_config"] = None
            llm_cfg["torch_dtype"] = torch.float16
        else:
            raise ValueError(f"unknown llm_bits {self.llm_bits!r}")
        if self.shard_two_gpus:
            llm_cfg["device_map"] = "auto"
            if self.offload_folder:
                llm_cfg["offload_folder"] = self.offload_folder

    def _place_model(self) -> None:
        """Place the model on GPU(s). With LLM sharding (`device_map="auto"`) the LLM is already
        placed by accelerate and must not be moved wholesale; otherwise the official flow moves
        the whole model onto `self.device`."""

        if self.shard_two_gpus:
            # Only the non-sharded submodules need explicit placement.
            self._model.visual_encoder.to(self.device)
            self._model.projector.to(self.device)
            if hasattr(self._model, "projector_text2vision"):
                self._model.projector_text2vision.to(self.device)
        else:
            self._model.to(self.device)

    def unload(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- inference -------------------------------------------------------------------------
    def predict(self, image: np.ndarray, text_prompt: str) -> OMGLLavaOutput:
        """Run one full-page prediction. `image` is RGB (H, W, 3); `text_prompt` is the raw
        referring expression / autonomous instruction (no image token, no template -- the
        adapter adds them exactly as the official tool does)."""
        if self._model is None:
            raise RuntimeError("adapter is not loaded -- call load() first")
        import torch
        from xtuner.dataset.utils import expand2square
        from xtuner.model.utils import prepare_inputs_labels_for_multimodal
        from xtuner.tools.utils import get_stop_criteria

        model = self._model
        tokenizer = self._tokenizer
        image_processor = self._image_processor

        start = time.perf_counter()

        # --- image preprocessing (official first-turn flow) -------------------------------
        from PIL import Image

        pil_image = Image.fromarray(image)
        pad = SquarePad.from_page_size(tuple(image.shape[:2]))
        pad_color = tuple(int(x * 255) for x in image_processor.image_mean)
        image_for_show = expand2square(pil_image, pad_color)
        processed = image_processor.preprocess(image_for_show, return_tensors="pt")[
            "pixel_values"
        ][0]
        processed = (
            processed.unsqueeze(0).to(self.device).to(model.visual_encoder.dtype)
        )
        with torch.no_grad():
            visual_outputs = model.visual_encoder(processed, output_hidden_states=True)
            pixel_values = model.projector(visual_outputs)

        # --- prompt assembly (official template) ------------------------------------------
        prompt_text_built = prompt_text(text_prompt)
        chunks = prompt_text_built.split(_DEFAULT_IMAGE_TOKEN)
        ids: list[int] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                cur = tokenizer.encode(chunk)
            else:
                cur = tokenizer.encode(chunk, add_special_tokens=False)
            ids.extend(cur)
            if idx != len(chunks) - 1:
                ids.append(_IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(ids).to(self.device).unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=model.llm, input_ids=input_ids, pixel_values=pixel_values
        )

        stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=_TEMPLATE_STOP_WORDS)
        generation_kwargs = {
            "generation_config": _generation_config(
                tokenizer, self.max_new_tokens, self.temperature, self.top_p,
                self.top_k, self.repetition_penalty,
            ),
            "streamer": None,
            "bos_token_id": tokenizer.bos_token_id,
            "stopping_criteria": stop_criteria,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        with torch.no_grad():
            generate_output = model.llm.generate(
                **mm_inputs,
                **generation_kwargs,
            )

        # --- text decode + mask extraction (official flow) --------------------------------
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        output_ids = generate_output.sequences[0][:-1]
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, output_ids, seg_id=model.seg_token_idx
        )
        masks: list[np.ndarray] = []
        seg_ids: list[int] = []
        if len(seg_hidden_states) != 0:
            seg_hidden_states = model.projector_text2vision(seg_hidden_states)
            batch_idxs = torch.zeros(
                (seg_hidden_states.shape[0],), dtype=torch.int64, device=seg_hidden_states.device
            )
            pred_masks_list = model.visual_encoder.forward_llm_seg(
                seg_hidden_states, batch_idxs
            )
            masks = _show_mask_threshold(pred_masks_list[-1], pad.canvas_size)
            seg_ids = (output_ids == model.seg_token_idx).nonzero().flatten().tolist()

        text_out = tokenizer.decode(generate_output.sequences[0])
        latency = time.perf_counter() - start
        return OMGLLavaOutput(text=text_out, masks=masks, latency_seconds=latency,
                              seg_ids=seg_ids)


def _generation_config(tokenizer, max_new_tokens, temperature, top_p, top_k,
                       repetition_penalty):
    """The official `GenerationConfig` (same fields/values as the chat tool)."""
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
