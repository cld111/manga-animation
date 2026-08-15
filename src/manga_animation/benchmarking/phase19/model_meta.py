"""Verified OMG-LLaVA model facts, from direct inspection of the official code and weights.

Phase 19 section 3 requires verifying the model from the official repository *before* building
benchmark infrastructure and forbids assuming functionality from the paper title/abstract. Every
fact below was confirmed by reading the official `lxtGH/OMG-Seg` `omg_llava/` source (chat tool,
gradio app, finetune config) and the official HF weights listing (`zhangtao-whu/OMG-LLaVA`) --
not from the arXiv abstract.

Module-level purity: this file is plain data, no ML imports, so it is testable and importable on
the local dev machine.
"""

from __future__ import annotations

# --- Official upstream identity -----------------------------------------------------------
OFFICIAL_REPO = "https://github.com/lxtGH/OMG-Seg"
OMG_LLAVA_SUBTREE = "omg_llava"
HF_REPO = "zhangtao-whu/OMG-LLaVA"
OFFICIAL_INFERENCE_ENTRY = "omg_llava/tools/chat_omg_llava.py"
OFFICIAL_GRADIO_ENTRY = "omg_llava/tools/app.py"
FINETUNE_CONFIG = "omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py"
PAPER = "https://arxiv.org/abs/2406.19389"

# --- Weights -----------------------------------------------------------------------------
# The chat tool builds the model from the config (base LLM + ConvNeXt encoder + OMG-Seg head,
# loaded via init_cfg) and then loads the finetune checkpoint with strict=False. All of these
# files must be on disk on the GPU worker; the config's local paths must point at them.
CHECKPOINT = "omg_llava_7b_finetune_8gpus.pth"
CHECKPOINT_BYTES = 8_428_615_400  # from the HF file listing (8.4 GiB)
BASE_LLM = "internlm2-chat-7b"  # shipped as a directory inside HF_REPO
CONVNEXT_HEAD_PRETRAIN = "omg_seg_convl.pth"
OV_CLASS_EMBED = "convnext_large_d_320_CocoPanopticOVDataset.pth"
# Specialized finetunes also shipped on HF (refseg = referring-expression segmentation focus).
REFSEG_CHECKPOINT = "finetuned_refseg.pth"
GCG_CHECKPOINT = "finetuned_gcg.pth"

REQUIRED_WEIGHT_PATHS = (BASE_LLM, CONVNEXT_HEAD_PRETRAIN, OV_CLASS_EMBED, CHECKPOINT)
"""All four artifacts the official finetune config + checkpoint need on the worker."""

# --- Architecture (from the finetune config) ---------------------------------------------
LLM_BACKBONE = "internlm2-chat-7b"
LLM_DTYPE = "float16"
VISUAL_ENCODER = (
    "ConvNeXt-Large-320 (openclip laion2b soup) + Mask2Former head "
    "(MSDeformAttn pixel decoder, 9-layer transformer decoder, 300 queries) "
    "-- the OMG-Seg universal segmentation encoder"
)
SEG_TOKEN = "[SEG]"  # the LLM emits this token where a pixel mask is requested
IMAGE_PROCESSOR = "CLIPImageProcessor"
IMAGE_SIZE = 1024  # official finetune config: size=1024, center-crop 1024
IMAGE_MEAN = (0.4814, 0.4578, 0.4082)
IMAGE_STD = (0.2686, 0.2613, 0.2757)
PROMPT_TEMPLATE = "internlm2_chat"  # official default in both chat and app tools
# max_length = 2048 - (1024/64)**2 - 100  ->  the visual-token count for a 1024px image.
MAX_LENGTH = 2048 - (IMAGE_SIZE // 64) ** 2 - 100

# --- Verified capabilities (from code + official README, not the abstract) ---------------
# 1. Natural-language referring segmentation: the tool and the refcoco eval entry
#    (`refcoco_omg_seg_llava`) confirm the model consumes arbitrary referring expressions.
# 2. Pixel-level segmentation: `chat_omg_llava.py` extracts hidden states at every [SEG]
#    output token, projects them with `projector_text2vision`, and runs
#    `visual_encoder.forward_llm_seg(...)` -> masks (sigmoid > 0.5).
# 3. Multiple instances: an output may contain several [SEG] tokens; each gets its own mask
#    (confirmed by the gradio app's grounded-caption description and the colors loop).
# 4. Mask representation: `pred_masks_list[-1]` shape (N_masks, 1, H, W), thresholded at 0.5
#    and bilinear-resized to the padded-square canvas.
# 5. Coordinate convention: images are padded to a square (expand2square, mean-color fill)
#    before preprocessing; masks are produced on that padded square. The app crops the padding
#    back out (sx/sy/ex/ey) -- `masks.py` implements the inverse transform.
# 6. Prompt style that yields [SEG]: grounded-caption form, e.g.
#    "Could you provide me with a detailed analysis of this photo? Please output with
#    interleaved segmentation masks for the corresponding parts of the answer." and the
#    referring form "Can you please segment <target> in the given image".

# --- Hardware / runtime facts ------------------------------------------------------------
OFFICIAL_VRAM_NOTE = (
    "the official README suggests at least a 32 GB GPU for the 7B models; on 2xT4-16GB the "
    "fp16 LLM alone (~14 GB weights) plus activations will not fit, so the official config's "
    "bitsandbytes 4-bit LLM quantization (--bits 4) or device offload is the practical path"
)
OFFICIAL_STACK = (
    "python 3.10 + torch 2.1.2 (cu118) + mmengine 0.8.5 + mmcv (2024 commit) + mmdet 3.1.0 + "
    "mmsegmentation 1.1.1 + mmpretrain 1.0.1 + xtuner + peft + transformers (2024-era, "
    "internlm2 trust_remote_code) -- see the official INSTALL.md"
)
LICENSE = "Apache-2.0 (OMG-LLaVA subtree; the OMG-Seg repo root is MIT)"
"""
from the repo README: OMG-LLaVA follows the Apache-2.0 license of LLaVA/XTuner; OMG-Seg follows
MIT. Third-party weights on HF carry their own licenses (internlm2 chat 7B is the InternLM
model-family license).
"""

# --- Checkpoint/preprocessing summary for the report --------------------------------------
VERIFICATION_SUMMARY = (
    "OMG-LLaVA 7B (internlm2-chat-7b + ConvNeXt-Large-320 OMG-Seg encoder), finetune "
    "checkpoint `omg_llava_7b_finetune_8gpus.pth`, official chat entry "
    "`chat_omg_llava.py`, CLIP preprocess at 1024x1024 on an expand2square canvas, natural-"
    "language prompts with interleaved [SEG] tokens for pixel masks, multiple instances "
    "supported, masks thresholded at sigmoid 0.5 on the padded canvas, Apache-2.0, official "
    "guidance >= 32 GB VRAM for the 7B model."
)
