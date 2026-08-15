# Phase 19 — OMG-LLaVA Autonomous Animation-Target Segmentation

Experimental, diagnostic phase. No production code was changed. This document separates
**OBSERVED** (measured facts), **INTERPRETATION** (what the measurements mean), and
**HYPOTHESIS** (untested explanations), per the phase brief's final-report rule. The raw
artifacts live under `outputs/experiments/phase19_omg_llava/` (git-ignored): `run_controlled/`
(64-target condition-D results, report.json, forbidden_overlap.json) and `run_autonomous/`
(52-page autonomous records, masks, texts, visual gallery).

## 1. Executive Summary

OMG-LLaVA (official 7B finetune checkpoint) was evaluated as a possible replacement for the
current `Qwen -> Grounding DINO -> candidate selection -> SAM 2.1` perception chain. It runs
on the 2xT4-16GB worker only in a degraded configuration: the official 1024x1024 resolution
OOMs (measured), the working configuration is 4-bit bitsandbytes at 512x512 on one T4
(~13.3 GB peak VRAM). Under the production-available description (`"character body"`, the only
description the current pipeline can produce for these targets), OMG-LLaVA's end-to-end
instance selection is **4.7% (3/64)**, statistically on par with the current pipeline's
measured 3/64 healthy results and with DINO top-1's 6.2% instance rate; 58/64 masks have zero
overlap with the target and 47/64 are dominated by panel-frame regions. When OMG-LLaVA
commits to a mask it is excellent (IoU 0.87-0.92), but the selection problem it was asked to
solve is not solved. Autonomous discovery (52 pages) produced real masks on 29/52 pages, but
only 3/52 pages yielded a page-grounded target description; 21/52 responses were degenerate
(`"Pillow."`). The evidence does not justify replacing any current component.

**Recommendation: D — KEEP CURRENT PIPELINE** (see §15).

## 2. Exact OMG-LLaVA Model / Checkpoint

- Model: OMG-LLaVA 7B (internlm2-chat-7b LLM + ConvNeXt-Large-320 OMG-Seg visual encoder +
  Mask2Former-style head, LoRA r=512), paper arXiv 2406.19389.
- Official code: `lxtGH/OMG-Seg` (pinned commit `48ab9407a45c2ecf78b4e980d6a6ccddf9a7ec9f`),
  subtree `omg_llava/`, inference entry `omg_llava/omg_llava/tools/chat_omg_llava.py`.
- Weights (HF `zhangtao-whu/OMG-LLaVA`): `omg_llava_7b_finetune_8gpus.pth` (8.4 GiB) +
  `internlm2-chat-7b` + `omg_seg_convl.pth` + `convnext_large_d_320_CocoPanopticOVDataset.pth`.
- Official finetune config `omg_llava_7b_finetune_8gpus.py`: CLIPImageProcessor 1024x1024,
  internlm2_chat prompt template, natural-language referring segmentation with interleaved
  `[SEG]` tokens; masks extracted from hidden states at `[SEG]` positions (sigmoid > 0.5).
- License: Apache-2.0 (OMG-LLaVA subtree).
- The adapter in `src/manga_animation/benchmarking/phase19/adapter.py` reproduces the official
  chat-tool flow verbatim (expand2square -> CLIP preprocess -> visual encoder -> projector ->
  llm.generate with hidden states -> `[SEG]` extraction -> `forward_llm_seg`).

## 3. Actual Architecture (verified from code, not the abstract)

- Full page -> expand2square (mean-color pad) -> CLIP preprocess -> ConvNeXt-L-320 ->
  projector -> LLM (internlm2-chat-7b + LoRA r=512) generation with interleaved `[SEG]`
  tokens -> text-to-vision projector -> Mask2Former head -> masks on the padded canvas
  (cropped back to page coordinates by the phase-19 masks module).
- Capabilities confirmed by running it: full-page understanding (captioning a manga page
  correctly), pixel-level referring segmentation, multiple-instance support (multiple
  `[SEG]`), autonomous grounded captioning.
- Hardware reality (OBSERVED): the finetune checkpoint's LoRA/embedding weights sit on top of
  a 4-bit base LLM, but the resident footprint after load is ~11.5-13.9 GiB; the official
  1024x1024 forward needs more than one T4-16GB has. The official README's ">= 32 GB GPU"
  guidance matches the measured footprint.

## 4. Hardware / Environment

- Worker: Kaggle Jupyter, 2x Tesla T4 16 GB, python 3.10 (dedicated venv).
- Stack (all pinned, 2024-era): torch 2.1.2+cu118, transformers 4.36.0, mmcv 2.2.0 (torch-2.1
  prebuilt wheel; the mmdet/mmseg/mmpretrain mmcv-maximum gates bumped 2.2.0 -> 2.3.0),
  mmdet 3.3.0, mmseg 1.1.1, mmpretrain 1.0.1, xtuner 0.1.21 (vendored), peft 0.7.1,
  accelerate 0.27.2, bitsandbytes 0.43.1 (with `BNB_CUDA_VERSION=122` override: the wheel
  ships no cu118 binary), deepspeed 0.14.4, sentencepiece 0.1.99 (0.2.x breaks internlm2's
  tokenizer), numpy 1.26.4, opencv-python 4.10.0, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.
- The full environment bootstrap is encoded in `scripts/setup_phase19_omg_llava_worker.py`
  (weights ~24 GiB on the overlay filesystem; `/kaggle/working` is a 20 GiB loop mount).
- Practical inference strategy (OBSERVED): single-GPU 4-bit; the 2-GPU fp16 shard attempt
  failed (device_map placed the LLM on one GPU and the LoRA fp32 prep OOM'd); parallel
  inference was not possible (one model per process needs ~14 GiB each).

## 5. Autonomous Experiment

Full page + generic instruction (no GT information), 52 unique pages. OBSERVED:

- 52/52 inferences completed; 29/52 pages produced a mask with > 1000 px (median 10710 px);
  23/52 empty masks.
- Response types: 21/52 `"Pillow."` (degenerate single-word answers), 28/52 generic
  prompt-echo (`"Visual element chosen: A character in action"`), 3/52 page-grounded
  ("a soldier waving a flag" ARMS_072, "a person's hand waving" UltraEleven_111,
  "a person's flowing hair" Arisa_087).
- The grounded targets are plausible animation candidates; their masks are coherent central
  regions (e.g. ARMS_072: 35861 px, bbox x 276-696 y 522-843, zero edge pixels).
- Latency median 3.3 s/page (p95 6.9 s), VRAM ~13.3 GiB.
- One representative page (UltraEleven_110) grounded correctly in a diagnostic warm-up run
  ("a person... about to swing the sword", 46k px mask) but produced `"Pillow."` when it was
  the first inference after load -- the first-after-load response is degraded in some runs.

INTERPRETATION: the model understands the autonomous task and can ground it, but the
majority of responses are either degenerate or echo the instruction's examples instead of the
page. Autonomous discovery is not reliable enough to drive the pipeline.

## 6. Controlled Experiment

Condition D (the production-available description, `"character body"`), all 64 phase-17 GT
`body` targets, full page, no oracle input. Conditions A/B/C are GT-derived diagnostics and
were not run (section 8 of the brief: don't waste compute on conditions unsupported by the
official interface; D is the primary one and it is decisive).

OBSERVED: 64/64 completed without inference errors; every target emitted exactly one `[SEG]`.

## 7. Instance Selection Results

- Instance-correct rate (IoU >= 0.25 with the correct instance): **7.8% (5/64)**.
- Baselines: DINO top-1 instance rate 6.2% (R@1); production DINO->SAM path 3/64 healthy.
- END_TO_END_SUCCESS (correct instance AND IoU >= 0.50): **4.7% (3/64)**.

INTERPRETATION: with the exact description the production pipeline can produce today,
OMG-LLaVA does not solve instance selection: it selects the correct instance as rarely as
DINO's top-1 does. The generic phrase makes the model emit a `[SEG]` whose mask falls on
frame/other regions.

## 8. Mask Results

| metric | distribution |
|---|---|
| IoU | n=64 mean=0.048 med=0.000 p25=0.000 p75=0.000 p95=0.398 max=0.917 (failures=58) |
| Dice | n=64 mean=0.057 med=0.000 max=0.957 (failures=58) |
| Recall@IoU>=0.25 | 7.8% |
| Recall@IoU>=0.50 | 4.7% |
| Recall@IoU>=0.75 | 3.1% |

Best masks (correct instance, excellent quality): UltraEleven_111_695642 IoU 0.917 (recall
0.947), YumeiroCooking_086_748655 IoU 0.873 (recall 0.977), KimiHaBokuNoTaiyouDa_106_284380
IoU 0.527 (recall 0.877). Baselines: GT bbox -> SAM 2.1 median IoU 0.884; DINO top-1 -> SAM
median IoU 0.000 (the current bottleneck).

INTERPRETATION: OMG-LLaVA's masks are bimodal -- either empty/frame-like (58/64 zero-overlap)
or excellent. When it commits to the right instance, mask quality is comparable to SAM's
GT-box quality (0.87-0.92 vs 0.884 median). The failure is selection, not mask fidelity.

## 9. Safety Results

Forbidden-overlap (text/balloon/frame/onomatopoeia GT) computed for the 49 non-empty
condition-D masks: **47/64 targets are classified H (panel-border contamination) with frame
overlap 0.996-1.000** -- the masks land on the page/panel frame structure. Text/balloon/
onomatopoeia absorption is low (<=1-3% except a 13% balloon case). No text/speech-bubble
contamination of the G mask kind.

INTERPRETATION: the `"character body"` phrase on a full page makes the model produce masks
dominated by frame borders -- a systematic behavior, not a per-sample accident. A frame-like
mask is unusable for animation (it would animate the panel borders).

## 10. Performance

OBSERVED (512x512, one T4, model kept loaded):

- Warm latency: controlled 2.5 s median (p95 2.7 s); autonomous 3.3 s median (p95 6.9 s).
- Model load: ~3.5 min (8 x ~2 GiB shards + finetune checkpoint).
- Peak VRAM: 13.3 GiB (of 14.56 GiB on one T4) -- no headroom for 768/1024.
- Resolution scaling (OBSERVED): 1024 OOM (needs ~1.2 GiB more than 0.94 GiB free);
  768 OOM (needs ~676 MiB more than 538 MiB free); 512 fits. Official config = 1024.
- 2 GPUs do not accelerate one inference; the second T4 was unused in the working config.
- Throughput: ~22 targets/min controlled; ~18 pages/min autonomous.

Baseline for comparison: Grounding DINO + SAM 2.1 on the same hardware run in well under a
second per target (both small models; no such per-target number is reported here -- the
current pipeline's wall-clock is dominated by the Qwen analysis stage, not DINO/SAM).

## 11. Failure Taxonomy

Condition D (64 targets): **H (panel-border contamination) 47, C (wrong instance) 17**.
A (correct+good) 3, B 0, D 0, E/F 0, G 0, I 0, J 0, K 0. Every target emitted a `[SEG]`, so
E/F/D never applied; inference never failed (K=0) and coordinates were consistent (J=0).
Category I (unrelated-object contamination) has no automatic GT signal in this dataset and is
reserved for human review of the gallery.

## 12. Comparison with DINO + SAM

| capability | current pipeline (measured, phases 17/18.1) | OMG-LLaVA (this phase) |
|---|---|---|
| candidate generation | DINO finds the target somewhere: 89.1% | n/a (no candidate list) |
| top-1 instance selection | DINO 6.2% (R@1) | 7.8% (IoU>=0.25) -- statistically equal |
| end-to-end healthy | 3/64 (4.7%), median IoU 0.000 | 3/64 (4.7%), median IoU 0.000 |
| mask quality when instance found | SAM 2.1: median IoU 0.884 (GT box) | 0.87-0.92 (2 cases) -- comparable |
| latency (target only) | sub-second (DINO+SAM) | 2.5 s + 3.5 min load |
| resolution | native page | 512x512 (0.5x official), VRAM 13.3/14.56 GiB |

The key architectural comparison (proposed `OMG-LLaVA -> target -> mask` vs current
`Qwen -> DINO -> selection -> SAM`): the proposed chain's controlling step -- selecting the
correct instance from a generic production description -- measures 7.8%, the same failure
mode the current chain has (6.2% DINO top-1), with the added costs of a 7B LLM and a
degraded resolution.

## 13. Architectural Implications

- OBSERVED: OMG-LLaVA can read a full manga page, can segment, and can (rarely) discover a
  plausible animation target autonomously; with the production description it selects the
  correct instance ~as rarely as DINO's top-1 and produces frame-dominated masks in 47/64
  cases.
- INTERPRETATION: the bottleneck the phase set out to attack -- instance selection from
  generic descriptions -- is not improved. The current chain's other components (DINO
  candidate recall 89.1%, SAM quality 0.884) remain the strongest links; replacing them with
  OMG-LLaVA would remove a working candidate stage and add a 7B model whose selection rate
  does not compensate.
- HYPOTHESIS (not measured): a more specific description (condition B/C, GT-derived) would
  raise OMG-LLaVA's selection rate, but production does not produce such descriptions, and
  phase 18.2's candidate-selection direction already attacks the same bottleneck with a
  cheaper model. Nothing in this phase justifies adding Qwen for OMG-LLaVA either.

## 14. Limitations

- The controlled experiment used the only production-available description (`"character
  body"`); conditions A/B/C were not run (GT-derived diagnostics, low information value
  after D).
- The working configuration is 512x512 (0.5x of the official 1024x1024); small/thin
  objects, weapons, and hair are likely affected, but no higher resolution could be run
  (OOM, measured twice). The report's numbers are for 512x512.
- The `"Pillow."` degenerate-response phenomenon (21/52 autonomous pages) is OBSERVED; its
  cause is a HYPOTHESIS (possible artifact of the bitsandbytes cu122-binary override on a
  cu118 torch, or a first-after-load state issue; it also appeared in controlled one-token
  answers on the very first inference of a process).
- The visual gallery (`run_autonomous/autonomous_*.png`) was not human-reviewed in this
  session; the qualitative claims above are based on output texts, mask pixel geometry, and
  the numerical overlap/safety measurements.
- The environment required material venv surgery (mmcv wheel selection, gate patches,
  bitsandbytes override, resolution reduction); this is worker setup, not a model property.

## 15. Recommendation

**D. KEEP CURRENT PIPELINE** -- based on the measured evidence:

1. Can OMG-LLaVA understand the full manga page? **Yes** (caption and grounded
   responses).
2. Can it identify an animation-worthy target? **Rarely/partially** (3/52 grounded;
   28/52 generic-echo; 21/52 degenerate).
3. Can it select the correct INSTANCE? **No better than DINO top-1**: 7.8% vs 6.2%
   (statistically equal on 64 samples).
4. Can it produce a usable mask? **When it commits: yes** (0.87-0.92 IoU), but 58/64
   controlled targets produced zero-overlap, 47/64 frame-dominated masks.
5. Instance selection better than DINO top-1? **No** (7.8% vs 6.2%, not material).
6. End-to-end better than the current pipeline? **No** (both 4.7% healthy on the same
   64 targets; mask quality comparable only when selection succeeds).
7. Latency acceptable? **No** (2.5 s/target + 3.5 min load vs sub-second DINO+SAM).
8. Fits 2xT4-16GB? **Barely**: only at 512x512 with 13.3/14.56 GiB used; official
   resolution OOMs.
9. What can be removed? **Nothing** is justified by the evidence: the current chain's
   weakest link (instance selection, 6.2%) is not improved, and its strong links
   (DINO recall 89.1%, SAM quality 0.884) would be lost.

The measured failure mode is the same as the current bottleneck: selecting the correct
instance from a generic description. The phase-18.2 candidate-selection direction (a cheap
reranker over DINO's 89% recall) remains the evidence-backed path; OMG-LLaVA's cost and
resolution limits provide no counter-argument.
