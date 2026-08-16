# ADR 0023: Per-GPU VLM Instances (int8) With a Worker Pool

Status: Accepted

Supersedes: the `device_map="auto"` fp16 sharding of the VLM (ADR 0005's Qwen path) as the
runtime default, for the Phase 22 int8 candidate. The fp16 sharded client remains available
as `qwen3-vl-8b`.

## Context

Phase 20/21 ran Qwen3-VL-8B as ONE fp16 model sharded across the session's 2xT4
(`device_map="auto"`): each decoded token crosses the GPU boundary, and the real run
(phase20_21_e2e_single.json) showed both cards at 37%/61% utilization during the Qwen
stage, with Qwen busy 1075.7 s of a 1222.3 s run (88%). The other GPU-level alternative --
one full fp16 8B instance per T4 -- is impossible: 16 GiB of weights exceed a 14.56 GiB
usable card.

bitsandbytes int8 quantization halves the weights to ~9.5 GiB, which fits one T4 with
headroom (the full co-residency budget: 9.5 + DINO 1.8 + SAM 0.6 + LaMa 0.2 + activations).
Two independent instances then make BOTH cards do useful VLM work concurrently, and panels
can be split between them.

## Decision

- The runtime VLM candidate becomes `qwen3-vl-8b-int8` (`configs/default.yaml`): a
  `Qwen3VLInt8Client` per CUDA device, each loading the same model directory with
  `BitsAndBytesConfig(load_in_8bit=True, pre_quantized=True)` and
  `device_map={"": "cuda:N"}` -- one complete instance per card, never sharded.
- The model directory must be PRE-QUANTIZED: transformers 5.0.0 materializes the full
  fp16 checkpoint on the target device before quantizing, which OOMs a single T4
  (verified on the worker). The fp16 repo checkpoint is converted once
  (load fp16 int8 with `device_map="auto"` across both cards -> `save_pretrained` ->
  a ~9.5 GiB int8 safetensors) and the client loads that; loading takes ~3 s and
  allocates ~9.3 GiB on its card.
- The object-description stage runs as a WORKER POOL (one worker per instance) inside the
  Phase 21 panel pipeline: all VLM workers consume the shared description queue and feed
  the shared segmentation queue. The pipeline's sentinel cascade is generalized with
  per-queue `end_expected`/`end_sent` counts so the pool terminates cleanly (each upstream
  worker emits one sentinel per downstream consumer; a consumer stops only after seeing
  one sentinel per producer).
- The per-panel `descriptions.json` checkpoint write is guarded by a lock (one file per
  page, several concurrent writers).
- `run_pages`/`run_page_panels` accept either a single `VLMClient` or a sequence; one
  client keeps the exact Phase 20/21 behavior (tests unchanged).

## Consequences

- Positive: both cards process panels concurrently (worker pool); no cross-GPU token
  traffic; int8 decode is typically faster than fp16 on T4 (less memory bandwidth);
  loading a pre-quantized instance is ~3 s vs ~300 s for the fp16 shard.
- The two instances are independent and unsynchronized: panel-to-instance assignment is
  non-deterministic, but per-panel results are identical regardless of which instance
  processed a panel (the same per-panel stage function), so run outputs stay
  deterministic except for the model's inherent non-determinism.
- Pre-quantization must happen once per worker/deploy (documented in the run script); the
  fp16 sharded client remains for comparisons and single-GPU fallback.
- GPU memory split: the VLM instances are run-level resident (ADR 0021), but DINO/SAM/LaMa
  are STAGE-OWNED instead of co-resident (ADR 0021's original "all models" reading now
  excludes them): each small model loads when its pipeline worker starts and unloads when
  the worker finishes. A full int8 Qwen per GPU needs the card's headroom for its KV cache
  and prefill -- with DINO+SAM+LaMa (2.6 GiB) permanently co-resident on card 0, the real
  run OOM'd (CUDA out of memory, 356 MiB alloc) on every Qwen call. Stage ownership frees
  card 0 before Qwen decode: DINO (grounding) finishes and unloads before the first Qwen
  panel; SAM and LaMa run on panels already described, overlapping Qwen's decode of LATER
  panels, but they are small and their activation peaks are short.

## Evidence

- Worker verification (this phase): fp16-with-quantization OOMs one T4; the pre-quantized
  int8 safetensors loads on `cuda:1` in ~3 s using 9559 MiB with `device_map={"": "cuda:1"}`.
- tests/test_lifecycle.py `test_run_pages_vlm_worker_pool_splits_panels_across_instances`:
  two VLM instances split the 4 panels (each called at least once, 4 total, one teardown
  each) and every panel still renders PASS.
- The real 2xT4 run will be recorded in the phase results doc.
