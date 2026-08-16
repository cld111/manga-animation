# Phase 22 A/B: Panel vs. Full-Page VLM Description

## Question

On one page (`wind_breaker_sprint`, 4 panels, 52 grounded candidates), does the VLM
produce equal-or-better object descriptions from a single full-page call (mode B) versus
the production per-panel calls (mode A)?

- **A (panel mode, production path)**: `run_pages` describes each panel's candidates in its
  own scene crop -- one Qwen call per panel, bboxes in crop-local coordinates.
- **B (full-page mode)**: ONE Qwen call on the whole page (2048px long edge, `max_new_tokens=16000`)
  with all 52 candidates' bboxes in page coordinates, results split back per panel and
  checkpointed, then the standard pipeline resumes from checkpoints.

Same page, same DINO grounding, same labels, same random seed.

## Setup

- Worker: fresh 2xT4 Kaggle session, transformers 5.0.0, Qwen3-VL-4B-Instruct fp16
  (`device_map="auto"`, ~8.5 GiB spread across both cards), DINO/SAM/LaMa pinned to GPU1
  (Qwen + aux models on one T4 OOM'd during panel render -- real failure).
- `scripts/run_phase22_ab_test.py`, results in `outputs/experiments/phase22_ab_test.json`.

## Results

| | **A (panel)** | **B (full page)** |
|---|---|---|
| elapsed_s | 1015.7 (~17 min) | 1581.8 (~26 min) |
| vlm_calls | 5 (4 panels + 1 recovery) | 1 |
| n_grounded | 52 | 52 |
| n_parsed | 52 | 52 (all unparseable) |
| panels | PASS / REJECTED / PASS / PASS | REJECTED x4 |

Mode A Qwen decode per panel: ~3.5-4 min (batch starts 20:26:29 / 20:30:02 / 20:33:54 /
20:37:36); full A elapsed 1015.7s including SAM/plan/render.

## Finding

Mode B failed closed: Qwen3-VL-4B's single full-page JSON covered only **10 of 52**
candidate boxes ("batch answer covers 10 of 52 candidate boxes"), so the batch parse
failed and every candidate was recorded unparseable -> all four panels REJECTED. The model
cannot emit a complete, schema-valid description for 52 bboxes in one 2048px call on a T4.

## Conclusion

Panel-mode (A) remains the production default. Full-page single-call description is not
viable at this candidate count on Qwen3-VL-4B. Qwen3-VL-4B fp16 per-panel (~3.5-4 min/panel)
beats the Phase 20/21 Qwen3-VL-8B fp16 sharded run (1222s) and Phase 22 int8 run (1818s)
on the same page while producing 3/4 PASS panels.

## Operational fixes landed in this phase

- Pipeline join timeout raised 600s -> 7200s (guard, not perf budget): a single VLM
  instance describing a 4-panel page exceeds 10 min on a T4.
- transformers 5.0.0 lazy-import race: pre-load `Sam2Model`/`Sam2Processor` in the main
  thread before spawning pipeline workers (spurious "cannot import name 'Sam2Model'").
- A/B script: reload stage-owned DINO before mode B grounding; pin DINO/SAM/LaMa to GPU1.