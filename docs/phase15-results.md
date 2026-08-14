# Phase 15 Results: Post-Phase-14 GPU Regression & Stability

Phase 15 validated the Phase 14 stage-level model lifecycle across multiple real pages and
repeated GPU runs, on a real 2xT4 Kaggle worker plus a 1xT4 smoke lane. It is a validation /
hardening phase: the architecture was not redesigned. The only code changes are the two
lifecycle teardown defects the adversarial review found and the Phase 15 GPU validation
script + `docs/kaggle-jupyter.md` (the verified remote-workflow guide).

## Scope

Validated the production stage order with deterministic model lifecycle:

```text
page -> panels -> analysis -> grounding -> validation -> segmentation -> mask_semantics
  -> animation -> reconstruction -> compositing -> rendering
```

Do NOT redesign: lifecycle, panel detection, STATIC, mask semantics, reconstruction,
compositing, segmentation. Only fix issues demonstrated by this validation.

## Environment

- Worker: Kaggle Jupyter session, `2xTesla T4` (15360 MiB each, 14912 MiB usable), torch
  2.10.0+cu128, Python 3.12.13, transformers 5.0.0, ffmpeg present.
- Models: Qwen2.5-VL-7B-Instruct, Grounding DINO, SAM 2.1, LaMa (Phase 14 local-dir layout).
- Connection: the Kaggle "VSCode Compatible URL" used as a Jupyter REST + kernel-websocket
  transport (documented in `docs/kaggle-jupyter.md`); verified by executing Python on the
  kernel, not by a browser GET.

## Multi-page test set (6 pages, 17 panels, one 2xT4 session)

| Page | Panels | Characteristics | v2 statuses |
|---|---|---|---|
| `villainess_ending_scuffle` | 4 | Phase-14 page; known segmentation + mask_semantics rejection | STATIC, STATIC, REJECTED(mask_semantics), REJECTED(segmentation) |
| `space_monster_hypersenses` | 1 | sparse scene, simple bg | REJECTED(segmentation) |
| `sss_hunter_gladiator` | 5 | extreme aspect (12.2), dense, crowded | PASS, PASS, REJECTED(grounding), PASS, STATIC |
| `villainess_ending_scuffle` (repeat) | 4 | repeated execution in same session | identical to run 1 |
| `wind_breaker_finish` | 7 | dense, complex bg, known defect present | REJECTED(mask_semantics), 5x STATIC, PASS |
| `reality_lie_office` | 1 | animation_possible="no" | REJECTED(validation) |

Panel outcomes varied as expected (VLM nondeterminism); every safety gate fired only where it
should. STATIC panels produced no video (unchanged). The known `wind_breaker_finish` visual
defect remains a pre-Phase-14 known defect and is not attributed to this phase.

## Repeated execution (same session)

- `villainess_ending_scuffle` run 1 and run 4 (repeat, same process, models reloaded per
  stage): identical per-panel statuses `STATIC/STATIC/REJECTED/REJECTED` with the same
  failure stages (mask_semantics on panel_003, segmentation on panel_004). No stale GPU
  state, hidden model references, or behavior difference caused by prior execution.
- A separate resume test (same out_dir twice) reused the completed STATIC panels on the
  second invocation (second run elapsed 103.3 s vs 157.4 s first, both models reloaded once
  per stage for the still-pending REJECTED work) and ended at the same ~73/9 MiB allocator
  state.

## VRAM lifecycle result

Per-page release logs on all 6 pages (each stage released exactly its own footprint):

| Stage | Released |
|---|---|
| analysis (Qwen) | 15816-15818 MiB |
| validation (Qwen) | 15816-15818 MiB |
| mask_semantics (Qwen) | 15816-15818 MiB |
| grounding (DINO) | 892 MiB |
| segmentation (SAM) | 281-284 MiB |
| reconstruction (LaMa) | 196-197 MiB |

- `torch.cuda.memory_allocated` after every page returned to **73.1 / 9.1 MiB** per device;
  17 of 130 timeline samples had both GPUs fully released (page boundaries); no progressive
  growth across the 6-page sequence.
- Timeline peak allocated: **8725 MiB (8.7 GiB) on one T4**, matching the Phase 14 acceptance
  number.
- Final state after all 6 pages: allocated 73.1/9.1 MiB, live CUDA tensors 2 (128 MiB), no
  lifecycle-related CUDA OOM.

## Failure isolation

`--inject-grounding-failure 2` wrapped the real Grounding DINO client to raise a raw
`RuntimeError` ("simulated CUDA OOM") on the 2nd detect call:

- The failing panel became **ERROR** and was isolated; the other three panels still processed
  (STATIC, STATIC, and a REJECTED from the segmentation safety gate).
- All model stages released on the exception path too: analysis 15816 MiB, grounding 892 MiB,
  validation 15816 MiB, segmentation 282 MiB, reconstruction 196 MiB.
- Final allocated 73.1/9.1 MiB, live tensors 2. Stage cleanup works on failure as on success.

(The first injected run surfaced a `model_id` AttributeError from the test script's wrapper,
not from the pipeline; fixed in the script and re-run clean.)

## Resumability

Manifest-based resume on the real worker reused completed STATIC panels (second run skipped
re-analysis of those panels, 103.3 s vs 157.4 s) and re-processed only the not-reused
REJECTED panels; completed PASS panels were reused when present; the resume path showed the
same ~73/9 MiB allocator return and no VRAM accumulation.

## Safety gates

All Phase 8-12 gates observed firing correctly on real GPU without modification:

- segmentation asymmetric edge-touch rejection (villainess panel_004, space_monster_hypersenses)
- mask_semantics PRIMARY rejection stays REJECTED (villainess panel_003, wind_breaker panel_001)
- grounding no-detection PRIMARY rejection (sss_hunter panel_003)
- target-validation rejection (reality_lie_office, wind_breaker panel_005)
- STATIC unchanged (no video invented)

## 1xT4 result

`CUDA_VISIBLE_DEVICES=0` smoke run on `space_monster_hypersenses` (424 s): Qwen2.5-VL 7B fit
on a single T4 (allocator peak ~14.1 GiB, CPU-offloaded parameters, one GPU visible, confirmed
`single_t4=True`), the stage lifecycle released each model (Qwen ~12.1 GiB per VLM stage),
the panel was REJECTED by the same segmentation gate as on 2xT4, and final allocated was
73.1 MiB with 2 live tensors. The 2xT4 primary validation is unaffected.

## Performance impact

Total 2xT4 multi-page run (6 pages, 17 panels): 675.6 s. Per page: 76-171 s. Model loads
dominate (Qwen ~9 s load + inference per stage; DINO/SAM/LaMa small). No unreasonable
lifecycle overhead was found; the structural benefit (one model load per stage per page vs
per panel) is unchanged from Phase 14.

## Visual regression

Four PASS videos produced by the new stage-level runner (3 sss_hunter panels, 1
wind_breaker panel) were downloaded and checked numerically: correct panel crop (video
dimensions == scene_crop_bbox dimensions), localized motion (0.3-13% of pixels changed over
the loop), and seamless loop closure (wrap-step mean-abs-diff ratio 1.04-1.10 vs ordinary
adjacent step). No duplicate original object, missing object, or broken reconstruction was
evidenced. The known historical defects (wind_breaker_finish PRIMARY, marika_love_meter) were
not re-investigated (pre-Phase-14, unchanged by Phase 14 orchestration).

## Fixes made (Phase 15)

Adversarial review found two real teardown-path defects in `ModelStage`
(`src/manga_animation/pipeline/lifecycle.py`), both reproduced locally and on the review:

1. **HIGH**: a raising `client.unload()` escaped the with-block, masked the stage body's own
   exception, and skipped `release_device_memory()` (the gc.collect -> empty_cache ->
   ipc_collect Phase 14 leak fix). Now: unload failure is logged when unwinding a stage
   exception and still fails a successful stage (fail-closed); `release_device_memory()` runs
   in a `finally`.
2. **MEDIUM**: a `client.load()` failure in `__enter__` never runs `__exit__` (Python
   semantics), leaving `_active=True` (poisoned stage object) and any partial CUDA blocks
   un-released. Now: the guard resets and the deterministic release runs before re-raising.

4 new regression tests in `tests/test_lifecycle.py`.

Also: `scripts/run_phase15_gpu_regression.py` (multi-page memory-lifecycle + resume + injected
failure + 1xT4 lane), `docs/kaggle-jupyter.md` (verified remote workflow), and a Phase 15
harness fix (removed the gc.get_objects live-tensor scan from the 5 s sampler thread -- it
raced mid-inference and produced a real `SystemError: bad argument to internal function`
inside pipeline panels; the scan now runs only at page boundaries).

## Tests

- `uv run pytest`: 596 passed, 2 deselected.
- `uv run ruff check .`: clean.
- `uv run mypy src`: clean.

## Known limitations

- 1xT4 was exercised as a smoke lane only (Qwen CPU-offloaded, ~14.1 GiB peak); it is not a
  primary-validated config, and its inference is slow.
- Panel outcome nondeterminism from the Phase 12 mask_semantics VLM is unchanged and out of
  scope (documented separately); a REJECTED panel can flip verdict across runs, which is why
  repeated-run statuses can differ between the two full runs of the session (v1 had a harness
  artifact; v2 was clean and self-consistent).
- The remote checkout and models are worker-local; a fresh Kaggle session must re-clone,
  re-install `.[ml]`, re-download models, and re-fetch pages (see `docs/kaggle-jupyter.md`).
- Wall-clock differs across identical pages because VLM generation length is nondeterministic.

## IMPORTANT NEGATIVE FINDINGS

- No lifecycle-related CUDA OOM occurred across 6 pages / 17 panels / repeated runs / 1xT4.
- No progressive VRAM growth (allocator returned to ~73/9 MiB after every page; timeline peak
  bounded at 8.7 GiB on one T4).
- No cross-page state leakage (identical repeated-run statuses; models released after every
  stage on every page).
- The two teardown defects fixed here were the only code findings; no orchestration logic
  regression was found.
