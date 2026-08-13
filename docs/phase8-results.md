# Phase 8 results: end-to-end production validation

Status: **in progress** — infrastructure and local groundwork complete and verified; real-model
E2E execution against the golden dataset on a live Kaggle GPU worker is the remaining work. This
document is being written incrementally, in place, as real evidence lands (same convention as
`docs/phase7-results.md`) — every number below is checked against an actual test run, downloaded
artifact, or quoted log line, not asserted from memory.

## 1. Scope

Phase 8's brief: "Prove that the complete system can reliably transform a real manga page into a
visually valid, seamless looping H.264 animation without manual intervention between pipeline
stages." System-level validation, not new model complexity. See
[ADR 0014](decisions/0014-phase8-e2e-validation.md) for the full design of everything below.

## 2. Pre-Phase-8 audit findings

Before any Phase 8 change, the repository was audited against `docs/phase7-results.md`'s own
claims and against the actual implemented pipeline:

- `git status` clean on `phase-6-wip`, `uv run pytest` **442 passed, 2 deselected**,
  `uv run pytest -m slow` **2 passed**, `uv run ruff check .` clean, `uv run mypy src` clean (41
  files) — exact match to `docs/phase7-results.md`'s own reported baseline, no discrepancy found.
- **The Phase 8 brief's own "expected conceptual pipeline"** (`candidate pages -> real color
  detection -> EXACTLY 10 ACCEPTED COLOR PAGES -> Qwen2.5-VL -> Florence-2 -> ...`) **does not
  match the real, implemented pipeline.** No candidate-page-selection or color-detection gate
  exists anywhere in this repository (the source image is loaded directly,
  `pipeline/orchestrator.py`); grounding uses **Grounding DINO** (`grounding-dino-swin-l`), not
  Florence-2 (`grounding/client.py`). Per the brief's own instruction ("do not assume every arrow
  exists merely because the architecture says it should"), Phase 8 is built against the real
  pipeline (`docs/pipeline.md`), not the brief's sketch — this discrepancy is disclosed, not
  silently reconciled.
- No JSON artifact or rendered video from any real GPU run (Phase 3 through 7) exists in this
  local checkout (`outputs/` is git-ignored per ADR 0002, and none happened to be downloaded to
  this exact path/session) — Phase 7's real-model numbers could not be independently
  re-verified from local artifacts, only from `docs/phase7-results.md`'s own prose and commit
  history. This is expected under ADR 0002/0003 (remote compute is disposable, outputs are never
  canonical), not a defect, but it means Phase 8's own real evidence (section 6+ below) is
  gathered fresh, not inherited.

## 3. Implemented changes (local, no GPU required)

| Area | Summary |
| --- | --- |
| `pipeline/types.py`, `rendering/encode.py` | `LoopMetrics` (pixel-level + new SSIM-based structural signal), exposed on `RenderResult.loop_metrics`; `compute_loop_metrics` made public |
| `evaluation/metrics.py` | `E2EStatus`, `classify_outcome`, `StatusBreakdown`, `EvaluationReport.status_breakdown` |
| `evaluation/schemas.py` | `FailingStage` (fixes a real latent bug — see below), `RenderSummary`/`LoopMetricsOutcome`, `PageRunOutcome.render_summary`, `schema_version` meaning 3 |
| `evaluation/dataset.py` | `GoldenCategory`, `GOLDEN_DATASET_CATEGORIES`, `EvalSample.golden_categories`, `golden_category_coverage`, `uncovered_golden_categories` |
| `configs/phase3_3_eval_dataset.yaml` | Formalized as the Phase 8 golden dataset: `golden_categories` added to all 7 samples, header note documents 2 real coverage gaps |
| `scripts/run_phase3_3_evaluation.py` | Populates `render_summary`, prints per-sample `E2EStatus` and per-report `status_breakdown`, `schema_version` bumped to 3 |

**A real, latent bug found and fixed during this work** (not merely documented around):
`PageRunOutcome.failing_stage` was typed `Stage | None` — `Stage` is the closed 8-value
`Literal` of real `PipelineStageError.stage` values. But
`scripts/run_phase3_3_evaluation.py`'s own `except Exception` catch-all constructs
`PageRunOutcome(failing_stage="unexpected", ...)`, a string outside that `Literal`. This would
have raised `pydantic.ValidationError` — discarding the original exception — the first time a
real run ever hit a bare (non-`PipelineStageError`) exception. Never actually observed as a
crash (no real run has hit that path yet), only found by reasoning about the type — the same way
`docs/phase7-results.md` section 13 originally surfaced it as a disclosed-but-unfixed `mypy`
finding. Fixed via a new `evaluation.schemas.FailingStage = Stage | Literal["unexpected"]`,
scoped to evaluation reporting only (`pipeline.types.Stage` itself is untouched and stays exact).

## 4. Local test/lint/type results

```
uv run pytest -q
```
**472 passed, 2 deselected** (up from Phase 7's 442; +30 new tests: 5 rendering/SSIM, 17
evaluation status-classification, 4 golden-dataset coverage, 4 script-helper regression tests —
see section 3's table for what each protects).

```
uv run pytest -m slow
```
**2 passed** (`tests/test_performance.py`, unmodified by Phase 8).

```
uv run ruff check .
```
Clean.

```
uv run mypy src
```
Clean, 41 source files (unchanged file count — Phase 8 extended existing modules, added no new
`src/` package).

## 5. Golden E2E dataset

`configs/phase3_3_eval_dataset.yaml` (unchanged sample set — reused per the brief's own "reuse
existing evaluation assets" instruction, not duplicated into a second file) now carries
`golden_categories` per sample, cited against real, already-documented evidence:

| Category (Phase 8 brief item) | Covered by | Evidence |
| --- | --- | --- |
| 1. single obvious animatable object | `sample_page_01`, `sample_page_02`, `eval_weapon_effects` | Phase 3.2, ADR 0010 Phase 5 audit |
| 2. multiple animatable objects | `phase3_action_page`, `verified_action_1`, `verified_action_2` | Phase 7 real multi-object renders (`docs/phase7-results.md` 6.2) |
| 3. partially occluded objects | **none** | real, disclosed gap |
| 4. near panel/image boundary | `verified_action_1` (incidental only) | Phase 7 dropped-candidate edge-margin rejection |
| 5. complex backgrounds | `phase3_action_page`, `eval_weapon_effects` | dataset notes (dark action panels) |
| 6. weapon/effect | `phase3_action_page`, `eval_weapon_effects`, `verified_action_1` | ADR 0011, Phase 7 |
| 7. rotation | `phase3_action_page`, `eval_weapon_effects`, `verified_action_1` | ADR 0011, ADR 0008 |
| 8. translation | `sample_page_01`, `sample_page_02`, `phase3_action_page`, `verified_action_2` | Phase 3.2, Phase 7 |
| 9. scale/deformation | **none** | real, disclosed gap (only ad hoc, uncommitted Phase 7 evidence exists, never added as a dataset sample) |
| 10. should-NOT-animate | `eval_static_dialogue` (confident); `sample_page_01`/`sample_page_02` (uncertain, not confident controls) | dataset ground truth |

Locked in as a checkable fact, not only this table:
`tests/test_evaluation.py::test_real_golden_dataset_has_exactly_the_two_disclosed_coverage_gaps`.

## 6. Real-model E2E execution

*Pending — to be filled in after execution against the live Kaggle GPU worker provided for this
session.*

## 7. Visual QA findings

*Pending.*

## 8. Loop validation on real output

*Pending.*

## 9. Known limitations

- Two real, disclosed golden-dataset coverage gaps: `partially_occluded_object`,
  `scale_or_deformation` (see section 5).
- `object_near_boundary` coverage is incidental (one dropped-candidate rejection reason), not a
  sample deliberately chosen to exercise the category.
- The new SSIM-based loop check is verified against a synthetic test sequence
  (`tests/test_rendering.py`), not yet against a real rendered page's own periodic motion — that
  requires section 6's real execution.
