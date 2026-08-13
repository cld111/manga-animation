# Phase 8 results: end-to-end production validation

Status: **substantially complete** — infrastructure implemented and tested locally, a real
GPU E2E run executed against the full golden dataset, real artifacts downloaded and
independently re-verified, and two real, previously-undiscovered visual defects found by direct
inspection of real rendered output. This document was written incrementally, in place, as real
evidence landed (same convention as `docs/phase7-results.md`) — every number below is checked
against an actual test run, downloaded artifact, or quoted log line, not asserted from memory.

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

Executed on a live Kaggle Jupyter GPU worker (2x Tesla T4), reached via the project's established
non-browser Jupyter REST/kernel-WebSocket transport (a dedicated kernel started for this session's
work, not the user's own already-connected notebook kernel — same convention as
`docs/phase7-results.md`). Real model versions: `torch==2.13.0+cu130`, `transformers==5.15.0`,
real `Qwen2.5-VL-7B-Instruct`, real `grounding-dino-swin-l`, real `sam2.1-hiera-base`, real LaMa
(`lama-large`) — resolved from the run's own recorded `model_variants`/`environment` metadata, not
asserted.

`uv sync --extra dev --extra cv --extra video --extra ml` (fresh clone, commit `e5a4070`), then
`uv run pytest -q` **472 passed, 2 deselected** and `uv run ruff check .` clean on the remote
worker too (mypy found one pre-existing, unrelated finding — see section 10). All 7 golden-dataset
samples fetched (5 via their `fetch_script`s; the two `examples/verified_action/*.png` samples, no
fetch script, uploaded via the Jupyter Contents API from the local files, same as
`docs/phase7-results.md`'s established method).

`uv run python scripts/run_phase3_3_evaluation.py --env kaggle` — all 7 samples, both
`analysis_mode`s, plus the nondeterminism check (3 repeated `analyze_page` calls per sample).
Wrote `outputs/experiments/phase3_3_evaluation_20260813T103143Z.json` (27KB, downloaded locally
via the Jupyter Contents API, `git_commit: e5a407043b943a809f670f87d9a9fd632109d318` — confirmed
matching the pushed commit under test). The run took ~44 minutes; the client-side WebSocket
connection was lost partway through (a proxy-layer timeout, not a kernel crash — the remote
kernel kept executing independently, confirmed by polling `/api/kernels/<id>` until
`execution_state` returned to `idle`), so the live stdout stream was not fully captured, but the
JSON artifact — the actual evidence — was written successfully and is unaffected.

### 6.1 Page-level report

`usable_target_rate` 5/7 (71.4%), `end_to_end_completion_rate` 1/7 (14.3%),
`secondary_object_render_rate` 1/1 (100%), `micro_object_render_rate` 0/0 (n/a).
**`status_breakdown` (new Phase 8 field, computed and populated for real for the first time):
PASS=1, PASS_WITH_FALLBACK=0, REJECTED=3, ERROR=3.**

### 6.2 Panel-level report

`usable_target_rate` 6/7 (85.7%), `end_to_end_completion_rate` 3/7 (42.9%),
`secondary_object_render_rate` 3/7 (42.9%), `micro_object_render_rate` 2/2 (100%),
`panel_detection_multi_panel_rate` 4/7 (57.1%).
**`status_breakdown`: PASS=3, PASS_WITH_FALLBACK=0, REJECTED=3, ERROR=1.**

Per-sample classification (recomputed locally from the downloaded JSON via
`evaluation.classify_outcome`, not merely copied from the run's own stdout, as independent
verification that the shipped function produces the claimed output on real data):

| Sample | Page mode | Panel mode |
| --- | --- | --- |
| `sample_page_01` | REJECTED (analysis, all-STATIC) | REJECTED (grounding empty) |
| `sample_page_02` | REJECTED (validation rejected) | REJECTED (validation rejected) |
| `phase3_action_page` | **ERROR** (grounding empty; confident `yes` ground truth) | **PASS** (completed, `character_hair`/TRANSLATE) |
| `eval_static_dialogue` | REJECTED (all-STATIC, matches confident `no`) | REJECTED (all-STATIC, matches confident `no`) |
| `eval_weapon_effects` | **ERROR** (validation rejected; confident `yes`) | **ERROR** (validation rejected; confident `yes`) |
| `verified_action_1` | **ERROR** (validation rejected; confident `yes`) | **PASS** (completed, `raised_sword`/ROTATE + 4 real SECONDARY/MICRO objects) |
| `verified_action_2` | **PASS** (completed, `character movement`/TRANSLATE) | **PASS** (completed, identical render — this sample has no real multi-panel structure, so panel mode falls back to page-level analysis, per ADR 0007) |

This is a real, concrete finding about `classify_outcome`'s current design (already flagged as an
open question in ADR 0014): `eval_weapon_effects` is marked ERROR both modes, even though this
sample's own `acceptable_outcome` in `configs/phase3_3_eval_dataset.yaml` explicitly says "an
honest grounding/validation failure (no video) is also acceptable" for this specific sample
(effect-heavy motion cue, not a single easily-prompted object) — `classify_outcome` only consults
the structured `animation_possible`/`ground_truth_uncertain` fields, not this richer free-text
nuance, so it flags a real, expected, previously-repeatedly-documented rejection (byte-for-byte
the same one `docs/phase7-results.md`/ADR 0011 already recorded) as ERROR rather than REJECTED.
Not a bug — a disclosed design limitation now observed for real, not just anticipated.

`phase3_action_page` and `verified_action_1` completing at panel level but not page level
reproduces this project's own established finding (ADR 0007/0011: page-level analysis struggles
on large/extreme-aspect-ratio pages; panel-aware analysis fixes it) — for a third, independent
real session.

### 6.3 Nondeterminism check

Every sample was internally stable this session (`outcome_stable=True`,
`target_category_stable=True`, 3/3 repeated calls agreed, for all 7 samples) — matches this
project's established finding that within-session repeated calls are self-consistent. One new,
disclosed cross-session variation: `sample_page_02` read PRIMARY `weapon` 3/3 times this session —
different from earlier documented sessions' reads of this same page (`character_hair` in the
original Phase 3.2 session, all-STATIC in Phase 3.3/3.3.1) — a third distinct real read for a page
already established (ADR 0009) as cross-session nondeterministic; consistent with, not
contradicting, that existing finding.

### 6.4 Real render evidence

4 real completions produced real MP4s (downloaded, independently re-decoded and re-measured
locally with `cv2.VideoCapture` + this project's own `rendering.compute_loop_metrics` — not
trusting the remote run's self-report):

| Sample (mode) | Resolution | Frames | `wrap_step_within_2x_ordinary` | `wrap_ssim_within_tolerance` | `seamless_loop_verified` |
| --- | --- | --- | --- | --- | --- |
| `phase3_action_page` (panel) | 720x5062 | 96 | True (ord=1.41, wrap=1.50) | True (ord=0.9693, wrap=0.9685) | True |
| `verified_action_1` (panel) | 1100x6614 | 96 | True (ord=0.64, wrap=0.69) | True (ord=0.9843, wrap=0.9837) | True |
| `verified_action_2` (panel) | 1350x1920 | 96 | True (ord=0.83, wrap=0.91) | True (ord=0.9756, wrap=0.9746) | True |
| `verified_action_2` (page) | 1350x1920 | 96 | True (ord=0.83, wrap=0.91) | True (ord=0.9756, wrap=0.9746) | True |

Locally re-decoded numbers matched the remote-computed numbers in `render_summary.loop_metrics`
(the new Phase 8 field) to within float-serialization rounding — genuine, independent
confirmation that `RenderResult.loop_metrics` (previously private, now the shipped public API)
computes correctly on real model output, not only the synthetic test sequence it was unit-tested
against.

## 7. Visual QA findings

Direct visual inspection (not inferred from successful execution or passing numeric checks) of
real downloaded frames — frame 0, frame 24 (mid-cycle), and frame 95 (the true last frame,
adjacent to the loop wrap) for the region a numeric pixel-diff identified as changed between
frame 0 and frame 24, cropped from the actual PNG frame dumps and cross-checked against the same
crop re-extracted from the actual encoded MP4 (not just the pre-encode PNG).

**Two real, previously-undocumented visual defects found:**

1. **Silhouette "ghosting"/doubling — `verified_action_1`, panel mode, mid-cycle (frame 24).** A
   visible semi-transparent duplicate contour, offset from the "real" position — **not confined
   to the hair strands**: independent re-inspection confirms the doubling extends across the
   goggles/eye-patch outline and the jaw/mouth line too, i.e. a large part of the head silhouette,
   not just one lock of hair. This sample's real `object_outcomes` list two distinct
   `character_hair`/`secondary`/`rendered` object ids (`obj_character_hair_4`,
   `obj_character_hair_9`) — two independently-grounded real objects, both legitimately validated
   and animated via TRANSLATE, whose real masks most plausibly partially overlap and/or extend
   onto neighboring face regions; their independent motion produces the visible double-exposure
   at peak displacement. Confirmed present in the actual encoded `output.mp4` (re-extracted the
   same crop from the decoded video, byte-identical to the pre-encode PNG crop), not a PNG-only
   artifact.
2. **Hard vertical seam — `phase3_action_page`, panel mode, mid-cycle (frame 24).** A sharp,
   rigid vertical boundary spanning roughly the left ~15% of the crop's full height, cutting
   through both the background wall texture and the white panel border/gutter line — clearly
   extending beyond the animated hair silhouette itself, suggesting the TRANSLATE-animated
   layer's effective footprint (or its reconstruction hole-fill boundary) leaks beyond the
   intended object region at peak displacement.

**Both defects are absent at frame 95** (the true last frame before the loop wraps back to frame
0) — visually indistinguishable from frame 0 in both cases. **This means the loop-boundary claim
itself (section 8's specific concern) is genuinely not violated by either defect** — both are
real, visible, mid-cycle-only artifacts (present at peak displacement, absent at rest pose),
independent of loop-seam continuity. This is a real, useful distinction: this project's existing
whole-frame `LoopMetrics` (pixel-diff + SSIM) correctly reports `seamless=True` for both renders
(section 6.4) — accurately, for what it measures — but neither metric is designed to catch a
localized mid-cycle artifact averaged into an otherwise-healthy whole-frame signal, exactly the
failure mode the project's own `evaluation` skill already warns about ("mask-edge regions
specifically... not whole-frame metrics, which dilute a small visible seam into an
acceptable-looking average"). Both defects were found only by targeted, mask-region-specific
visual inspection, not by any existing automated numeric check.

**A clean control case**: `verified_action_2` (`character_movement`/TRANSLATE +
`object_interaction`/SECONDARY) showed no ghosting, no seams, no artifacts at frame 24 —
confirming the pipeline can and does produce clean multi-object output, and that the two defects
above are specific to certain real grounding/segmentation instances, not a universal encoding
defect.

Independently re-verified by a fresh `qa-agent` audit — see section 9.

## 8. Loop validation on real output

Covered in sections 6.4 and 7 above: all 4 real completions report `seamless_loop_verified=True`
via both the pre-existing pixel-diff check and the new SSIM-based structural check, independently
reproduced by a fresh local decode of the actual downloaded MP4s. Direct visual inspection of the
true wrap-boundary frame (frame 95 vs. frame 0) for both samples with a mid-cycle defect confirms
no visible discontinuity at the actual loop seam in either case — the loop-correctness acceptance
criterion (F) is met by both numeric evidence and direct visual inspection, distinctly from the
mid-cycle defects in section 7 (a real but different failure category).

## 9. Independent audit (`qa-agent`, fresh session, no access to this document while auditing)

Mandate: mutation-test the three new Phase 8 checks, independently re-derive the golden-dataset
coverage table by hand, and independently re-inspect the two visual defects from section 7 with
fresh eyes, without trusting this document's own prose.

**Mutation testing (all three caught, all mutations reverted, `git diff` on `src/` empty
afterward):**
1. Removed the `_check_regression(...)` call from `classify_outcome` entirely →
   `test_classify_outcome_error_for_a_reproduced_regression` failed (`PASS` instead of the
   expected `ERROR`). Caught.
2. Changed `_SSIM_WRAP_TOLERANCE` from `0.05` to `1.0` (making the SSIM check unable to ever
   fail) → `test_compute_loop_metrics_flags_both_checks_for_a_non_periodic_sequence` failed
   specifically on `assert metrics.wrap_ssim_within_tolerance is False`, while the neighboring
   `wrap_step_within_2x_ordinary` assertion still passed — confirms the test genuinely exercises
   the SSIM tolerance independently of the pixel-diff check, not a duplicate assertion. Caught.
3. Hand-recomputed `golden_categories` for all 7 samples in
   `configs/phase3_3_eval_dataset.yaml` against all 10 `GoldenCategory` values independently —
   landed on exactly the same two zero-coverage categories
   (`partially_occluded_object`/`scale_or_deformation`), no hidden typo or third gap found.

No mutation went uncaught — no red flags from Part 1.

**Independent visual re-verification**: confirmed both defects as real (not confirmed by taking
this document's word for it — verified directly from the same PNG crops). Refined defect A's own
characterization beyond what this document originally said: the doubling is not confined to hair
strands, it visibly extends across the goggles/eye-patch outline and the jaw/mouth line too — a
large part of the head silhouette, not one lock of hair (already folded into section 7 above).
Confirmed defect B's characterization as accurate (hard vertical boundary crossing both
background texture and the panel border/gutter line).

**A finding beyond the original mandate, worth stating explicitly**: `LoopMetrics` (by design)
only ever compares the wrap transition (last frame vs. first) against an ordinary adjacent step —
it does not, and structurally cannot, inspect anything about intermediate mid-cycle frames. Both
real defects in section 7 are mid-cycle-only, so `seamless_loop_verified=true` was reported (and
is, narrowly, correct) for both renders in section 6.4 despite each containing a real visual
defect. This is not a bug in `LoopMetrics` (it is honestly scoped as a loop-continuity check, not
a general artifact detector) but it does mean `seamless_loop_verified=true` must not be read as
"the whole video is defect-free" — the codebase currently has no automated per-frame/mid-cycle
artifact check at all; both defects in this document were found solely by manual, targeted visual
inspection. Recorded as its own, distinct known limitation below, not merged into the loop-check
line item.

## 10. Known limitations

- Two real, disclosed golden-dataset coverage gaps: `partially_occluded_object`,
  `scale_or_deformation` (see section 5) — unchanged by this real run (no sample newly added).
- `object_near_boundary` coverage is incidental (one dropped-candidate rejection reason), not a
  sample deliberately chosen to exercise the category.
- **Two real, newly-discovered mid-cycle visual defects** (section 7) — a hair-ghosting artifact
  (`verified_action_1`) most likely from two overlapping real `character_hair` SECONDARY
  objects, and a hard seam artifact (`phase3_action_page`) most likely from a TRANSLATE layer's
  effective footprint leaking beyond the intended region. Neither was fixed in this phase:
  root-causing either would require inspecting the actual segmentation masks/bboxes at the
  moment of peak displacement (not just the composited output pixels this phase's evidence is
  limited to — the live GPU worker that produced them is an ephemeral Kaggle session, already
  gone), and a real fix plausibly touches `compositing`/multi-object z-order logic (ADR 0010) or
  a new cross-object overlap check in the `validation` stage — architectural scope this phase's
  brief explicitly excludes ("do not redesign grounding/segmentation architecture"). Flagged as
  concrete, evidenced follow-up work, not silently absorbed or fixed blind.
- **`LoopMetrics` structurally cannot catch a mid-cycle-only defect** (confirmed by the
  independent audit, section 9): it only ever compares the wrap transition against an ordinary
  adjacent step, never any intermediate frame — `seamless_loop_verified=true` is an honest,
  narrowly-scoped loop-continuity claim, not a general artifact-free claim. The codebase has no
  automated per-frame/peak-displacement artifact check at all yet; both real defects above were
  found only by manual, targeted visual inspection. A genuinely new, distinct gap from the two
  defects themselves — not previously identified before this phase's real evidence surfaced it.
- `classify_outcome` (section 6.2) is confirmed, on real data, to classify a sample's own
  documented-acceptable validation rejection (`eval_weapon_effects`) as ERROR rather than
  REJECTED, because it only consults structured ground-truth fields, not a sample's free-text
  `acceptable_outcome` nuance — the exact limitation ADR 0014's "Open questions" already
  anticipated, now observed for real, not merely hypothetical.
- The remote worker's `uv run mypy src` surfaced one further, real, pre-existing finding not
  reproducible locally (this local checkout has no `ml` extras installed, so the affected code
  path is untyped/skipped here): `segmentation/client.py:62` — `Sam2Model.from_pretrained(...).to(device)`
  flagged as a type mismatch through what appears to be a `transformers`-internal wrapped-method
  stub issue (`_Wrapped.__call__`). Not touched by any Phase 8 change (this file was not edited),
  not reproducible with the `transformers==5.0.0` version `docs/phase7-results.md` used (which
  reported this file clean) — most plausibly a type-stub regression introduced by the
  `transformers` version bump to `5.15.0` between sessions (this project's `pyproject.toml` does
  not pin an exact version). Disclosed, not fixed — a third-party library-interop question, not a
  Phase 8 code defect, and not this phase's declared scope.
- The client-side WebSocket connection to the remote kernel was lost mid-run (see section 6) —
  worked around by polling kernel state and reading the resulting file directly rather than
  relying on the live stream, but this is a real, disclosed operational fragility of the
  established remote-execution transport for long-running real GPU work, not previously
  documented at this level of detail.
