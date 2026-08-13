# 14. Phase 8: end-to-end production validation infrastructure

Status: Accepted

## Context

Phase 8's brief ("End-to-End Production Validation") asks for a golden E2E dataset, a
reproducible E2E runner exposing per-stage status, a machine/human-readable report that
"clearly distinguishes PASS / PASS WITH FALLBACK / REJECTED / ERROR," and loop validation that
does not "rely solely on raw pixel equality." A repository audit (this ADR's own prerequisite,
per the Phase 8 brief's "do not start implementation until that audit establishes the actual
state of the system") found the actual pipeline, evaluation, and rendering code already
substantially covers this ground — Phase 3.2 through 7 already built real per-stage failure
attribution (`pipeline.types.Stage`/`PipelineStageError`), a real evaluation dataset with honest
ground truth (`configs/phase3_3_eval_dataset.yaml`, ADR 0009), and a real, GPU-proven E2E runner
(`scripts/run_phase3_3_evaluation.py`). Three concrete, real gaps remained, found by direct
inspection, not assumption:

1. **Loop-continuity numbers were computed but not exposed or reusable.**
   `rendering/encode.py::_loop_continuity` was private, discarded its result after logging, and
   was pixel-diff-only (`cv2.absdiff(...).mean()`) — no structural/perceptual signal existed
   anywhere in `src/`. `docs/phase7-results.md` section 6.3's real `ordinary_adjacent_step`/
   `wrap_step` numbers were obtained by a third, independent, uncommitted re-implementation of
   this same logic against a decoded video — the production code itself never surfaced them.
2. **No unified PASS/PASS_WITH_FALLBACK/REJECTED/ERROR vocabulary existed.** Three incompatible
   ad hoc status vocabularies were in use: `PageRunOutcome.status: Literal["completed",
   "failed"]`, `ObjectAttemptOutcome.status: Literal["rendered", "dropped"]`, and
   `scripts/run_phase3_pipeline.py`'s own uppercase `"COMPLETED"`/`"FAILED"` strings — none
   distinguished an honest negative result (e.g. a correct all-STATIC read) from a real defect.
3. **`PageRunOutcome` carried no render-output evidence at all** (dimensions, frame count, fps,
   duration, codec, loop metrics) even for a `status="completed"` outcome — the brief's required
   report fields ("output dimensions; frame count; FPS; duration; codec/container; loop
   metrics") had no home to be written to.

A fourth, smaller, real bug was found while designing the fix for gap 2:
`PageRunOutcome.failing_stage: Stage | None` did not admit `scripts/run_phase3_3_evaluation.py`'s
own `failing_stage="unexpected"` value (its `except Exception` catch-all, for a bare exception
`PipelineStageError`'s stage-attribution never covered) — `Stage` is a closed 8-value `Literal`.
This was a **latent, disclosed-but-unfixed bug** (`docs/phase7-results.md` section 13): the first
time that exception path actually fired for real, constructing the `PageRunOutcome` would have
raised `pydantic.ValidationError`, discarding the original exception. Never exercised in
practice (no real run has yet hit a bare, non-`PipelineStageError` exception), so never observed
as a crash — found here by static reasoning about the type, the same way `mypy` originally
surfaced it.

The golden dataset itself: the Phase 8 brief lists 10 required coverage categories (single
object, multiple objects, partial occlusion, near-boundary, complex background, weapon/effect,
rotation, translation, scale/deformation, should-not-animate) and explicitly says "reuse existing
evaluation assets whenever appropriate" / "do not artificially construct cases that only
exercise happy paths if real/evaluation fixtures already exist." The existing 7-sample
`configs/phase3_3_eval_dataset.yaml` already has real, cited evidence (Phase 3.1 through 7,
`docs/phase7-results.md` section 6.2 in particular) covering 8 of the 10 categories; two —
`partially_occluded_object` and `scale_or_deformation` — have zero real evidence anywhere in
this project's history (SCALE/MESH_WARP were only ever exercised via uncommitted, ad hoc Phase 7
driver scripts against `sample_page_01.png`, never added as a real dataset sample).

## Decision

**1. `pipeline.types.LoopMetrics`** (new frozen dataclass, next to `RenderResult`): both the
pixel-level pair (`ordinary_adjacent_step_mean_abs_diff`, `wrap_step_mean_abs_diff`,
`wrap_step_within_2x_ordinary`) already computed pre-Phase-8, and a new structural pair
(`ordinary_adjacent_step_ssim`, `wrap_step_ssim`, `wrap_ssim_within_tolerance`) computed via a
standard windowed SSIM (Wang et al. 2004, 11x11 Gaussian window) implemented directly on
`cv2`/`numpy` (`cv2.getGaussianKernel`/`filter2D` — core `imgproc`, not `contrib`; no new
dependency). `RenderResult.loop_metrics: LoopMetrics | None` (defaults to `None`, the only
existing construction site is keyword-only, so this is non-breaking).
`rendering.encode._loop_continuity` is renamed to the public `compute_loop_metrics` (exported
from `manga_animation.rendering`) and now returns `LoopMetrics` rather than a private dict.
`seamless_loop_verified` now requires **both** signals to agree
(`LoopMetrics.seamless = wrap_step_within_2x_ordinary and wrap_ssim_within_tolerance`) — either
one alone flagging a problem withholds the seamless claim. `_SSIM_WRAP_TOLERANCE = 0.05` is a
documented, evidenced-by-test choice (verified against both a genuine periodic sequence and a
deliberately non-periodic one, `tests/test_rendering.py`), not a statistically calibrated set —
same status as every other threshold in this codebase (e.g. ADR 0008's transform-geometry
bounds). This does not claim the SSIM check catches a case pixel-diff alone would miss (no such
adversarial case was constructed) — it is a second, algorithmically independent signal, which is
what the brief asks for.

**2. `evaluation.metrics.E2EStatus`** (`Literal["PASS", "PASS_WITH_FALLBACK", "REJECTED",
"ERROR"]`) and **`classify_outcome(outcome, sample) -> E2EStatus`**, built entirely from signals
this module already computed (`_check_regression`, and the same semantic false-positive/
false-negative logic `compute_metrics` already used) — no new ground-truth interpretation
invented. Order: a reproduced regression or a confident (`ground_truth_uncertain=False`)
ground-truth contradiction is always **ERROR** first, regardless of `status`; an attributed
failure (`failing_stage` set, including the harness's own `"unexpected"`) that doesn't
contradict ground truth is **REJECTED** — an honest negative per "Static Is a Valid Result"
(`docs/architecture.md`), not a defect; a genuinely unattributed failure
(`failing_stage is None`) is **ERROR**; a completion is **PASS_WITH_FALLBACK** when
`used_fallback_plan` else **PASS**. `evaluation.metrics.StatusBreakdown` (four counts + `.total`)
and `EvaluationReport.status_breakdown` aggregate this per report, computed once inside
`compute_metrics` (no parallel counting pass). `scripts/run_phase3_3_evaluation.py` prints the
per-sample classification alongside the existing `status`/`failing_stage` line and the
per-report breakdown alongside the existing rates — no new report-formatting path.

**3. `evaluation.schemas.FailingStage = Stage | Literal["unexpected"]`** fixes the latent bug
above, scoped to `PageRunOutcome.failing_stage` only — `pipeline.types.Stage` itself stays
exact/closed (every real `PipelineStageError.stage` genuinely is one of the 8 real stages; this
is a strictly evaluation-reporting concern). **`evaluation.schemas.RenderSummary`** (mirrors the
serializable fields of `RenderResult`, plus `LoopMetricsOutcome` mirroring `LoopMetrics`) and
**`PageRunOutcome.render_summary: RenderSummary | None = None`**, populated in
`scripts/run_phase3_3_evaluation.py::_run_one`'s success path from `result.render`.
`PageRunOutcome.schema_version` gains meaning `3` ("also populates `render_summary`"), set
unconditionally by this producer on both the completed and failed paths (mirroring ADR 0013's
own `object_outcomes`/`schema_version=2` convention exactly) — `schema_version < 3` remains the
honest signal that a producer never populated this field, not "nothing was ever rendered."

**4. Golden dataset: `EvalSample.golden_categories: list[GoldenCategory] = []`** (additive,
`dataset.py`), where `GoldenCategory` is a closed 10-value `Literal` matching the brief's
required categories verbatim (`GOLDEN_DATASET_CATEGORIES`). Reuses
`configs/phase3_3_eval_dataset.yaml` directly — the file's own header comment now documents that
it doubles as the Phase 8 golden dataset — rather than fabricating a second, parallel dataset
file, per CLAUDE.md's "do not introduce a second parallel implementation" and the brief's own
"reuse existing evaluation assets whenever appropriate." Every one of the 7 existing samples got
`golden_categories` populated from real, already-documented evidence (cited inline per sample:
Phase 7's real multi-object renders, ADR 0011's real ROTATE grounding, ADR 0008's real
edge-margin rejections, etc.) — no category was assigned by guessing what a sample "should"
cover. `golden_categories` is explicitly **not** a ground-truth field under ADR 0009's
`annotation_version` convention (like the pre-existing `diversity_tag`, it is descriptive
coverage metadata, not a truth claim about the page). `dataset.golden_category_coverage`/
`uncovered_golden_categories` make gap-reporting a checkable function, not only prose — a test
(`test_real_golden_dataset_has_exactly_the_two_disclosed_coverage_gaps`) locks in that exactly
`partially_occluded_object` and `scale_or_deformation` are uncovered today, so a future edit that
silently changes this without updating the header note fails loudly. `object_near_boundary` has
exactly one contributing sample (`verified_action_1`), disclosed in both the YAML comment and
this ADR as **incidental** coverage (a dropped candidate's own real edge-margin rejection
reason), not a sample deliberately chosen to exercise boundary proximity.

## Consequences

- Fully backward compatible: `RenderResult.loop_metrics`, `PageRunOutcome.render_summary`,
  `EvalSample.golden_categories`, and `EvaluationReport.status_breakdown` are all additive with
  safe defaults; `FailingStage` only widens what was previously an unreachable-in-practice
  invalid state into a valid one. No existing field changes meaning.
- This ADR does not itself constitute a real GPU E2E run — see `docs/phase8-results.md` for
  what was and wasn't actually executed against live models, mirroring ADR 0013's own equivalent
  disclosure.
- Two real, disclosed coverage gaps remain in the golden dataset
  (`partially_occluded_object`, `scale_or_deformation`) — not fabricated around. Closing them
  for real would require either a new real manga page with genuine occlusion, or adding a real
  SCALE/MESH_WARP-targeted sample with real evidence behind it (the existing ad hoc Phase 7
  driver-script evidence against `sample_page_01.png`, `docs/phase7-results.md` section 6.5,
  could become that sample if promoted to a committed, reproducible fetch/evidence path — not
  attempted here, left for a future phase per the same "don't fabricate a fixture just to fill a
  quota" discipline `configs/phase3_3_eval_dataset.yaml`'s own header already establishes).

## Open questions

- Whether `classify_outcome` should also consult a sample's free-text `acceptable_outcome`
  field (e.g. via a second VLM call verifying the classification against that prose) is left
  open — this phase deliberately uses only structured, already-tested fields
  (`animation_possible`, `ground_truth_uncertain`, `regression_reference`), consistent with
  `_check_regression`'s own established preference for structured evidence over free-text
  parsing.
- `_SSIM_WRAP_TOLERANCE`'s value (0.05) is evidenced against exactly one real geometry (the
  synthetic moving-circle sequence in `tests/test_rendering.py`), not calibrated against a real
  rendered page's own periodic motion — flagged, not hidden, same status as every other
  threshold in this codebase at the time it was introduced.
