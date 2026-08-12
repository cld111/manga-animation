# 6. Explicit grounding-target validation stage (Phase 3.2)

Status: Accepted

## Context

Phase 3.1's real end-to-end run ([`docs/phase3-results.md`](../phase3-results.md)) produced a
genuine, seamlessly-looping video with one real, uncontrolled defect: Grounding DINO's match
for the `flag_banner` prompt scored only 0.269 (barely above the model's own 0.25 detection
threshold) and landed on a face-and-speech-bubble region — there is no actual banner-shaped
object anywhere near that location on the source page. SAM 2.1 then produced a clean,
technically valid mask for that wrong region, and the deterministic `mesh_warp` animation
faithfully distorted a character's face and made a speech bubble unreadable at peak deflection.
Every *mechanical* invariant held (pixels outside the mask untouched, the loop genuinely
seamless, the encode valid) — the defect is semantic, not mechanical: nothing in the Phase 3.1
pipeline ever asked "is this actually the right object" before committing to animate it.

Real calibration evidence available from Phase 2/3.1 (`docs/phase2-benchmark-results.md`,
`docs/phase3-results.md`) rules out the simplest fix (a stricter minimum grounding-confidence
threshold):

| Semantic label | Grounding score | Actually correct? |
| --- | --- | --- |
| `hand` (Phase 2, first pass) | 0.53, 0.55 | yes |
| `face` (Phase 2, third pass) | 0.83 | yes (SAM IoU 0.921) |
| `hair` (Phase 2, third pass) | 0.32 | yes (SAM IoU up to 0.948) |
| `flag_banner` (Phase 3.1) | 0.269 | **no** — face/speech-bubble region |

The one confirmed-wrong detection (0.269) and one confirmed-right detection (0.32) are closer
to each other than either is to any plausible cutoff — a single score-based gate cannot
separate them without also rejecting real, correct detections. An independent signal is
required, not a tighter number on the same score (see the Phase 3.2 brief's explicit
"do not make arbitrary confidence thresholds without calibration/evidence" constraint).

## Decision

Add an explicit **validation stage** (`src/manga_animation/validation`), sitting between
grounding and segmentation in the pipeline:

    analysis -> grounding -> validation -> segmentation -> reconstruction -> animation
    -> compositing -> rendering

1. **Grounding now exposes ranked candidates, not just the best one.**
   `grounding/ground.py::ground_object_candidates` returns every usable detection (clipped,
   degenerate ones dropped) in score order, up to a small cap — one `client.detect()` call
   already returns multiple boxes above threshold, so this costs no extra grounding-model
   inference. `ground_object` (the pre-Phase-3.2 API) becomes a thin wrapper returning just the
   top candidate, so every Phase 3.1 caller/test is unaffected.

2. **`validate_target` (`validation/validate.py`) produces an explicit ACCEPT/REJECT with
   structured diagnostics (`pipeline.types.ValidationResult`) for one grounding candidate:**
   - A cheap, deterministic **bbox-plausibility pre-filter** first (no model call) — reuses
     the exact coverage-fraction bounds `segmentation/segment.py` already applies to its tight
     mask (`MIN_OBJECT_COVERAGE_FRACTION`/`MAX_OBJECT_COVERAGE_FRACTION`, moved to
     `pipeline/types.py` so both stages share one source of truth instead of two independently
     tuned numbers). A bbox is always ≥ its eventual tight mask, so this is, if anything, more
     permissive than the check already proven acceptable for masks.
   - A **semantic-agreement check**: one cheap `VLMClient.generate()` call (the same protocol
     `analysis/client.py` already defines — no new model dependency) on the cropped candidate
     region, asking whether it actually depicts the target `semantic_label`, with the intended
     `MotionSpec.transform_kind` given as context so a rigid/unrelated region reads as a
     mismatch even if it loosely resembles the target. This is exactly the "cheap VLM-based
     visual sanity check on the grounded crop" `docs/phase3-results.md` itself proposed as the
     fix for this exact failure.
   - An unparseable VLM response **fails closed** (rejected, not swallowed into a false
     accept) — never a second, more-expensive recovery call, since validation may already run
     several times per object (once per grounding candidate).
   - `validate_target` never raises — a REJECT is a normal, expected, loggable outcome (see
     the Phase 3.2 acceptance criterion: "a correct REJECT is a successful result").

3. **Orchestrator wiring** (`pipeline/orchestrator.py::run_pipeline`): grounds once, gets all
   ranked candidates, tries `validate_target` on each in score order, and proceeds to
   segmentation with the first ACCEPTed one. If every candidate is rejected, the run raises
   `PipelineStageError(stage="validation", ...)` — a new `Stage` literal value, distinguishing
   "grounding found nothing at all" (`stage="grounding"`) from "grounding found candidates but
   none were semantically correct" (`stage="validation"`), which matters for the real-run
   reporting this ADR's acceptance run needs (grounding rejection rate vs. grounding-empty
   rate).

4. **Scope of automatic retry, read literally off the Phase 3.2 brief's own "FAILURE BEHAVIOR"
   wording:** "attempt another ranked grounding candidate if available; otherwise retry using
   the defined fallback path; preserve the existing controlled fallback." This is implemented
   as exactly two levels, no more:
   - Try every ranked grounding candidate for the plan's one chosen object (new, this ADR).
   - If all are exhausted, the run fails outright; a human (or the calling script) decides
     whether to re-invoke `run_pipeline(..., plan=...)` with a different, human-verified
     `AnimationPlan` — the *existing*, unmodified Phase 3.1 controlled-fallback mechanism.
   The orchestrator deliberately does **not** automatically substitute a different
   VLM-proposed object (e.g. the next-ranked SECONDARY/MICRO candidate) when every grounding
   candidate for the first fails — the brief's failure-behavior section names only these two
   levels, and adding automatic object-substitution would be new orchestration control flow
   beyond what "preserve the existing controlled fallback" asks for (see the Phase 3.2 brief's
   "do not redesign the animation engine ... unless strictly required for validation"). The
   analysis-stage ranking fix below still improves the plan's *initial* choice of object, which
   is where the value of "ranked candidates" is actually realized this phase.

5. **The fallback path is validated too, not rubber-stamped.** A human-authored
   `plan=` override still has its grounded region run through `validate_target` — "never
   silently animate an unvalidated candidate" applies regardless of who chose the target. This
   is a deliberate behavior change from Phase 3.1 (where the fallback path made zero VLM calls
   at all): the fallback now still skips the *analysis* stage's full-page VLM call, but the
   validation stage's crop-check still runs.

6. **Analysis-stage fix, addressing the *other* real Phase 3.1 finding (all-STATIC reads):**
   `analysis/plan_builder.py`'s prompt only ever asked whether motion was deformation drawn
   directly on the object itself — which structurally cannot justify motion for a page (like
   Phase 3.1's real sample) whose only cue is a page-level speed-line effect, not per-object
   deformation. The prompt now explicitly lists panel/page-level effect lines and pose-implied
   motion as valid evidence categories, alongside on-object deformation. Separately,
   `_rank_candidates` (replacing `_select_single_primary`) now ranks *every* non-STATIC
   decision (`primary`/`secondary`/`micro`) instead of discarding SECONDARY/MICRO reads
   entirely when no object was labeled literally "primary" — a real gap where the model *did*
   identify motion-worthy signal but wasn't confident enough to call it primary used to be
   treated identically to a genuine all-STATIC read. `motion_type` still strictly dominates
   `confidence` in the ranking, so an existing real "primary" always wins, preserving Phase
   3.1's tested selection behavior whenever one exists.

## Consequences

- New `Stage` literal value `"validation"`; new `pipeline.types.ValidationResult`; new shared
  `pipeline.types.MIN_OBJECT_COVERAGE_FRACTION`/`MAX_OBJECT_COVERAGE_FRACTION` (relocated from
  `segmentation/segment.py`, values unchanged).
- `src/manga_animation/validation` is a new stage package; see `docs/pipeline.md`'s "Stage
  ownership" section for who owns it and why.
- `grounding.ground_object` keeps its exact Phase 3.1 signature/behavior (thin wrapper over
  `ground_object_candidates`) — no caller outside the orchestrator needed to change.
- The AnimationPlan schema itself is unchanged — ranking is an internal `plan_builder`
  concept (a list of `_RawObjectDecision`, not a schema field), so the plan stays pixel-free
  and model-agnostic exactly as `docs/animation-plan-schema.md` specifies.
- The controlled-fallback mechanism's call contract (`run_pipeline(..., plan=...)`) is
  unchanged; only its internal behavior (also runs validation now) changed, and is covered by
  a new regression test (`tests/test_pipeline.py::test_run_pipeline_fallback_plan_can_still_be_rejected_by_validation`).

## Open questions

- The bbox-plausibility bounds are reused from segmentation's own (already "deliberately
  permissive, not tuned per-class") numbers, not independently calibrated for bboxes
  specifically — still no real mask/bbox dataset large enough to tune against.
- The VLM crop-check's own `confidence` field is recorded as a diagnostic but is **not** used
  as a second numeric gate on top of `matches` — stacking an uncalibrated confidence cutoff on
  a brand-new signal would repeat exactly the mistake this ADR's evidence argues against.
  Revisit once a real run produces enough `ValidationResult` records to see whether
  `matches`/`confidence` actually diverge in practice.
- Whether this validator generalizes to a second, visually distinct manga series/art style is
  untested — all real evidence so far (Phase 2 and Phase 3.1) is one MangaDex series.
