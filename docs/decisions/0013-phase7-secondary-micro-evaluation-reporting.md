# 13. Phase 7.2.1: SECONDARY/MICRO evaluation reporting

Status: Accepted

## Context

ADR 0010 (Phase 4/5, multi-object layer decomposition) explicitly deferred this exact gap to
Phase 7: "`evaluation/` is untouched by this ADR. `PageRunOutcome.primary_semantic_label`/
`primary_motion_type` still describe only the PRIMARY object, same as before — extending
evaluation to report on secondary/micro objects too is real future work, not attempted here
(out of this phase's scope; the repository's own plan places broader evaluation work at Phase
7, not here)." `README.md`'s Phase 7 row ("End-to-end QA, evaluation, regression testing") and
the Phase 7 scope audit that preceded this ADR both name this directly.

Two real gaps existed, not one:

1. **`PageRunOutcome` had no representation at all for SECONDARY/MICRO objects.** A page whose
   plan proposed a PRIMARY + two SECONDARY objects, where the PRIMARY rendered but both
   SECONDARY objects were dropped (ADR 0010's non-fatal SECONDARY/MICRO failure policy), was
   evaluation-indistinguishable from a page whose plan only ever proposed a single PRIMARY
   object. Both produce `status="completed"` with identical `primary_*` fields.
2. **`pipeline.orchestrator.PipelineRunResult` itself had no representation of a *dropped*
   SECONDARY/MICRO object**, only a `logger.warning(...)` call at each of the two drop sites
   (grounding failure, validation failure). `secondary_objects` only ever contained the
   objects that succeeded. Even a caller willing to extend `PageRunOutcome` had nothing to
   read a drop's object_id/stage/reason from — the information was discarded, not merely
   unreported.

Fixing gap 1 alone (a schema change with no new data feeding it) would have produced a field
that could never be populated for real by the one real driver script
(`scripts/run_phase3_3_evaluation.py`), so this ADR closes both together.

## Decision

**1. `pipeline.orchestrator.PipelineRunResult` gains `dropped_objects: list[DroppedObjectResult]`**
(additive, defaults to `[]`). `DroppedObjectResult` (`object_plan`, `failing_stage:
Literal["grounding", "validation"]`, `reason`) is populated at the exact two sites that
previously only logged a warning — no new stage, no change to the PRIMARY/SECONDARY/MICRO
failure policy itself (a PRIMARY failure still raises `PipelineStageError` and aborts the run;
only a SECONDARY/MICRO failure ever produces a `DroppedObjectResult`). `secondary_objects`'
existing meaning (only the objects that made it into the render) is unchanged.

**2. `evaluation/schemas.py` gains `ObjectAttemptOutcome`** (`object_id`, `semantic_label`,
`motion_type: Literal["secondary", "micro"]`, `status: Literal["rendered", "dropped"]`,
`validation_attempts`) and `PageRunOutcome.object_outcomes: list[ObjectAttemptOutcome] = []`.
PRIMARY is deliberately **not** represented in `object_outcomes` — it stays exactly where it
already was (`primary_semantic_label`/`primary_motion_type`/`validation_attempts`, unchanged
in meaning), since a PRIMARY failure already fails the whole run, so there is at most one
PRIMARY per outcome and it was already fully reported by those pre-existing fields. Adding a
redundant PRIMARY entry to `object_outcomes` too would create two disagreeing representations
of the same fact for no benefit.

**3. `PageRunOutcome.schema_version: int = 1`.** Every `PageRunOutcome` recorded before this
ADR (including any already-written `outputs/experiments/*.json` from a real run) loads with
`schema_version=1` (the pydantic default for a missing key) and `object_outcomes=[]` — this is
the honest, structurally-forced interpretation: "this producer never populated
`object_outcomes`," not "this page had zero SECONDARY/MICRO objects." A schema_version=2
producer (this ADR's updated `scripts/run_phase3_3_evaluation.py`) always sets
`schema_version=2` explicitly, on both the completed and failed paths, even when
`object_outcomes` ends up empty for a schema_version=2 record too (e.g. a single-PRIMARY-only
plan, or a PRIMARY failure with no visibility into secondary objects at all) — the version
field, not the list's emptiness, is what distinguishes "no data" from "genuinely none."

This intentionally does **not** reuse `EvalSample.annotation_version`'s convention (ADR 0009):
that field versions *ground truth*, bumped only by a human hand-editing
`configs/phase3_3_eval_dataset.yaml` alongside a reviewed git commit — it is a revision-audit
signal for data a human owns. `PageRunOutcome.schema_version` versions a *prediction record's
own schema*, set programmatically by whatever code constructs the record — the ordinary
meaning of a schema/format version, chosen deliberately over conflating the two.

**4. `evaluation/metrics.py::EvaluationReport` gains `secondary_object_render_rate`/
`micro_object_render_rate`.** Each pools every `object_outcomes` entry of the matching
`motion_type` across all outcomes in the report and reports `rendered / (rendered + dropped)`.
`denominator=0` (`Rate(0, 0)`, "0/0 (n/a)") whenever no such entries exist — schema_version=1
input and "genuinely zero SECONDARY/MICRO objects proposed" both produce this, which is the
same "don't fabricate a rate from nothing" discipline every other `Rate` in this module
already follows (see `Rate`'s own docstring). No other existing `EvaluationReport` field
changes meaning; this is purely additive.

**5. `scripts/run_phase3_3_evaluation.py::_run_one` updated** to populate `object_outcomes`
from `result.secondary_objects` (status="rendered") and `result.dropped_objects`
(status="dropped"), and to set `schema_version=2` on every constructed `PageRunOutcome`
(completed or failed). `main()`'s printed report gains the two new rates alongside the
existing ones, reusing the same `Rate.__str__` rendering — no new report-formatting code path.

## Consequences

- Fully backward compatible: every pre-existing `PageRunOutcome` JSON field is unchanged in
  meaning; `object_outcomes`/`schema_version` are additive with safe defaults, so old stored
  experiment results still parse and their existing metrics compute identically (see
  `tests/test_evaluation.py::test_object_outcomes_do_not_affect_primary_only_metrics`).
- `pipeline.orchestrator.PipelineRunResult.dropped_objects` is now real, inspectable evidence
  of exactly which SECONDARY/MICRO object was dropped, at which stage, and why — previously
  only visible via log output. Existing tests (`tests/test_pipeline.py`'s two "drops a
  secondary" tests) were extended, not replaced, to assert on this directly.
- This is reporting-only: it changes nothing about which objects get grounded/validated/
  animated, nothing about the PRIMARY/SECONDARY/MICRO failure policy (ADR 0010), and nothing
  about `secondary_objects`' pre-existing meaning.
- A real, schema_version=2, non-trivial multi-object evaluation run (one that actually
  exercises a nonzero `secondary_object_render_rate`/`micro_object_render_rate` denominator)
  still requires a live GPU worker producing a real multi-object plan — this ADR builds the
  reporting machinery and tests it with deterministic fixtures; it does not itself constitute
  that real run. See `docs/phase7-results.md` for what was and wasn't actually executed.

## Open questions

- Whether a per-object `failing_stage` distinction (`"grounding"` vs. `"validation"`) belongs
  in `ObjectAttemptOutcome` too (mirroring `DroppedObjectResult`), not just `status="dropped"`,
  is left open — no real evaluation run has yet needed to distinguish the two for reporting
  purposes, and adding it now would be speculative schema growth ahead of evidence, which this
  project's own conventions caution against (see ADR 0009's "Open questions" for the same
  reasoning applied to ground-truth provenance).
