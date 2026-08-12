# Phase 7 results: end-to-end QA, evaluation, regression testing

Status: **in progress** (this document is being written incrementally as real evidence lands;
see the "Reproducibility instructions" section for the exact commands used to produce every
number below, and the closing summary for the final PASS/FAIL/DEFERRED verdict per acceptance
criterion).

## 1. Scope

Phase 7's canonical definition (`README.md`): "End-to-end QA, evaluation, regression testing."
This phase covers:

- **7.1 Deterministic regression layer**: multi-object E2E encode/decode regression,
  whole-pipeline determinism, panel-aware regression on real page geometry, a defensive
  `panel_bbox_px` check, and opt-in performance regression protection for Phase 6's
  local-rendering property.
- **7.2 Evaluation infrastructure**: extending `PageRunOutcome`/`EvaluationReport` to report
  SECONDARY/MICRO object outcomes (ADR 0010's explicitly-deferred item), and expanding the
  nondeterminism harness's default sample coverage to the full evaluation dataset.
- **7.3 Real-model QA**: a real automatic multi-object E2E attempt, real LaMa visual QA, and
  real multi-object encode/decode QA where a real multi-object render succeeds — all on a live
  Kaggle GPU worker (2x Tesla T4), never claimed from mocked/placeholder execution.

Explicitly out of scope (per the Phase 7 brief and CLAUDE.md's phase-boundary discipline):
batch/multi-page processing (ADR 0012's own "stays Phase 7+ scope" deferred to a *later* phase,
not claimed here), fixing PRIMARY grounding itself, redesigning grounding/segmentation
architecture, confidence-threshold redesign, codec support beyond H.264, frame-dump retention
policy, and reopening Phase 6 architecture.

## 2. Baseline

Verified before any Phase 7 change, on branch `phase-6-wip` at commit `065b19e` ("Phase 6
closure audit"): `git status` clean, `uv run pytest` **429 passed**, `uv run ruff check .`
clean, `uv run mypy src` clean (41 source files). This matches the Phase 6 closure claim
exactly — no discrepancy found, so Phase 7 work proceeded without a stop-and-report.

## 3. Implemented changes

| Commit | Summary |
| --- | --- |
| `49b9e05` | Phase 7.1: deterministic regression layer |
| `5c93028` | Phase 7.2.1: extend evaluation reporting to SECONDARY/MICRO objects |
| `4ce5f3d` | Phase 7.2.2: default the nondeterminism harness to the full evaluation dataset |

(Additional commits for Phase 7.3 evidence/docs are appended to this table as they land — see
`git log` for the authoritative, current list.)

### 7.1 — deterministic regression layer

- `tests/test_pipeline.py::test_run_pipeline_multi_object_e2e_encode_decode_regression`: a real
  multi-object scenario through the actual render/encode path, decoded back via
  `cv2.VideoCapture` from the real `.mp4` on disk (not the intermediate frame PNGs) — verifies
  frame count, resolution, seamless loop, object identity, static-region preservation, and no
  cross-object color contamination, all from the decoded video.
- `tests/test_pipeline.py::test_run_pipeline_is_deterministic_for_identical_fake_inputs`:
  identical deterministic fake inputs through the whole orchestration path produce
  byte-identical composited frames, run to run. Explicitly scoped to deterministic pipeline
  code — does not and cannot claim anything about real VLM determinism (see ADR 0009's own,
  opposite, real finding).
- `tests/test_pipeline.py::test_run_pipeline_panel_aware_regression_on_real_action_page_geometry`:
  deterministic regression using `examples/phase3_action_page.png`'s real 720x5062 dimensions
  and real `detect_panels()` output, through the real ADR-0011 panel-aware grounding path, with
  fake VLM/grounding/segmentation/reconstruction clients. Skips cleanly (documented reason) if
  the example file is genuinely absent.
- `src/manga_animation/grounding/ground.py::_grounding_region`: new defensive `ValueError` when
  `panel_bbox_px` doesn't fit inside `image`'s actual bounds — closes the exact gap ADR 0011's
  "Known limitations" disclosed (silent numpy out-of-range slice truncation). Boundary-exact
  panels (the ordinary real case) remain valid; only genuinely out-of-range boxes now raise.
  Tested from both directions in `tests/test_grounding.py`.
- `tests/test_performance.py` (new file, `pytestmark = pytest.mark.slow`, excluded from the
  default suite via a new `pyproject.toml` marker + `addopts = "-ra -m \"not slow\""`): two
  opt-in tests protecting Phase 6's local-rendering property with a generous 4x bound (page
  pixel count grows ~7.6x for the extreme-aspect-ratio case; a full-page-scaling regression
  would show ~6-8x, comfortably above the bound; the real, current, hoisted-bbox behavior
  measures ~2.4x, comfortably below it — see the QA audit's independent verification in
  section 6). Run explicitly: `uv run pytest -m slow`.

### 7.2 — evaluation infrastructure

- `src/manga_animation/evaluation/schemas.py`: new `ObjectAttemptOutcome` type and
  `PageRunOutcome.object_outcomes`/`schema_version` fields — per-SECONDARY/MICRO-object
  rendered/dropped reporting, additive and backward-compatible (old stored JSON without these
  keys loads with safe defaults: `object_outcomes=[]`, `schema_version=1`).
- `src/manga_animation/evaluation/metrics.py`: `EvaluationReport` gains
  `secondary_object_render_rate`/`micro_object_render_rate`, pooled across outcomes,
  `Rate(0, 0)` ("0/0, n/a") when there is nothing to report.
- `src/manga_animation/pipeline/orchestrator.py`: new `DroppedObjectResult` type and
  `PipelineRunResult.dropped_objects` — the two sites that previously only `logger.warning(...)`
  a SECONDARY/MICRO grounding/validation failure now also record it, so evaluation reporting
  can see past `secondary_objects` (which only ever held the successful ones).
- `docs/decisions/0013-phase7-secondary-micro-evaluation-reporting.md`: the ADR for this
  decision, including the schema-versioning convention (deliberately distinct from
  `EvalSample.annotation_version`'s ground-truth-revision convention — see ADR 0009).
- `scripts/run_phase3_3_evaluation.py`: updated to populate the new fields on every real run,
  and its `--nondeterminism-samples` default now expands to every sample_id in the loaded
  dataset (was a hardcoded 2-sample subset) — the code-level half of "expand nondeterminism
  evaluation beyond its current tiny subset."

## 4. Deterministic regression results

```
uv run pytest -q
```
**440 passed, 2 deselected** (the 2 opt-in `slow` performance tests), locally and independently
reproduced on the remote Kaggle worker after a fresh `uv sync` (**439 passed, 1 skipped, 2
deselected** there — the 1 skip is the panel-aware-geometry test, which needs
`examples/phase3_action_page.png` present; that file is git-ignored per ADR 0002 and wasn't yet
fetched on the worker at that point in the session, so the skip is the correct, honest
behavior, not a failure).

```
uv run pytest -m slow
```
**2 passed** (`tests/test_performance.py`).

```
uv run ruff check .
```
Clean, both locally and on the remote worker.

```
uv run mypy src
```
Clean, 41 source files.

An independent `qa-agent` audit (separate session, no access to this document while writing
its own findings) additionally ran real mutation testing against three of the new Phase 7.1
tests — introducing a real bug, confirming the test fails, then reverting via `git checkout
--` — and confirmed all three genuinely catch what they claim:
- A color leak into the static region (mutated `compositing/__init__.py`) was caught by the
  multi-object encode/decode regression test.
- A single non-deterministic pixel (mutated `pipeline/orchestrator.py`) was caught by the
  whole-pipeline determinism test (1 of 57,600 pixels).
- Disabling panel-aware grounding (`panel_bbox_px=None`) was caught by the real-geometry
  regression test on the actual 720x5062 page.

Verdict: **PASS-WITH-NOTES** — no logic bugs found; two documentation gaps (this file didn't
exist yet at audit time, and `README.md`'s phase table wasn't yet updated for Phase 7) are
resolved by this document and the README update accompanying it.

## 5. Evaluation-schema changes

Covered in section 3 above (ADR 0013). Deterministic test coverage:
`tests/test_evaluation.py`'s "Phase 7.2.1" section (5 new tests: default-empty/schema_version=1
backward compatibility, pooled secondary/micro render rates, zero-denominator "n/a" behavior,
non-interference with pre-existing PRIMARY-only metrics, and JSON round-tripping of the new
type). `tests/test_pipeline.py`'s two "drops a secondary" tests were extended (not replaced) to
assert on the new `dropped_objects` field directly.

## 6. Real-model evaluation results

All real-model work in this section ran on a live Kaggle Jupyter GPU worker (2x Tesla T4),
reached via the project's established non-browser Jupyter REST/kernel-WebSocket transport (see
`docs/phase3.2-results.md`'s "How this run was executed" for the same method used in every
prior real phase) — no `claude-in-chrome`, no browser automation. A dedicated kernel was
started for this session's work (not the user's own already-connected notebook kernel).

*(This section is filled in as real runs complete — see the git history of this file / the
final version of this document for the complete, final picture.)*

## 7. Visual QA findings

*(filled in below as results land)*

## 8. Negative results / failures

*(filled in below as results land)*

## 9. Known limitations

*(filled in at closure)*

## 10. Deferred work

- SHEAR/SCALE/MESH_WARP real-model transform-boundary evidence: attempted only as time/GPU
  budget allowed, using the existing controlled-fallback `run_pipeline(..., plan=...)`
  infrastructure (no new infrastructure built) — see section 6 for what was actually obtained.
- A full, exhaustive real multi-object render (PRIMARY + SECONDARY/MICRO all successfully
  grounded/validated/segmented/animated/composited on one real page) remains unobserved as of
  this document's real-model evidence, consistent with ADR 0010's "Revision (Phase 5 audit)"
  section — see section 6/8 for this phase's own real attempts and their outcomes.

## 11. Exact commands used

Local (this repository, no GPU):
```
uv run pytest -q
uv run pytest -m slow
uv run ruff check .
uv run mypy src
```

Remote (Kaggle GPU worker, after `git pull` at the commit under test):
```
uv sync --extra dev --extra cv --extra video --extra ml
uv run python scripts/fetch_sample_pages.py --count 2
uv run python scripts/fetch_phase3_sample_page.py
uv run python scripts/fetch_phase3_3_eval_pages.py
uv run pytest -q
uv run python scripts/run_phase3_3_evaluation.py --env kaggle
```
Plus two small, session-local driver scripts (not committed — ad hoc evidence-gathering
scripts, not reusable infrastructure) mirroring `scripts/run_phase3_pipeline.py`'s real
automatic-operation pattern: one real automatic panel-mode run against
`examples/phase3_action_page.png` reporting `secondary_objects`/`dropped_objects` explicitly
(not just the PRIMARY summary `run_phase3_pipeline.py` itself reports), and one real LaMa
visual-QA run against `examples/sample_page_01.png` saving debug crops. Both are reproducible
by any future session using `scripts/run_phase3_pipeline.py --page <page> --env kaggle` for the
pipeline mechanics; the debug-crop-saving and secondary/dropped-object reporting were the only
parts not already covered by a committed script.

The two `examples/verified_action/*.png` samples (no `fetch_script` — manually provided,
non-reproducible by design, see ADR 0009's revision) were copied to the remote worker via the
Jupyter Contents API directly from the local, already-present files, to let
`scripts/run_phase3_3_evaluation.py` load the complete 7-sample dataset without hard-failing on
missing images. This mirrors their established provenance (manually placed, never fetched by
script) rather than inventing a new distribution mechanism for them.

## 12. Test counts

- Baseline (Phase 6 closure, commit `065b19e`): 429 passed.
- After Phase 7.1: 435 passed, 2 deselected (new `slow` marker).
- After Phase 7.2: 440 passed, 2 deselected.
- Remote worker (fresh clone, same commit): 439 passed, 1 skipped (missing example file at
  that point in the session, since fixed by fetching it), 2 deselected.

## 13. ruff/mypy status

Clean throughout, both locally and on the remote worker, at every commit in section 3's table.

## 14. Git/commit state

See `git log --oneline` for the authoritative, current history. Working tree is clean at every
commit boundary in this document; no Phase 6 commit was amended, squashed, or reordered.

## 15. Reproducibility instructions

1. `git checkout phase-6-wip && git pull`
2. Local checks: `uv run pytest -q && uv run pytest -m slow && uv run ruff check . && uv run mypy src`
3. For real-model evidence: obtain a live Jupyter/Kaggle GPU worker URL (never guess or reuse a
   stale one — ask the project owner), then follow section 11's remote commands.

---

*This document is updated in place as Phase 7 work completes; it is not a final closure report
until this line is replaced with an explicit "Phase 7: COMPLETE / PARTIALLY COMPLETE" verdict
and the placeholder sections above are filled in with real content.*
