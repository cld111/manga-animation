# Phase 3.3 results: panel-aware analysis + evaluation framework

Real, remote-GPU, end-to-end results for Phase 3.3 (see the Phase 3.3 brief, delivered
directly to the assistant, not committed as a file, and
[ADR 0007](decisions/0007-panel-aware-analysis.md) for the design this implements). This is a
point-in-time results record, not a design doc — ADR 0007 is the source of truth for *why* the
panel detector looks like this; this file is *what happened when it ran*, for real, against
real models.

**Status: implementation complete, locally verified (pytest/ruff/mypy all green), AND a real
end-to-end evaluation run completed on the remote Kaggle GPU worker, comparing page-level and
panel-aware analysis over 5 real pages across 4 different manga/webtoon series.**

## What Phase 3.3 set out to do

Per `docs/pipeline.md` and Phase 3.2's own "Remaining limitations" (`docs/phase3.2-results.md`):
no real panel/scene splitting existed before the VLM call — the whole page was always treated
as one `PanelPlan` covering `(0, 0, 1, 1)`. This is a real, confirmed gap for pages where the
motion-relevant panel is a small fraction of a large page (the original Phase 3.1 720x5062px
page, specifically). Phase 3.3's two goals: (1) build a real panel-aware analysis path, and
(2) build an honest, reproducible evaluation framework to measure whether it actually helps.

## Panel detection: what was inspected before implementing

Before writing any detector, the existing repository, dependencies, and real sample pages were
inspected (per the brief's explicit "First: inspect before implementing" instruction):

- `opencv-python-headless` is already a project dependency (the `cv` extras group), already
  imported at module level in `animation/transforms.py` — no new dependency needed.
- All three pre-existing real sample pages (`examples/sample_page_01.png`,
  `sample_page_02.png`, `phase3_action_page.png`) were visually inspected: all three are
  digital, full-color, single-column webtoon-style pages with wide, near-uniform gutters
  between panels — a case classical gutter-detection handles well, with no labeled dataset
  needed to train/calibrate a learned model against.
- `docs/architecture.md`'s "Deterministic First" principle already states the project's
  default preference here.

**Decision: a deterministic, model-free, recursive gutter-based ("XY-cut") panel detector**
(`analysis/panels.py::detect_panels`) — no new model dependency. Full rationale, algorithm
description, and known limitations: [ADR 0007](decisions/0007-panel-aware-analysis.md).

Two real, evidenced algorithm bugs were found and fixed *during* local testing against
synthetic pages *and* the real sample pages (not just synthetic fixtures):

1. A uniform run spanning nearly an entire row/column (e.g. a genuinely flat-colored panel, or
   — found on the real `sample_page_02.png` — a UI header bar whose background reads as
   row-uniform for a long stretch) was being treated as a real inter-panel gutter, producing a
   spurious split down the middle of real content. Fixed by excluding runs longer than 85% of
   their axis from being treated as splittable gutters (`_MAX_GUTTER_RUN_FRACTION`).
2. A gutter run touching the true page edge (pure leading/trailing margin, not a boundary
   *between* two panels) was still generating a spurious cut, carving a meaningless sliver
   "panel" out of blank margin — found via a synthetic very-tall-page test, confirmed for real
   on `examples/phase3_action_page.png` (a bottom-margin sliver at y=4950–5000 was being
   reported as its own 2-column "panel" before the fix).

Both fixes are covered by regression tests (`tests/test_panels.py`).

## Real detector behavior on every page in the evaluation dataset

| Page | Size (px) | Panels found | Confidence range |
| --- | --- | --- | --- |
| `sample_page_01.png` | 800x2305 | 2 (`gutter_xy_cut`) | 1.00 |
| `sample_page_02.png` | 800x2216 | 1 (`fallback_full_page`) | 0.50 |
| `phase3_action_page.png` | 720x5062 | 4 (`gutter_xy_cut`) | 0.98–1.00 |
| `eval_static_dialogue.png` | 720x4180 | 1 (`fallback_full_page`) | 0.50 |
| `eval_weapon_effects.png` | 900x2182 | 1 (`fallback_full_page`) | 0.50 |

**Real, honest finding: 3 of 5 real pages produced no internal gutter split at all** — the
detector correctly, safely fell back to a single whole-page candidate rather than inventing a
boundary. `sample_page_02.png`'s case was specifically investigated: its apparent "panels" (an
icon-bar header directly above a phone-screen graphic) don't have real inter-element
whitespace — the artwork itself reads as row-uniform across a long stretch that crosses what a
human might call a boundary. This is the exact class of case ADR 0007's "Open questions"
already flagged (`_MAX_GUTTER_RUN_FRACTION`'s docstring) — not a bug, a real, evidenced
limitation of a pure-variance-based signal on this kind of layout.

## Panel-aware analysis architecture (what changed)

- **`pipeline/types.py`**: `PanelCandidate`/`PanelSource` (new types); `normalized_bbox_to_px`
  / `bbox_px_to_normalized` (new, tested, bidirectional page↔panel coordinate mapping —
  `pipeline/orchestrator.py::_panel_bbox_px` now calls the shared function instead of
  duplicating the conversion).
- **`analysis/panels.py`** (new module): `detect_panels()`, the deterministic detector above.
- **`analysis/plan_builder.py`**: `analyze_page` is **completely unchanged** (Phase 3.3
  acceptance criterion #2). New `analyze_page_panels()` runs one VLM call per detected panel,
  pools every panel's ranked candidates with the same `(motion_type priority, confidence)` rule
  `_rank_candidates` already used, and falls back to `analyze_page` only when panel detection
  itself provides no usable signal (zero panels, or every panel's VLM call unparseable) — never
  when the real result is a genuine all-STATIC read (see `analyze_page_panels`'s docstring for
  why that distinction matters: silently retrying at the page level would let VLM
  nondeterminism quietly overrule a real per-panel finding).
- **`pipeline/orchestrator.py`**: `run_pipeline(..., analysis_mode: Literal["page","panel"] =
  "page")` — default unchanged; grounding, validation, segmentation, animation, compositing,
  and rendering are **identical code paths either way**, per ADR 0007's explicit decoupling
  from Grounding DINO/SAM 2.1/the animation engine.

## Evaluation dataset

`configs/phase3_3_eval_dataset.yaml` — 5 real pages across **4 different manga/webtoon series**
(a deliberate improvement over Phase 3.2's evidence base, which was 100% one series):

| Sample | Series | Diversity tag | Ground truth (`animation_possible`) |
| --- | --- | --- | --- |
| `sample_page_01` | (Phase 2 fetch, citation not preserved — see manifest) | ambiguous/nondeterministic | uncertain |
| `sample_page_02` | (Phase 2 fetch, citation not preserved) | hair | yes (Phase 3.2 positive control) |
| `phase3_action_page` | The Skeleton Soldier Failed to Defend the Dungeon | weapon/action/effects | yes |
| `eval_static_dialogue` | Who Made Me a Princess | static/dialogue | no |
| `eval_weapon_effects` | Latna Saga: Survival of a Sword King | weapon/action/effects | yes |

No ground truth was fabricated: every field is either a directly-verifiable visual read (e.g.
`eval_static_dialogue`'s STATIC call — no motion lines, no implied force, purely dialogue
panels) or an already-real, already-confirmed prior result (`sample_page_02`'s Phase 3.2 visual
QA). Genuinely uncertain/ambiguous cases (`sample_page_01`; the exact single best target on
`phase3_action_page`) are left `null`/`"uncertain"` rather than guessed — see each sample's
`notes` field in the manifest for the reasoning.

**Honest gap**: no "clothing" (wind-blown cloth/dress) or pure "environmental motion" sample
was found during a bounded real search (2 candidate series previewed, 5 candidate pages visually
inspected) — not fabricated to fill the category, left as an explicit, disclosed gap.

## Real end-to-end run

Performed on the user's live Kaggle Jupyter session (2x Tesla T4, `torch` 2.10.0+cu128,
`transformers` 5.0.0 — same environment as ADR 0005/Phase 3.1/3.2), reached via a genuinely
programmatic, non-browser transport: the session's Jupyter REST/kernel-WebSocket API, driven
directly over HTTP/WebSocket from the local session (no `claude-in-chrome`). A dedicated,
isolated kernel was started for this work and torn down after use each time (never reusing the
user's own already-connected kernel — confirmed via `GET /api/kernels` before and after).

**Code sync**: per real-time user approval (this session asked explicitly, since the Phase 3.3
brief's "no commit/push without approval" instruction conflicts with ADR 0002/0003's "code
moves to remote only via git, never manual copying" for the specific case of needing the
remote worker to test brand-new local code before final review), an interim commit was made on
a non-`main` branch (`phase-3.3-wip`, commit `7d05bd2`) and pushed so the remote worker could
`git clone --branch phase-3.3-wip` it — not a final commit, and explicitly *not* pushed to
`main`. `uv run pytest -q` was re-run on the remote worker after clone/install and passed
identically (256 passed). The two non-reproducible pre-existing sample images
(`sample_page_01.png`/`sample_page_02.png` — fetched via a live, time-varying MangaDex query in
Phase 2, not a hardcoded/reproducible one) were transferred via the Jupyter Contents API
(uploaded as the exact same bytes already in this project's history, sha256-verified identical
on both sides) — a data-artifact transfer, not a source-code one, consistent with this
project's own "generated/fetched artifacts, not canonical, re-derived as needed" convention for
`examples/*.png`.

### Page-level vs. panel-level comparison (n=5 samples, both modes)

| Metric | Page-level | Panel-level |
| --- | --- | --- |
| VLM usable-target rate | 3/5 (60.0%) | 3/5 (60.0%) |
| STATIC rate | 2/5 (40.0%) | 2/5 (40.0%) |
| Grounding success rate | 2/3 (66.7%) | 2/3 (66.7%) |
| Validation acceptance rate | 2/4 (50.0%) | 2/4 (50.0%) |
| Validation rejection rate | 2/4 (50.0%) | 2/4 (50.0%) |
| End-to-end completion rate | 2/5 (40.0%) | 2/5 (40.0%) |
| Semantic false-positive rate | 0/1 (0.0%) | 0/1 (0.0%) |
| Semantic false-negative rate | 2/3 (66.7%) | 2/3 (66.7%) |
| Regression violations | 0/2 | 0/2 |
| Panel detection multi-panel rate | n/a | 2/5 (40.0%) |

**Every completion/failure metric is bit-for-bit identical between the two modes on this real
run.** Per-sample detail (from `outputs/experiments/phase3_3_evaluation_20260812T094106Z.json`):

| Sample | Page-level result | Panel-level result |
| --- | --- | --- |
| `sample_page_01` | COMPLETED — `character_hair` PRIMARY, ACCEPT (score 0.610, *"clearly shows a character's hair"*) | COMPLETED — identical: `character_hair` PRIMARY, ACCEPT (score 0.610) |
| `sample_page_02` | FAILED (`analysis`, all-STATIC) | FAILED (`analysis`, all-STATIC across every analyzed panel) |
| `phase3_action_page` | FAILED (`grounding`) — VLM proposed **`weapon`**, 0 detections above threshold | FAILED (`grounding`) — VLM proposed **`character hair`** (a *different* candidate than the page-level call), 0 detections above threshold |
| `eval_static_dialogue` | FAILED (`analysis`, all-STATIC) | FAILED (`analysis`, all-STATIC across every analyzed panel) |
| `eval_weapon_effects` | COMPLETED — `weapon` PRIMARY; 2 candidates REJECTed, 3rd ACCEPTed (score 0.255) | COMPLETED — byte-identical render to page-level |

**Reading this honestly, per the brief's explicit "do not declare success merely because the
code exists" instruction: no measurable end-to-end benefit from panel-aware analysis was
observed in this run.** Three of five pages had no real internal gutter structure at all (panel
mode degrades to an equivalent of page-level for them by construction — see the detector
results table above), so panel-awareness had no opportunity to change anything there, and it
didn't. The one page where panel-awareness demonstrably changed VLM behavior
(`phase3_action_page`: `weapon` vs. `character hair`) still ended in the same outcome
(grounding found nothing for either candidate), so the change wasn't visible in any completion
metric. `sample_page_01` (2 real panels, both modes completed identically) is the one case that
exercised panel-awareness *and* completed — a real, positive "no regression" data point, not a
"panel-level was better" one.

**Sample size caveat, stated explicitly per the brief's requirement**: n=5 pages (n=2 with real
multi-panel structure) is far too small to establish statistical significance in either
direction. This comparison demonstrates the panel-aware path is real, wired correctly, and
produces at least one visibly different VLM read — it does not demonstrate a measurable
reliability improvement on this dataset, and is not claimed to.

### VLM nondeterminism (repeated-run measurement)

3 repeated `analyze_page` calls each, same process/session, identical input:

| Sample | Outcome stable? | Target category stable? | Outcomes seen | Labels seen |
| --- | --- | --- | --- | --- |
| `sample_page_01` | Yes | Yes | `usable` (all 3) | `character_hair` (all 3) |
| `sample_page_02` | Yes | Yes (vacuously) | `static_or_unusable` (all 3) | — |

**Real, new, more significant finding than Phase 3.2's**: within *this* session, 3 repeated
calls against each page were internally self-consistent. But `sample_page_02` — Phase 3.2's own
positive control, with a real, visually-confirmed correct hair-sway render
(`docs/phase3.2-results.md`) — came back **all-STATIC in every single call this session** (both
in the main comparison run and in all 3 repeated nondeterminism-check calls), a full reversal
from its previously-documented behavior. This is a real, evidence-based escalation of Phase
3.2's documented finding: VLM nondeterminism on this system is not only a within-session,
call-to-call property — it can flip an entire page's read across different sessions/model
loads, not just across repeated calls in one loaded session. Both are real, both are now
recorded; no explanation is claimed beyond what was actually observed (no seed/temperature is
pinned in `Qwen25VLClient.generate()` — unchanged this phase, out of scope, see "Remaining
limitations").

**Consequence for the evaluation numbers above**: this comparison run's page-level and
panel-level metrics should be read as *this run's* real numbers, not fixed properties of either
analysis mode — a different session could show `sample_page_02` (or others) complete instead of
fail, independent of `analysis_mode`.

### Phase 3.1/3.2 regression re-verification

Explicitly re-run (not merely cited) on this phase's code: `phase3_action_page.png` with the
exact reconstructed Phase 3.1 fallback plan (`semantic_label="flag_banner"`, `mesh_warp`,
amplitude 0.12, matching `_MOTION_HEURISTICS`' flag/banner template) —

```
CORRECTLY REJECTED:
stage= validation
detail= all 1 grounding candidate(s) for semantic_label='flag_banner' failed target validation:
  rank=0 The crop shows a character's head and dialogue box, not a flag banner.
```

Identical diagnostic wording to Phase 3.2's original real confirmation. **The historically-
observed face/speech-bubble distortion still cannot recur through this path**, on the current,
modified codebase (acceptance criterion #8: CONFIRMED). Note: the *automatic* comparison run
above never reached this exact scenario on its own (the VLM proposed `weapon`/`character hair`
this session, not `flag_banner`) — this regression check is a deliberate, explicit
reproduction, not something the automatic run happened to re-trigger by chance.

## Visual QA (real, decoded frame pixels, not simulated)

Frames fetched directly via the Jupyter Contents API and inspected directly (not through
`claude-in-chrome` — a same-origin authenticated REST call, then read like any other file).

**`sample_page_01` (both modes)**: comparing frame 0 (rest) to frame 24 (quarter-cycle) —
the character's hair visibly translates/sways; the face, both speech bubbles ("WAIT A MINUTE.",
"EXCUSE ME?"), and the blue system-message box are pixel-identical outside the animated region,
in both the page-level and panel-level render. A genuine, repeat-confirmed true positive.

**`eval_weapon_effects` (both modes — byte-identical renders)**: a **real, new visual defect**,
not previously seen on this project. The validated ("ACCEPT", score 0.255) `weapon` candidate's
grounding box is not tightly scoped to a weapon shape — comparing frame 0 to frame 24, the
`rotate` transform visibly tilts/skews the **entire dark action panel** (the "척" sound-effect
text, the panel's own border, the surrounding energy-effect artwork), not a specific object,
producing torn black wedge artifacts at the frame edges where the rotation reveals background.
Root cause: the semantic validation check correctly answered "yes, this crop plausibly shows a
weapon" (it does — a blade-like shape is visible in the crop), but nothing checks whether the
*box itself* is tightly scoped enough for the *intended transform* to look correct — the
existing bbox-plausibility pre-filter (`MIN/MAX_OBJECT_COVERAGE_FRACTION`, up to 90% of the
full image) is far too permissive for a `rotate`/`mesh_warp` transform specifically, even though
it may be reasonable for a `translate`/`opacity` one. **This is the same class of defect Phase
3.1 originally found** (a technically-accepted candidate that produces a visibly wrong result)
but a different, new mechanism — not a semantic mismatch this time, a box-size/transform-kind
mismatch. Out of scope to fix in Phase 3.3 (this is a Phase 3.1/3.2 validation-stage gap, not a
panel-detection one — see "Remaining limitations").

**Panel-boundary artifacts**: none observed in any inspected frame — panel-aware analysis never
touches grounding/segmentation/animation/compositing pixel paths (ADR 0007's decoupling), so
this is expected, and visually confirmed, not just assumed.

## Tests added

- `tests/test_panels.py` (16 tests): zero/one/multiple panels, tall pages, degenerate/malformed
  geometry, reading order, confidence-reflects-evidence-strength, determinism, real-sample-page
  dimensions.
- `tests/test_pipeline_types.py` (17 tests): `PanelCandidate` invariants,
  `normalized_bbox_to_px`/`bbox_px_to_normalized` correctness and round-trip stability.
- `tests/test_analysis.py` (+9 tests): panel-aware plan construction, page-level fallback
  (zero panels, all-unparseable panels), single-panel-failure resilience, cross-panel ranking,
  all-STATIC-does-not-fall-back, coordinate mapping into `PanelPlan.bbox`.
- `tests/test_pipeline.py` (+3 tests): `analysis_mode="panel"` end-to-end render, Phase 3.2
  validation-gate regression guard through the panel-aware path, default-mode-is-page guard.
- `tests/test_evaluation.py` (31 tests): `Rate` formatting/validation, every
  `compute_metrics` denominator/numerator case (including zero-denominator and
  missing-ground-truth-sample edge cases), regression-violation detection,
  `summarize_repeated_runs` (stable/unstable outcome and target-category cases), real dataset
  manifest validation.

Every existing Phase 1–3.2 test continues to pass unmodified.

## Test/lint/type results (local, then confirmed identical on remote)

```
uv run pytest -q       -> 256 passed (ffmpeg available via a local imageio-ffmpeg symlink)
uv run ruff check .    -> All checks passed!
uv run mypy src        -> Success: no issues found in 40 source files
```

Remote (`phase-3.3-wip` @ `7d05bd2`, same GPU worker as the E2E run): `uv run pytest -q` ->
**256 passed**, identical to local.

## Remaining limitations / next-phase work

- **No measurable end-to-end reliability improvement from panel-aware analysis was
  demonstrated on this dataset** (see the comparison table above) — the path is real, tested,
  and produces at least one different VLM read, but n=5 (n=2 with real panel structure) cannot
  show a statistically meaningful benefit. A larger real dataset — specifically more pages like
  the original Phase 3.1 motivating case (a very tall page whose motion cue is a small fraction
  of the whole) — is the natural next real test, not attempted here (this phase's dataset
  additions were bounded by a real, time-boxed search, per the brief's "small honest dataset"
  instruction).
- **Gutter-based detection does not generalize to every real layout** — 3/5 real pages in this
  very dataset had no internal gutter structure the detector could use (a real, evidenced
  limitation, not a hypothetical one). Traditional, non-webtoon, multi-column printed manga
  grids remain completely untested (no such real page exists anywhere in this project's sample
  history) — see ADR 0007's "Open questions".
- **New, real visual defect found this phase** (`eval_weapon_effects`): a validated candidate
  whose bounding box is too large for its assigned transform kind (`rotate`) produces a
  visibly wrong render (the whole panel tilts, not a weapon). This is a Phase 3.1/3.2
  validation-stage gap (the bbox-plausibility check doesn't consider `transform_kind`), not a
  Phase 3.3 panel-detection one — flagged for a future phase, not fixed here (out of this
  phase's explicit scope: "do not redesign SAM 2.1/Grounding DINO", "do not silently weaken
  validation" — also does not authorize silently *strengthening* it unreviewed mid-phase).
- **VLM nondeterminism is a larger, more consequential property than Phase 3.2 first
  documented** — `sample_page_02` fully reversed its STATIC/usable read across sessions this
  phase. `Qwen25VLClient.generate()` still pins no seed/temperature (unchanged, out of scope —
  the Phase 3.3 brief explicitly says not to "fix" nondeterminism via prompt changes without
  evidence it's the right fix; this finding is about decoding parameters, a different kind of
  change, and still not attempted here without being asked).
- **Candidate-ordering nondeterminism** (whether the *ranked list*, not just the final chosen
  PRIMARY, changes shape across runs) is not measured — the public `analyze_page`/
  `analyze_page_panels` API deliberately doesn't expose the pre-collapse ranked list (see
  `evaluation/nondeterminism.py`'s docstring) — adding a debug-only hook for this was judged
  speculative API growth without evidence it's needed, not attempted this phase.
- `sample_page_01.png`/`sample_page_02.png`'s exact source citation (series/chapter) was never
  recorded in Phase 2's docs, and `fetch_sample_pages.py`'s live "top-followed manga" query is
  not reproducible on demand — a real, disclosed documentation gap carried forward, not
  fabricated now (see `configs/phase3_3_eval_dataset.yaml`'s header comment).
- Cross-series generalization (Phase 3.2's own flagged open question) has real, if small,
  additional evidence now: `eval_weapon_effects` (a second, visually distinct series) exercised
  the same grounding-candidate-retry/validation machinery successfully (2 correct REJECTs, 1
  ACCEPT) — but also surfaced the new bbox/transform-kind defect above, so "generalizes" is not
  a clean yes.

## Verdict against the Phase 3.3 acceptance criteria

1. **A real panel-aware analysis path exists.** PASS — `analyze_page_panels`, wired into
   `run_pipeline(..., analysis_mode="panel")`, exercised for real on the remote GPU worker.
2. **Page-level analysis remains supported.** PASS — `analyze_page` unchanged; default mode.
3. **Panel coordinates map correctly to original-page coordinates.** PASS — tested
   (`tests/test_pipeline_types.py`), and confirmed for real (panel-aware `PanelPlan.bbox`
   values on real pages land where the detected panels actually are).
4. **Panel detection has deterministic tests.** PASS — `tests/test_panels.py`, 16 tests.
5. **Panel-level VLM analysis works on real samples.** PASS — real Qwen2.5-VL calls per real
   detected panel, on the remote GPU worker (`sample_page_01`: 2 real panel calls;
   `phase3_action_page`: 4 real panel calls).
6. **If panel detection fails, a defined page-level fallback exists.** PASS — tested
   (`tests/test_analysis.py`) and real (3/5 dataset pages actually took the
   `fallback_full_page` detector path this run).
7. **Existing Phase 3.2 validation remains fully functional.** PASS — real validation
   ACCEPTs/REJECTs observed this run (`eval_weapon_effects`: 2 REJECT + 1 ACCEPT); orchestrator
   regression test added for the panel-aware path specifically.
8. **The Phase 3.1 historical false-grounding case is still rejected.** PASS — explicitly
   re-reproduced on the current code, real models, this phase (see "regression
   re-verification" above) — identical diagnostic wording to Phase 3.2's original.
9. **A reproducible evaluation dataset exists using real samples.** PASS, with a disclosed
   caveat — 3/5 samples are reproducible on demand via hardcoded fetch scripts; 2/5
   (`sample_page_01`/`02`) are real but not re-fetchable on demand (documented, not hidden).
10. **Evaluation metrics include denominators and sample counts.** PASS — every `Rate` in this
    document and in `EvaluationReport` carries `numerator/denominator`; see `Rate.__str__`.
11. **Repeated VLM runs are measured for nondeterminism.** PASS — real repeated runs performed;
    a real, significant, previously-undocumented cross-session flip was found and reported
    (`sample_page_02`).
12. **Page-level vs. panel-level results are compared on the same samples.** PASS — see the
    comparison table; **honestly reported as showing no measurable benefit on this dataset**,
    not spun positive.
13. **Real E2E execution via direct Jupyter REST/WebSocket, never claude-in-chrome.** PASS.
14. **Visual QA is performed on real output.** PASS — 2 real render pairs inspected; one true
    positive confirmed, one new real defect found and honestly reported.
15. **pytest, ruff and mypy pass.** PASS — locally and re-confirmed on the remote worker.
16. **Remaining limitations are explicitly documented.** PASS — see the section above.

**Overall: Phase 3.3's engineering deliverables are complete and real (panel detection,
coordinate mapping, panel-aware analysis, evaluation framework, all tested and exercised
against real models). Its scientific question — "does panel-aware analysis make the pipeline
more reliable?" — has a real, honest answer: not measurably, not yet, on this small dataset;
the path is correct, tested, decoupled, and safe (no regressions, no panel-boundary artifacts,
the historical failure still rejected), but the brief's own bar ("must demonstrate a measurable
benefit or a clearly documented benefit... do not declare success merely because it exists in
code") is not met by "no difference on n=5." This is reported as a real null result, not
reframed as a win.**

---

# Phase 3.3.1: transform-aware geometric validation (post-E2E fix)

Follow-up to the run above. The `eval_weapon_effects.png` visual QA finding (a validated
candidate whose oversized bbox caused `rotate` to swing the whole panel instead of the weapon)
was investigated and fixed under a separate, narrower brief. See
[ADR 0008](decisions/0008-transform-aware-target-validation.md) for the full design.

## Root cause

Phase 3.2's semantic validation check answers exactly one question: "does this crop depict the
target?" It never asked a second, independent question: "is this specific bbox geometrically
safe for the transform the plan intends to apply to it?" A bbox can be a correct instance of
"a weapon" and still be far too large, or too close to a panel edge, for a `rotate` transform
specifically — the same bbox might be perfectly safe to `translate`. No check in the pipeline
evaluated that second question before segmentation/animation committed to it.

## Architecture change

Extended the existing validation stage (no new pipeline stage) — `validate_target` now runs a
third, deterministic, no-model-call check after the existing semantic check:
`validation/transform_geometry.py::check_transform_geometry(bbox, transform_kind, *,
panel_bbox_px, image_shape)`. Per-`TransformKind` bounds (`_TRANSFORM_GEOMETRY_PROFILES`) —
area fraction, edge-margin fraction, aspect ratio — each documented and derived from that
transform's own geometric mechanism, not one shared number. `ValidationResult` gains
`transform_compatible: bool | None`. `run_pipeline` now computes `panel_bbox_px` before
grounding (previously only before animation) and threads it into every `validate_target` call
so the geometry check's reference region is the object's real panel when known. Fail-closed
throughout: semantic mismatch → REJECT (geometry never runs); semantic match + geometrically
unsafe → REJECT; both pass → ACCEPT. No bbox is ever clipped to force a pass.

**Panel detection, the VLM, Grounding DINO, and SAM 2.1 were not touched.**

## Files changed

New: `src/manga_animation/validation/transform_geometry.py`,
`docs/decisions/0008-transform-aware-target-validation.md`. Modified:
`src/manga_animation/validation/validate.py`, `src/manga_animation/validation/__init__.py`,
`src/manga_animation/pipeline/types.py` (`ValidationResult.transform_compatible`),
`src/manga_animation/pipeline/orchestrator.py` (`panel_bbox_px` computed earlier),
`tests/test_validation.py`, `tests/test_pipeline.py`.

## Tests added

13 new tests across `tests/test_validation.py` (10) and `tests/test_pipeline.py` (3): valid
ROTATE candidate accepted; oversized ROTATE bbox rejected (the real defect, reproduced
directly); boundary-risk (edge-flush) ROTATE bbox rejected; extreme-aspect-ratio ROTATE bbox
rejected; `check_transform_geometry` is transform-specific; the identical bbox passes one
transform kind and fails another; panel-vs-page reference region produces different verdicts;
semantic rejection short-circuits before geometry ever runs; `transform_compatible` stays
`None` when an earlier check already rejected the candidate; a bbox flush against an edge is
still accepted for TRANSLATE (the false-rejection regression below); the historical
`flag_banner` case explicitly re-guarded; the orchestrator's candidate-retry loop triggered by
a *geometry* rejection specifically (not just semantic, as previously tested); a controlled
fallback plan rejected by geometry, not just semantics. All pass; all 256 pre-existing Phase
1–3.3 tests remain green.

## pytest / ruff / mypy

```
uv run pytest -q      -> 269 passed
uv run ruff check .   -> All checks passed!
uv run mypy src       -> Success: no issues found in 41 source files
```
Locally, and re-confirmed identically on the remote GPU worker after each of the two commits
below.

## Real E2E results (remote GPU worker, same session/environment as the main Phase 3.3 run)

Two rounds were needed — the second exists because the first round's re-verification itself
surfaced a real bug in the fix (see "A real false-rejection was found and corrected" below).

**Round 1** (commit `e03a788`, initial fix): `eval_weapon_effects.png` re-run automatically —
the same three grounding candidates as the original defect (rank 0/1: semantic REJECT,
unchanged; rank 2, previously the wrongly-ACCEPTed one): now correctly rejected —

```
rank=2: bbox covers 27.6% of its reference region, exceeding the 15% bound a rotate target
allows -- too large to safely animate without moving pixels outside the intended object
```

`phase3_action_page.png` (automatic): unchanged from the main run (VLM proposed `weapon`,
Grounding DINO found nothing above threshold — never reached validation, so this specific
automatic run doesn't exercise the fix either way). The historical `flag_banner` regression was
explicitly re-reproduced (reconstructed fallback plan, exact same real crop-verification
wording as Phase 3.2 and the main Phase 3.3 run: *"The crop shows a character's head and
dialogue box, not a flag banner"*) — still correctly REJECTed, unaffected (semantic check runs
first). `eval_static_dialogue.png`: still all-STATIC, as required. `sample_page_02.png` (the
Phase 3.2 positive control): all-STATIC again this session (a real, separate, already-documented
VLM nondeterminism finding — not caused by or related to this fix).

**A real false-rejection was found and corrected, before this fix could be accepted**: retrying
the hair positive case, `sample_page_01.png` (which *did* produce a usable `character_hair`
read this session) was **incorrectly REJECTed** —

```
bbox sits within 0.0% of its reference region's edge, closer than the 3% margin a translate
target needs to move without clipping against the boundary
```

Root cause: `TRANSLATE`'s initial `min_edge_margin_fraction=0.03` didn't account for the fact
that hair legitimately, normally starts flush against the top edge of a portrait-framed panel
— not a geometric defect. Corrected to `0.0` for `TRANSLATE` specifically (area fraction
remains the active bound for that kind; `ROTATE`/`SHEAR`/`SCALE` keep their margin requirements
— see ADR 0008's "Revision" for the full reasoning on why `TRANSLATE`'s real risk profile
doesn't need one). This was caught specifically *because* the task required re-confirming the
real positive case, not only the negative one — exactly the kind of check that catches an
overcorrection.

**Round 2** (commit `17f0ac2`, after the correction): `eval_weapon_effects.png` re-run again —
**still correctly rejected**, identical reason (27.6% bound), confirming the correction didn't
weaken the original fix. `sample_page_01.png` re-run — **now correctly COMPLETED**:
`character_hair`/`translate`, `semantic_match=True`, `transform_compatible=True`.

## Visual QA

`eval_weapon_effects.png`: both rounds produced **zero files** in the render output directory
(confirmed via the Jupyter Contents API — `content: []`) — no `output.mp4`, no frames. The
historical artifact (panel tilting, sound-effect text moving, black wedge artifacts) **cannot
occur**, not because a video was inspected and found clean, but because no video is ever
produced for a geometrically-rejected candidate — validation now runs strictly before
segmentation/animation/compositing/rendering, as designed.

`sample_page_01.png` (round 2, after the correction): frame 0 vs. frame 24 inspected directly
(decoded PNGs via the Jupyter Contents API). Hair visibly translates between frames (the same
real, correct motion visually confirmed earlier in the main Phase 3.3 run); the face, both
speech bubbles ("WAIT A MINUTE.", "EXCUSE ME?"), and the system-message box are pixel-identical
outside the animated region. `frame_0000.png`'s byte size (1,264,678 bytes) is identical to the
pre-fix render's frame 0, consistent with the rest pose being unaffected by this change.

## Is the historical weapon artifact actually eliminated?

**Yes.** Confirmed two ways: (1) the specific candidate that previously produced the artifact
(same grounding score, same crop, same semantic reasoning) is now deterministically REJECTed
before any pixel is ever transformed; (2) the render output directory is empty — there is no
video for the artifact to appear in. This is a structural guarantee (fail-closed before
rendering), not a statistical improvement.

## Remaining limitations

- The `_TRANSFORM_GEOMETRY_PROFILES` bounds are now grounded in **two** real, opposite-direction
  pieces of evidence (`ROTATE`'s area cap from the weapon defect; `TRANSLATE`'s margin removal
  from the hair false-rejection) — the other four kinds (`SHEAR`, `SCALE`, `MESH_WARP`, and
  `ROTATE`'s aspect-ratio bound) are still reasoned from mechanism alone, not independently
  observed real defects or false-rejections. Flagged, not hidden — see ADR 0008.
- No mask exists at validation time (segmentation runs after it) — these checks remain bbox-only
  by necessity, same limitation ADR 0008 already documents.
- `phase3_action_page.png`'s *automatic* run still never reaches validation (grounding finds
  nothing for whatever the VLM proposes each session) — the historical regression protection for
  this page is only ever confirmed via the explicit fallback-plan reproduction, not the
  automatic path, in every real run performed so far across Phase 3.2, the main Phase 3.3 run,
  and this fix.
- This fix does not (and was not asked to) address Phase 3.3's own main finding that
  panel-aware analysis showed no measurable benefit on n=5 — that remains open, unrelated,
  future work.

## Recommendation

**Phase 3.3 (including this 3.3.1 correction) can be accepted.** The real E2E visual defect
found during Phase 3.3's own evaluation run has been root-caused, fixed with a narrowly-scoped,
documented, tested architectural addition (no panel-detection/VLM/Grounding DINO/SAM changes),
and the fix itself was validated against both the negative case it targets (weapon: still
rejected) and the positive case it must not break (hair: initially broken by an
overly-strict bound, caught by the required re-verification, corrected, and re-confirmed
working) — real evidence for both directions, not just the one the brief called out. 269 tests
pass; ruff/mypy clean, locally and on the remote worker. Nothing has been committed to `main`;
everything above is on `phase-3.3-wip` pending explicit approval.

---

# Phase 3.3.2: evaluation oracle stabilization (ground-truth integrity)

Follow-up investigation, separate narrower brief: a serious evaluation-integrity concern was
raised after Phase 3.3.1 shipped — `sample_page_02`'s classification had changed between
independent sessions (once a usable `character_hair` PRIMARY read, twice all-STATIC), and the
central question was whether this project's evaluation ground truth could be silently redefined
by VLM output. See [ADR 0009](decisions/0009-evaluation-ground-truth-integrity.md) for the full
architectural decision; this section is the investigation record.

## Investigation approach

Per CLAUDE.md's standing policy (ADR 0002/0003), no model inference runs locally, and no
Jupyter/Kaggle URL may be guessed — none was available this session. Rather than block on a new
live GPU run, this investigation first inventoried the real, already-collected evidence already
in this repository: three independent real sessions against `sample_page_02.png`, each already
documented before this phase began:

| Session | When / commit | Result |
| --- | --- | --- |
| Phase 3.2 "Run 1" | `docs/phase3.2-results.md` | `PRIMARY: character_hair` (translate), grounded (score 0.610), validated ACCEPT (confidence 0.95), COMPLETED — real MP4 rendered |
| Phase 3.3 main run | commit `7d05bd2`, `outputs/experiments/phase3_3_evaluation_20260812T094106Z.json` (local, git-ignored per ADR 0002) | all-STATIC (page mode and panel mode both), and all-STATIC in all 3 repeated `analyze_page` calls in the same session's nondeterminism check |
| Phase 3.3.1 re-check | `outputs/experiments/phase3_3_1_recheck.json` (local, git-ignored) | all-STATIC |

This is real, multi-session evidence: the "usable" read happened exactly once; the all-STATIC
read has now been independently reproduced twice, in separate process/session boundaries
(matching the brief's Experiment 3.F distinction — this instability crosses session boundaries,
not only within-process repeated calls, which Phase 3.3's own nondeterminism harness already
showed were internally self-consistent within any one of these three sessions).

## Root cause

`Qwen25VLClient.generate()` (`src/manga_animation/analysis/client.py`) calls
`self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)` with no `do_sample`, no
`temperature`, and no fixed seed — this was already identified as a real, evidenced gap in
`docs/phase3.2-results.md` (the original `sample_page_01` flip) and repeated here for
`sample_page_02`: the model is free to sample differently across calls on byte-identical input,
and nothing in this codebase constrains that. This is category **(1) genuine VLM
stochasticity**, compounded by **(4) model/configuration**: the runtime never overrides
`transformers`' default generation config to make decoding deterministic. No evidence was found
for categories (2) prompt drift (the exact same `ANALYSIS_PROMPT` constant is reused
byte-for-byte across all three sessions — verified by reading `plan_builder.py`'s current
source, unchanged since Phase 3.1), (3) preprocessing differences (`_resized_for_vlm` is a pure
function of `config.resolution`, unchanged across these sessions), (5)/(6) caching or hidden
state (no cache layer exists anywhere in the analysis stage), or (7) an incorrect evaluation
assumption *in the metric computation itself* — `compute_metrics` was already comparing
predictions against stored `EvalSample` ground truth correctly (see "Architectural diagnosis"
below for the narrower, real gap that *does* exist).

**Experiment 3 (progressively freezing variables) was not run live this phase** — no remote GPU
worker was available, and per CLAUDE.md's explicit policy this project does not guess at one.
This is a real, disclosed limitation (see ADR 0009's "Open questions"), not a gap papered over:
the existing real evidence is sufficient to identify the *mechanism* (unpinned decoding) without
a new run, but does not by itself prove decoding-parameter pinning is sufficient to fully
eliminate the instability — that remains to be confirmed against real hardware in a future
phase, deliberately not attempted here.

## Scope: dataset-wide stability

Beyond `sample_page_02`, this project's own repeated-run nondeterminism harness
(`DEFAULT_NONDETERMINISM_SAMPLE_IDS` in `scripts/run_phase3_3_evaluation.py`) only ever covered
`sample_page_01`/`sample_page_02` — the two samples already flagged from Phase 3.2. The real
data already collected for the full 5-sample dataset (`docs/phase3.3-results.md`'s "Page-level
vs. panel-level comparison" table above) shows **every other sample's real outcome has stayed
directionally consistent across the sessions it was actually re-run in**
(`phase3_action_page`: grounding-stage failure, both the main run and Phase 3.3.1's re-check;
`eval_static_dialogue`: all-STATIC, both times; `eval_weapon_effects`: COMPLETED then, after the
Phase 3.3.1 geometry fix, correctly REJECTed at validation instead — an intentional, understood
change from the fix itself, not nondeterminism). **`sample_page_01` and `sample_page_02` remain
the only two samples in this dataset with real, evidenced cross-session instability** — both
already carry (after this phase's fix) `animation_possible: uncertain` /
`ground_truth_uncertain: true`. Extending the repeated-run harness to cover all 5 samples, on a
live worker, is flagged as natural future work, not attempted this phase (no live worker;
doing so also isn't required to fix the architectural gap this phase targets).

## Architectural diagnosis

`evaluation/metrics.py::compute_metrics` was already, and remains, structurally correct: every
metric is computed by comparing a `PageRunOutcome`/`RepeatedRunRecord` (a prediction) against an
`EvalSample` (`evaluation/dataset.py`, ground truth) — never the reverse, and no code path
anywhere in this project writes to `configs/phase3_3_eval_dataset.yaml` or constructs an
`EvalSample` from live VLM output. The real gap was narrower and one level removed from the
obvious form:

1. `EvalSample` was an ordinary **mutable** pydantic model — nothing enforced the immutability
   the architecture already implicitly relied on.
2. There was no versioning/audit signal on ground-truth fields beyond raw git history.
3. `sample_page_02`'s specific ground truth had **insufficient independent provenance**: its
   `animation_possible: "yes"` traced back to a single VLM classification, "confirmed" only by a
   pixel-diff of that same classification's own downstream render (real evidence the *rendering*
   pipeline worked correctly, not independent evidence that hair is *the* justified target on
   this page) — unlike `eval_static_dialogue`'s STATIC label, which cites a direct check of the
   source artwork itself.

## Changes made

See [ADR 0009](decisions/0009-evaluation-ground-truth-integrity.md) for full rationale.

- **`src/manga_animation/evaluation/dataset.py`**: `EvalSample` gains `model_config =
  ConfigDict(frozen=True)` (any mutation now raises `pydantic.ValidationError`) and a new
  `annotation_version: int` field (default `1`), plus an expanded module docstring making the
  ground-truth/prediction separation explicit.
- **`configs/phase3_3_eval_dataset.yaml`**: `sample_page_02`'s `animation_possible` revised
  `"yes"` → `"uncertain"`, `ground_truth_uncertain` `false` → `true`, `annotation_version`
  bumped to `2` — the exact same pattern this dataset already used for `sample_page_01`'s
  cross-session nondeterminism, not a new invented category. `expected_target_category`/
  `expected_motion_category`/`expected_region_note`/`regression_reference` nulled to match.
  Every other sample gets an explicit `annotation_version: 1` for auditability. A new header
  comment documents the versioning convention next to the data it governs.
- **`docs/decisions/0009-evaluation-ground-truth-integrity.md`**: new ADR.
- **`tests/test_evaluation.py`**: 7 new tests (see "Tests" below).
- No changes to `evaluation/metrics.py`, `evaluation/nondeterminism.py`,
  `evaluation/schemas.py`, or any pipeline stage — this phase's fix is entirely in how ground
  truth is stored and protected, not in how predictions are computed or compared.

## Ground-truth model

`EvalSample` (frozen, versioned) is this project's only representation of evaluation ground
truth. `PageRunOutcome`/`RepeatedRunRecord` remain the separate, ordinary-mutable representation
of what one real pipeline run actually produced (a prediction). `compute_metrics` always takes
both as separate arguments and only ever reads the `EvalSample` side as the fixed comparison
target. Changing ground truth now means: edit `configs/phase3_3_eval_dataset.yaml`, bump the
affected sample's `annotation_version`, and commit — an explicit, reviewed, git-tracked change.
No code path may do this at runtime; attempting to mutate a loaded `EvalSample` raises
immediately.

## Tests

7 new tests added to `tests/test_evaluation.py`:
`test_eval_sample_ground_truth_is_frozen`,
`test_eval_sample_construction_still_works_when_frozen`,
`test_compute_metrics_result_depends_only_on_stored_ground_truth_not_on_predictions`,
`test_repeated_evaluation_never_mutates_the_real_dataset_manifest`,
`test_real_dataset_ground_truth_changes_carry_an_explicit_annotation_version`,
`test_transform_geometry_failure_does_not_alter_semantic_ground_truth`,
`test_compute_metrics_is_a_pure_deterministic_function_of_its_inputs`.

```
uv run pytest -q      -> 276 passed
uv run ruff check .   -> All checks passed!
uv run mypy src       -> Success: no issues found in 41 source files
```

All 269 pre-existing Phase 1–3.3.1 tests remain green, unmodified.

## Reproducibility demonstration

`test_repeated_evaluation_never_mutates_the_real_dataset_manifest` runs `compute_metrics`
against the real dataset 5 times with deliberately conflicting synthetic predictions per sample
(alternating COMPLETED/FAILED for the same `sample_id`s) and asserts
`configs/phase3_3_eval_dataset.yaml`'s on-disk bytes are byte-identical before and after.
`test_compute_metrics_result_depends_only_on_stored_ground_truth_not_on_predictions` shows two
directly opposite predictions for the same sample_id, scored against the same fixed ground
truth, produce the expected opposite metric verdicts while the ground truth object itself
(`samples["hair_page"].animation_possible`) is provably untouched by either call.

## Remaining limitations

- Experiment 3 (deterministic decoding config: `temperature=0`/`do_sample=False`/fixed seed) was
  not run live — no remote GPU worker was available this session, and CLAUDE.md/ADR 0002/0003
  forbid guessing at one. The mechanism (unpinned decoding in `Qwen25VLClient.generate()`) is
  well-evidenced by three independent real sessions already in this repository, but pinning it
  and re-confirming stability is real future work, not done here.
- The repeated-run nondeterminism harness (`scripts/run_phase3_3_evaluation.py`) still only
  targets `sample_page_01`/`sample_page_02` by default — extending it to the full dataset on a
  live worker would strengthen "Scope" above from "no *evidenced* instability elsewhere" to "no
  instability elsewhere, actively tested," which is a real, honest, currently-open gap.
- `sample_page_02` no longer has a confident positive-control sample to replace it in this
  dataset. Establishing a new one requires actual human adjudication — of `sample_page_02`
  itself (direct visual inspection of `examples/sample_page_02.png` for a real drawn motion cue
  on the hair) or of a different page — explicitly left to the user, not resolved here.
- The pre-existing `_check_regression` implementation in `evaluation/metrics.py` only ever
  flags a *completed* outcome as a regression violation, regardless of which object was
  actually chosen — a real, separate, pre-existing limitation noticed while reviewing
  `sample_page_02`'s old `regression_reference` text (which asserted "even going all-STATIC
  would be a regression," a claim the code never actually checked). Out of this phase's scope
  (unrelated to ground-truth mutability/provenance) — flagged, not fixed.

## Is the evaluation oracle now stable enough to begin Phase 3.4?

**Ground-truth integrity: yes.** Ground truth is now immutable at the type level, versioned,
and the one sample with insufficiently independent provenance has been honestly re-labeled
uncertain rather than left silently overconfident. `compute_metrics` was already comparing
predictions against stored ground truth correctly; that guarantee is now enforced by
construction, not merely by convention, and is covered by regression tests.

**VLM prediction reliability: not resolved, and not in this phase's scope.** Two of five
dataset samples (`sample_page_01`, `sample_page_02`) still have genuinely unstable underlying
VLM reads — that is a real property of the current unpinned-decoding VLM client, honestly
recorded as `ground_truth_uncertain` rather than hidden, but it is a *prediction*-quality
problem, not a ground-truth-integrity one, and fixing it (decoding parameters, prompt work, or
otherwise) is explicitly out of this phase's brief. A future phase should treat "pin/measure
VLM decoding determinism" as a real prerequisite before leaning heavily on this dataset's
usable-target/STATIC rate metrics for go/no-go decisions.
