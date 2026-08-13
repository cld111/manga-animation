# 16. Phase 9: real-world quality evaluation infrastructure

Status: Accepted

## Context

Phase 8/8.3 (ADR 0014/0015) established that the pipeline works end-to-end and fixed the two
real visual defects found on the 7-sample golden regression set. The Phase 9 brief asks a
different question -- not "is the pipeline technically correct and safe" but "how good is the
animation on real manga pages, what does it reliably handle, where does it fail" -- and is
explicit that the existing 7-sample golden set (`configs/phase3_3_eval_dataset.yaml`) is
"primarily designed for regression/safety validation" and must not be read as a statistically
meaningful real-world benchmark. The brief also explicitly forbids optimizing the pipeline
during this phase ("measure the system we have before changing the system we have") and
requires reusing existing evaluation infrastructure rather than duplicating it.

A repository audit (this ADR's own prerequisite) found the real, already-implemented evaluation
stack this phase must build on: `evaluation.dataset.EvalSample`/`load_eval_dataset` (the ground
truth schema, ADR 0009), `evaluation.metrics.compute_metrics`/`classify_outcome` (the
PASS/PASS_WITH_FALLBACK/REJECTED/ERROR vocabulary, ADR 0014), `evaluation.schemas.PageRunOutcome`
(the per-sample outcome record), `rendering.compute_loop_metrics` (pixel+SSIM loop-continuity,
ADR 0014), and `scripts/run_phase3_3_evaluation.py` (the real, GPU-proven E2E driver). All of
this is reused, not reimplemented.

## Decision

**1. A separate Real-World Evaluation Dataset, same schema.** `configs/phase9_realworld_eval_dataset.yaml`
(10 new real pages, 8 MangaDex series never used by any earlier phase, sourced by crossing
MangaDex's "Full Color" tag with a genre-tag spread -- sports, fantasy action, mecha/sci-fi,
horror, office comedy, slice-of-life, gothic drama, hunter/action -- then manually visually
reviewing real candidate pages, same selection policy `fetch_phase3_sample_page.py`/
`fetch_phase3_3_eval_pages.py` already established). Reuses `EvalSample` as-is rather than a
parallel schema. Ground truth (`animation_possible`/`ground_truth_uncertain`) for every sample
was assigned by direct visual inspection of the actual downloaded image against the
`manga-analysis` skill's STATIC vs. ANIMATED checklist (motion lines, impact lines, flowing
linework) -- the same "human/AI visual inspection" provenance `EvalSample.animation_possible`'s
own docstring already sanctions, performed by the Claude Code assistant during this phase's
curation, not derived from any pipeline/VLM run on the image (the exact anti-pattern ADR 0009
exists to prevent). `evaluation.dataset.load_combined_eval_dataset()` loads and duplicate-checks
both manifest files together for reports that want the full 17-sample picture, without merging
or duplicating either file.

**2. Additive Phase 9 tag fields on `EvalSample`** (`scene_complexity_tags`,
`potential_motion_tags`, `geometric_difficulty_tags`, `motion_type_tags`,
`expected_difficulty`), each a closed `Literal` (`SceneComplexityTag`/`PotentialMotionTag`/
`GeometricDifficultyTag`/`MotionTypeTag`/`DifficultyLevel`) matching the brief's section 4
taxonomy verbatim -- same typo-safety rationale as `GoldenCategory`. All default to
empty/`None`, non-breaking for the existing golden dataset's entries. Descriptive
dataset-composition metadata, not ground truth (does not require bumping `annotation_version`).
`dataset_composition()` computes real per-tag sample counts (including real zero-coverage gaps,
not hidden) for the brief's required "dataset size, composition, categories" report field.

**3. Shared per-sample runner logic extracted to `evaluation.harness`.** Phase 9 needs the exact
same "run `run_pipeline` once, catch `PipelineStageError`/bare exceptions, build a
`PageRunOutcome`" logic `scripts/run_phase3_3_evaluation.py` already had -- duplicating it into
a second script would violate CLAUDE.md's "do not introduce a second parallel implementation".
`evaluation/harness.py` (`run_one_sample`, `run_nondeterminism_check`,
`render_summary_from_result`, `object_outcome_motion_type`, `panel_detection_evidence`,
`environment_metadata`) is now the one real implementation; both
`scripts/run_phase3_3_evaluation.py` and the new `scripts/run_phase9_evaluation.py` import it.
Deliberately **not** re-exported from `evaluation/__init__.py` -- every other module in that
package imports nothing torch/transformers-related (a real property this project's local test
suite relies on, since this checkout has no `ml` extras installed), and `harness.py` is the one
module that actually calls `pipeline.orchestrator.run_pipeline`. `pipeline.orchestrator` itself
turns out to be torch-safe at import time (every real model client lazily imports torch inside
its own methods, per the "GPU Awareness" architecture principle), so this exclusion is a
belt-and-suspenders architectural boundary, not a strict technical requirement -- kept anyway so
the "torch-free" promise holds by construction.

**4. One validated automated visual-artifact signal: `evaluation.artifacts.detect_seam_like_artifacts`.**
The brief warns against speculative CV metrics added merely to grow the metric count. Before
writing any production code, the real Phase 8 defect evidence still present locally
(`outputs/videos/phase8_evidence/*.mp4`, git-ignored) was used to test a hypothesis: does the
real "vertical seam" defect (ADR 0015, a SAM 2.1 mask over-segmenting into adjacent background
along one straight edge) leave a detectable signature in the final COMPOSITED RGB output alone,
independent of any access to the segmentation mask? Empirically: the seam defect's largest
changed-region component (frame 0 vs. a mid-cycle frame, connected-components analysis) showed
82-92% edge-touch on one side vs. 4.7-5.7% on the geometrically opposite side; the real,
*different* "duplicate silhouette" defect and a clean control both stayed under 21% on every
edge -- a wide, real margin. The check mirrors `segmentation.segment._validate_mask_shape`'s own
reviewed asymmetry logic (`_HUGGED_EDGE_THRESHOLD = 0.3`, `_OPPOSITE_EDGE_CEILING = 0.15`) but
runs as a second, independent, black-box QA layer over final rendered pixels, not a replacement
for the production-stage gate. Deliberately narrow: validated to catch the seam defect class
only, confirmed NOT to fire on the unrelated ghosting defect or a clean control -- not claimed
as a general artifact detector. Wired into `harness.run_one_sample` via a new
`RenderSummary.seam_artifact_suspected: bool | None` field (`PageRunOutcome.schema_version`
meaning 4: "also populates render_summary.seam_artifact_suspected").

**5. Visual-quality scoring protocol + capability matrix: `evaluation.visual_qa`.** The brief's
0-5 rubric (sections 8-9) as a typed `VisualQAScore` (8 fixed dimensions, `VISUAL_QA_SCALE`
giving every score value an explicit written definition), a `VisualFailureCategory` taxonomy
(section 12's visual failure list -- deliberately NOT duplicating `pipeline.types.Stage`, which
already adequately covers PIPELINE failures), and `CapabilityMatrixEntry`/
`build_capability_matrix` for section 13's matrix (every dimension defaults to `UNKNOWN` with no
evidence; a non-`UNKNOWN` verdict is rejected at construction time unless it cites real
`evidence_sample_ids`). Inter-rater reliability (section 10): this project has exactly one
available evaluator for Phase 9 (the Claude Code assistant performing direct visual inspection)
-- `VisualQAScore.evaluator` records this on every score so the limitation is visible in the
data itself; no inter-rater statistic is computed or fabricated.

**6. `scripts/run_phase9_evaluation.py`**: drives `harness.run_one_sample` over the Real-World
Evaluation Dataset (optionally also the golden set via `--include-golden`, for one
internally-consistent single-session combined report rather than stitching together two
sessions' JSON). Resumable (brief section 17): every sample's outcome is written to the output
JSON immediately after it completes, not only at the end; `--resume <path>` skips every
(sample_id, mode) pair and nondeterminism sample_id already present, so an interrupted run never
re-pays for completed real GPU inference. `--nondeterminism-runs` defaults to 2 (vs. the
existing script's 3) and `--nondeterminism-samples` defaults to only the new realworld
sample_ids (the golden set's nondeterminism is already characterized, Phase 3.3/7.2.2/Phase 8)
-- explicit cost-control choices per the brief's resource-efficiency requirement.

## Consequences

- Fully backward compatible: every `EvalSample`/`RenderSummary`/`PageRunOutcome` change is
  additive with a safe default; the golden dataset's own 7 samples and
  `scripts/run_phase3_3_evaluation.py`'s existing behavior are unaffected (only its internal
  implementation moved to `harness.py`, verified via full regression suite before/after).
- Two real, disclosed dataset gaps remain (mirroring the golden set's own two): no Real-World
  Evaluation Dataset sample is tagged `geometric_difficulty_tags: [partially_occluded]` or
  `motion_type_tags: [deformation]` -- occlusion and deformation remain a real, unclosed gap
  across the COMBINED dataset (both files), not fabricated around. See
  `configs/phase9_realworld_eval_dataset.yaml`'s own header for the full disclosure.
- `detect_seam_like_artifacts` is validated against exactly one confirmed-defective real
  instance (plus one different real defect and one clean control as negative controls) -- same
  "evidenced, not statistically calibrated" status as this codebase's other deterministic
  thresholds (e.g. `_SSIM_WRAP_TOLERANCE`, `_MAX_BBOX_EDGE_TOUCH_FRACTION`).
- No production pipeline behavior changed in this phase -- every change here is evaluation/QA
  infrastructure (`evaluation/`, `scripts/`, `configs/`), per the brief's "do not optimize
  during the baseline" rule. See `docs/phase9-results.md` for the actual evaluation run this
  infrastructure feeds.

## Open questions

- `detect_seam_like_artifacts`'s largest-component-only restriction (added after an initial
  all-components version produced a false positive on the real ghosting-defect video's own
  small, unrelated diff fragments) means it could in principle miss a real seam defect that
  happens not to be the single largest changed region in a given frame comparison (e.g. a small
  seam next to a much larger, healthy motion). Not observed in the real evidence gathered so
  far; left open pending more real defective examples.
- Whether `CapabilityMatrixEntry` verdicts should eventually be partially auto-suggested from
  `VisualQAScore`/`EvaluationReport` statistics (rather than always hand-constructed) is left
  open -- deliberately not attempted here, since turning aggregate stats into a
  WORKS_WELL/PARTIAL/FAILS judgment is exactly the kind of call the brief says must stay
  evidence-driven and human/AI-reviewed, not formulaic.
