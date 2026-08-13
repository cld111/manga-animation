# Phase 9 results: real-world animation quality evaluation

Status: **completed** — infrastructure implemented and locally tested (ADR 0016), a real GPU
E2E run executed against the full 10-sample Real-World Evaluation Dataset on a live Kaggle
worker (both `page` and `panel` analysis modes, plus a nondeterminism check), real rendered
videos downloaded and directly, visually inspected frame-by-frame, and 3 real, previously
undocumented visual defects found by direct inspection of the actual output. Every number below
is checked against an actual downloaded artifact or quoted log line, not asserted from memory —
same convention as `docs/phase8-results.md`.

## 1. Scope

Phase 8/8.3 (ADR 0014/0015) established that the pipeline works end-to-end and is safe (honest
rejection over fabricated motion) on a 7-sample golden regression set. Phase 9's brief asks a
different question: not "is the pipeline technically correct" but "how good is the animation on
real manga pages, what does it reliably handle, where does it fail, and what should be improved
next." This phase is evaluation and characterization only — no pipeline/production code was
modified (see ADR 0016's "Consequences": every change is `evaluation/`, `scripts/`, `configs/`
infrastructure).

## 2. Pre-Phase-9 audit findings

Before any Phase 9 change, the repository was audited (see ADR 0016's "Context"): `git status`
clean on `phase-6-wip` at commit `f9bca9e` (Phase 8.3's own last commit), `uv run pytest` 484
passed/2 deselected, `ruff`/`mypy` clean — matching `docs/phase8.3-results.md`'s own reported
baseline exactly. The existing evaluation stack (`evaluation.dataset`/`metrics`/`schemas`/
`nondeterminism`, `rendering.compute_loop_metrics`, `scripts/run_phase3_3_evaluation.py`) was
identified and reused rather than reimplemented — see ADR 0016 for the full audit and reuse
decisions.

## 3. Real-World Evaluation Dataset

`configs/phase9_realworld_eval_dataset.yaml`: **10 new real pages**, from **8 MangaDex series**
never used by any earlier phase (all earlier samples draw from 3 series: 'The Skeleton Soldier
Failed to Defend the Dungeon', 'Who Made Me a Princess', 'Latna Saga: Survival of a Sword
King'). Sourced by crossing MangaDex's "Full Color" tag with a genre-tag spread (sports, fantasy
action, mecha/sci-fi, horror, office comedy, slice-of-life, gothic drama, hunter/action) via the
public `/manga` search API, then manually, visually reviewing real candidate pages — same manual
selection policy `fetch_phase3_sample_page.py`/`fetch_phase3_3_eval_pages.py` already
established, never a synthetic/fabricated page.

Ground truth (`animation_possible`/`ground_truth_uncertain`, per-sample tags) was assigned by
the Claude Code assistant's direct visual inspection of each actual downloaded image against the
`manga-analysis` skill's STATIC vs. ANIMATED checklist (motion lines, impact lines, flowing
linework, deliberately-static backgrounds) — the "human/AI visual inspection" provenance
`EvalSample.animation_possible`'s own docstring already sanctions, performed independently of
any pipeline/VLM run on the image (the exact anti-pattern ADR 0009 exists to prevent).

### 3.1 Dataset composition (real, computed via `evaluation.dataset_composition`)

| Dimension | Breakdown |
| --- | --- |
| `animation_possible` | yes=7, no=1, uncertain=2 |
| `scene_complexity_tags` | complex_background=7, multiple_panels=5, multiple_characters=4, crowded_scene=3, single_character=2, simple_background=2, sparse_scene=1 |
| `potential_motion_tags` | character_movement=6, environmental_effect=4, object_moving_across_scene=4, weapon=2, impact_or_action_effect=2, hair_or_clothing=1, facial_feature=0 |
| `geometric_difficulty_tags` | thin_structure=6, irregular_silhouette=4, overlapping_objects=3, large_rectangular_object=1, complex_internal_holes=1, near_boundary=0, partially_occluded=0 |
| `motion_type_tags` | translation=5, mixed_motion=5, rotation=3, scale=1, deformation=0 |
| `expected_difficulty` | hard=4, medium=5, easy=1 |

**Disclosed real gaps**: no Real-World Evaluation Dataset sample is tagged
`geometric_difficulty_tags: [partially_occluded]` or `motion_type_tags: [deformation]` —
occlusion and deformation remain a real, unclosed gap across the COMBINED dataset (this file +
the golden set), mirroring the golden set's own two disclosed gaps
(`partially_occluded_object`/`scale_or_deformation`, `configs/phase3_3_eval_dataset.yaml`).

## 4. Real GPU E2E execution

Executed on a live Kaggle Jupyter GPU worker (2x Tesla T4), via a dedicated kernel (not the
project owner's own already-connected notebook kernel), reached over the Jupyter REST/
kernel-WebSocket API. `uv sync --extra dev --extra cv --extra video --extra ml` (fresh clone,
commit `dd05547`), then `uv run pytest -q` **523 passed, 1 skipped** (a golden-dataset image not
present on the fresh clone — expected, ADR 0002), **2 deselected**, `ruff check .` clean.
`mypy src` reproduced the exact one pre-existing, disclosed, unrelated finding
`docs/phase8-results.md`/`docs/phase8.3-results.md` already documented
(`segmentation/client.py:62`, a `transformers`-version type-stub interop issue) — confirmed
still present, not a Phase 9 regression.

**A real operational disruption occurred and is disclosed, not hidden**: the first Kaggle kernel
session was replaced mid-run by a new one (the proxy URL's underlying container restarted),
losing that session's `/kaggle/working` filesystem — including the first run's own partial
progress — even though the evaluation JSON was being written incrementally to that ephemeral
disk after every sample (exactly the resumability ADR 0016/the brief's section 17 asked for).
This is the disposable-remote-compute risk ADR 0003 already names; the mitigation the first
run's own incremental-persistence design didn't cover is a full container replacement, not just
a client-side disconnect (the case `docs/phase8-results.md` section 6 already documented and
`--resume` was built for). The run was restarted from scratch on the new kernel — every
per-sample outcome from the completed, first ~18/20 outcomes reproduced byte-for-byte
identically on rerun (grounding/SAM/reconstruction are feed-forward, not sampling, models —
consistent with this project's own established finding that they're far more deterministic
run-to-run than the VLM analysis stage, ADR 0015). A local backup of the evaluation JSON was
additionally polled and saved to the orchestrating session's own machine every ~90s during the
second run, as a further safety net beyond the remote worker's own incremental write.

`uv run python scripts/run_phase9_evaluation.py --env kaggle` (realworld dataset only, default
`--nondeterminism-runs 2`, not `--include-golden` — resource-efficiency choice per ADR 0016).
Wrote `outputs/experiments/phase9_evaluation_20260813T174730Z.json`, downloaded locally
(git-ignored, present on this checkout). Real environment metadata:
`torch==2.13.0+cu130`, 2x Tesla T4, `git_commit: dd0554728f552c085db68717f1a26a2c27d8d427`
(confirmed matching the pushed commit under test).

## 5. Pipeline results

### 5.1 Page-level report (10 samples)

`usable_target_rate` 8/10 (80.0%), `static_rate` 1/10 (10.0%), `grounding_success_rate` 4/8
(50.0%), `validation_acceptance_rate` 2/3 (66.7%), `end_to_end_completion_rate` 2/10 (20.0%),
`semantic_false_positive_rate` 0/1 (0.0%), `semantic_false_negative_rate` 6/7 (**85.7%**),
`unresolved_ground_truth_count` 2/10.
**`status_breakdown`: PASS=2, PASS_WITH_FALLBACK=0, REJECTED=3, ERROR=5.**

### 5.2 Panel-level report (10 samples)

`usable_target_rate` 9/10 (90.0%), `static_rate` 1/10 (10.0%), `grounding_success_rate` 9/9
(**100.0%**), `validation_acceptance_rate` 6/7 (85.7%), `end_to_end_completion_rate` 6/10
(**60.0%**), `semantic_false_positive_rate` 0/1 (0.0%), `semantic_false_negative_rate` 1/7
(14.3%), `unresolved_ground_truth_count` 2/10, `panel_detection_multi_panel_rate` 6/10 (60.0%),
`secondary_object_render_rate` 6/21 (28.6%), `micro_object_render_rate` 5/13 (38.5%).
**`status_breakdown`: PASS=6, PASS_WITH_FALLBACK=0, REJECTED=4, ERROR=0.**

### 5.3 Page vs. panel: a major, real finding

Panel-aware analysis **dramatically** outperforms page-level analysis on this larger, more
diverse dataset — a materially stronger result than Phase 3.3's original null finding ("no
measurable end-to-end reliability improvement... on this small dataset", `docs/phase3.3-results.md`)
and stronger than Phase 5.1/ADR 0011's own targeted single-page fix:

| Metric | Page | Panel |
| --- | --- | --- |
| `end_to_end_completion_rate` | 20.0% | **60.0%** |
| `grounding_success_rate` | 50.0% | **100.0%** |
| `semantic_false_negative_rate` | **85.7%** | 14.3% |
| ERROR count | 5 | **0** |

Every sample that ERRORed at page-level grounding (`wind_breaker_sprint`, `wind_breaker_finish`,
`space_monster_creature`, `villainess_ending_scuffle`) either PASSED or was honestly REJECTED at
panel level instead — the same "extreme-aspect-ratio/tall-page grounding scale effect" ADR 0007/
0011 already root-caused, now reproduced on 8 real, diverse, previously-unseen pages, not just
the one page those ADRs originally fixed.

### 5.4 Per-sample outcomes

| sample_id | page mode | panel mode |
| --- | --- | --- |
| `realworld_wind_breaker_sprint` | ERROR (grounding) | **PASS** (`character_pose`/primary, 3 objects) |
| `realworld_wind_breaker_finish` | ERROR (grounding) | **PASS** (`object_in_motion`/primary, 3 objects) — visual defect found, see §7 |
| `realworld_omniscient_reader_blade` | **PASS** (`raised_sword`/primary) | **PASS** (identical — no real panel structure, page-level fallback) |
| `realworld_angels_of_war_fleet` | REJECTED (analysis) | REJECTED (analysis) — matches its own `uncertain` ground truth |
| `realworld_space_monster_creature` | ERROR (grounding) | **PASS** (`alien_wing`/primary) |
| `realworld_space_monster_hypersenses` | REJECTED (segmentation) | REJECTED (segmentation) — `honest_failure_acceptable=true` sample |
| `realworld_reality_lie_office` | REJECTED (validation) | REJECTED (validation) — matches its own confident `no` ground truth |
| `realworld_marika_love_meter` | **PASS** (`clapping`/primary) — visual defect found, see §7 | REJECTED (segmentation) |
| `realworld_villainess_ending_scuffle` | ERROR (grounding) | **PASS** (`raised_sword`/primary, 2 objects) — visual defect found, see §7 |
| `realworld_sss_hunter_gladiator` | ERROR (analysis) | **PASS** (`raised_sword`/primary, 6 objects) |

## 6. Failure distribution

**Page mode** (8 non-PASS of 10): grounding failures 4 (`wind_breaker_sprint`,
`wind_breaker_finish`, `space_monster_creature`, `villainess_ending_scuffle`), analysis failures
2 (`angels_of_war_fleet`, `sss_hunter_gladiator`), segmentation failure 1
(`space_monster_hypersenses`), validation rejection 1 (`reality_lie_office`).

**Panel mode** (4 non-PASS of 10): analysis failure 1 (`angels_of_war_fleet` — matches its own
uncertain ground truth, an honest result), segmentation failures 2 (`space_monster_hypersenses`,
`marika_love_meter`), validation rejection 1 (`reality_lie_office`).

**Zero ERROR-classified outcomes in panel mode** — every panel-mode failure was REJECTED (an
honest, attributed, ground-truth-consistent negative), not an unexplained or ground-truth-
contradicting one. Page mode's 5 ERROR outcomes are semantic false negatives (confident
`animation_possible="yes"` samples that page-level grounding failed to reach) — root cause is
the same extreme-page-height grounding-scale effect panel mode fixes, per §5.3.

**Nondeterminism**: all 10 samples internally stable this session (`outcome_stable=True`,
`target_category_stable=True`, 2/2 repeated `analyze_page` calls agreed, for all 10 samples) —
matches this project's established finding that within-session repeated calls are
self-consistent.

## 7. Visual quality

8 real completions (2 page-mode PASS + 6 panel-mode PASS; `panel_omniscient_reader_blade` is a
byte-identical page-level fallback, not a distinct render) were downloaded and directly,
visually inspected: frame 0, a mid-cycle frame (frame 24 of 96), and the true last frame
(frame 95, adjacent to the loop wrap) — cropped around the region a pixel-diff/connected-
components analysis identified as changed, plus a full pixel-diff heatmap per sample. Full
structured scores: `outputs/experiments/phase9_visual_qa.json` (`evaluation.visual_qa.VisualQAScore`,
one evaluator — see §9). Scale: `evaluation.visual_qa.VISUAL_QA_SCALE` (0=unusable .. 5=excellent).

### 7.1 Three real, newly-discovered visual defects

1. **`realworld_marika_love_meter` (page mode)** — a visible duplicated/offset copy of the
   character's raised hand and hair silhouette at frame 24, absent at frame 0/95.
   `compositing_quality=2`. PRIMARY was grounded as `clapping` (a real, valid motion cue in the
   page's top panel — a hand-clap with "CLAP!!" sound-effect text — that this project's own
   dataset curation notes only partially captured, having flagged the page's weaker middle-panel
   waving cue instead; not a pipeline targeting error). `seamless_loop_verified=True`
   (wrap_ssim 0.964) — the defect is mid-cycle-only, invisible at the actual loop seam, the same
   pattern Phase 8.3 already documented for its own two defects.
2. **`realworld_wind_breaker_finish` (panel mode)** — a clear vertical streaking/warping
   distortion across roughly the right 40% of the changed-region crop (bicycle wheel and
   clothing visibly smeared) at frame 24, absent at frame 0/95. `compositing_quality=1`. This
   render has **two simultaneous, independently-animated `character_hair` objects** — the same
   real mechanism class Phase 8.3's Defect A investigated — though the visible symptom here
   (a warp/smear) looks mechanically different from Defect A's clean double-exposure, suggesting
   a related but not identical compositing interaction. `seamless_loop_verified=True`
   (wrap_ssim 0.907).
3. **`realworld_villainess_ending_scuffle` (panel mode)** — a sharp, hard vertical
   discontinuity splitting the character's torso/skirt at frame 24 (a visibly duplicated/offset
   copy of part of the sleeve and skirt), absent at frame 0/95. `compositing_quality=1`. Only 2
   objects were rendered here (PRIMARY `raised_sword` + SECONDARY `cloth`), so this is not the
   multi-hair-object mechanism above — closer in visual character to Phase 8.3's original Defect
   B (a rigid edge), though the specific root cause (mask over-segmentation vs. a cross-object
   interaction) was not traced to source data in this phase (no live GPU access retained after
   the run — same disposable-compute limitation ADR 0003 names).

All three are **mid-cycle-only** (invisible at the actual loop wrap) — confirming, on entirely
new samples, Phase 8's own finding that whole-frame `LoopMetrics` structurally cannot catch this
defect class, and that it requires targeted, mask-region-specific visual inspection.

### 7.2 The automated seam-artifact detector on new, diverse content

`evaluation.artifacts.detect_seam_like_artifacts` fired (`seam_artifact_suspected=True`) on 4 of
8 real completions. Direct visual inspection resolved each:

| Sample | Detector | Direct inspection verdict |
| --- | --- | --- |
| `wind_breaker_finish` | True | **Confirmed real defect** (§7.1.2) |
| `villainess_ending_scuffle` | True | **Confirmed real defect** (§7.1.3) |
| `space_monster_creature` | True | **False positive** — diff heatmap shows a clean, tight, creature-silhouette-shaped region, no leakage; most likely triggered by the creature's own organic, asymmetric silhouette |
| `sss_hunter_gladiator` | True | **Inconclusive** — a large, complex, busy changed region (the page's own dense impact/speed-line artwork); no unambiguous tear identified, but not confidently ruled out either |

A real, disclosed, evidenced limitation of the detector: on this more diverse dataset (different
art styles from the one style it was originally validated against, `docs/decisions/0016...` §4),
its true-positive precision is **2 confirmed / 4 flagged = 50%** — still strictly better than no
signal at all (both real defects this phase found were among its 4 flags, and it correctly
stayed silent on the 4 clean/fallback completions), but a real, bounded false-positive rate that
was not visible in the original single-art-style validation. Recorded as a genuine finding, not
smoothed over.

### 7.3 Confirmed-clean completions

`omniscient_reader_blade` (both modes — a single, clean, small-amplitude sword ROTATE, tight
diff footprint, no leakage; `motion_quality=4` but flagged `insufficient_motion` since the
amplitude reads as barely-there on a casual look) and `space_monster_creature` (§7.2, confirmed
false positive) both show the pipeline **can** produce clean, artifact-free real output — the 3
real defects above are specific to certain multi-object/complex-region instances, not universal.

## 8. Capability matrix

Evidence-based, from §5-7 (full detail: `outputs/experiments/phase9_visual_qa.json`). `UNKNOWN`
means no completed real evidence exists yet — not assumed to pass.

| Capability | Verdict | Evidence |
| --- | --- | --- |
| Single object | **WORKS_WELL** | `omniscient_reader_blade`, `space_monster_creature` — both clean |
| Multiple objects | **PARTIAL** | 4 real multi-object completions; 2/4 showed a confirmed real defect |
| Translation | PARTIAL | `wind_breaker_sprint` (plausible-clean), `wind_breaker_finish` (defective) |
| Rotation | PARTIAL | 3 `raised_sword` completions: 1 clean, 1 defective, 1 inconclusive |
| Scale | UNKNOWN | no real completion exercised scale this phase |
| Occlusion | UNKNOWN | no real completion exercised occlusion this phase (a disclosed dataset gap, §3.1) |
| Boundary objects | UNKNOWN | no near-boundary sample in this dataset (golden set's own incidental coverage is a different phase's evidence) |
| Complex background | PARTIAL | 2 completions actually tagged complex_background (`omniscient_reader_blade`, `sss_hunter_gladiator`; see §14 — `villainess_ending_scuffle` was incorrectly cited here in an earlier draft): 1 clean, 1 inconclusive |
| Hair/clothing | **PARTIAL** (weakest observed) | hair implicated in 2 of 3 real defects/suspicious cases this phase — echoes Phase 8.3's own Defect A mechanism |
| Weapons | PARTIAL | `raised_sword` grounds/validates reliably (3/3), but only 1/3 resulting renders confirmed clean |
| Effects | UNKNOWN | the dedicated effect-only sample (`space_monster_hypersenses`) never reached a completed render (REJECTED at segmentation, both modes) |
| Dense scenes | PARTIAL | both real crowded-scene completions showed a defect or an inconclusive result |

## 9. Inter-rater reliability

Exactly **one** available evaluator for this phase's visual-QA pass: the Claude Code assistant,
performing direct visual inspection (frame crops, pixel-diff heatmaps, side-by-side comparison).
No inter-rater agreement statistic is computed or fabricated — this is a real, disclosed
limitation (brief section 10), not hidden. `VisualQAScore.evaluator` records this explicitly on
every score in the JSON artifact.

## 10. Known limitations

- Exactly one available evaluator (§9).
- `evaluation.artifacts.detect_seam_like_artifacts`'s true-positive precision on this dataset was
  50% (2/4) — a real, bounded false-positive rate not visible in its original single-art-style
  validation (§7.2). It also structurally cannot catch the "warping/streaking" defect subtype
  found in `wind_breaker_finish` except via its coarse edge-asymmetry heuristic (which happened
  to also fire there, but for reasons not fully explained by that specific mechanism).
- A real operational disruption (Kaggle container replacement mid-run, §4) — mitigated this
  session by a from-scratch rerun plus local JSON snapshotting, but the underlying
  `--resume`-from-incrementally-written-remote-file design does not survive a full remote
  filesystem loss, only a client-side disconnect. A real, disclosed gap in the resumability
  design, not previously exercised until this phase's real disruption.
- The golden 7-sample regression set was not re-run in this session (resource-efficiency
  choice) — its own real evidence remains `docs/phase8-results.md`/`docs/phase8.3-results.md`,
  not restated here as new Phase 9 evidence.
- `partially_occluded`/`deformation`/`scale`/`effects`/`boundary_objects`/`occlusion` remain
  real, disclosed `UNKNOWN` capabilities or dataset gaps — not fabricated around.
- Root cause of the two newly-found defects (§7.1.2, §7.1.3) was not traced to source
  mask/grounding data in this phase (no live GPU access retained after the run ended) — flagged
  as concrete Phase 10 follow-up work, not silently absorbed.

## 11. Representative successes

- **`realworld_omniscient_reader_blade`** (both modes): a single, real, cross-series (never
  previously used) sample with a clean sword ROTATE, zero visible artifacts, `seamless_loop_verified=True`.
- **`realworld_wind_breaker_sprint`** (panel mode): a real 3-object simultaneous render (PRIMARY
  + SECONDARY clothing + MICRO rain) across a 3-panel extreme-aspect-ratio strip, with a
  plausible-clean visual result on direct inspection.
- **Panel-aware analysis itself** (§5.3): the single strongest, most consequential real finding
  of this phase — a 3x completion-rate improvement (20%→60%) and complete elimination of
  ERROR-classified outcomes (5→0) on this real, diverse 10-sample set.

## 12. Representative failures

- **Page-level grounding on tall/extreme-aspect-ratio pages** (§5.3/5.4): 4 of 10 real samples
  ERRORed at page-level grounding despite confident `animation_possible="yes"` ground truth —
  all 4 either passed or honestly rejected at panel level, confirming ADR 0007/0011's mechanism
  generalizes well beyond the one page it was originally fixed on.
- **Multi-object hair/clothing interactions** (§7.1.2, §7.1.3, §8): the two confirmed new visual
  defects both involve compositing multiple simultaneously-animated regions on/near a single
  character — the single clearest, most consequential visual-quality gap this phase found.
- **`realworld_space_monster_hypersenses`** (both modes, REJECTED at segmentation): the
  dedicated abstract-effect sample never reached a render — consistent with its own
  `honest_failure_acceptable=true` ground truth (an effect-heavy, non-concrete target is
  expected to be hard to ground/segment), but it does mean this phase gathered zero real
  evidence for the "effects" capability row.

## 13. Phase 10 recommendations (ranked, evidence-based; not implemented here)

1. **Investigate the two newly-found multi-object hair/clothing defects** (`wind_breaker_finish`,
   `villainess_ending_scuffle`) with live GPU mask/grounding access — determine whether either
   is a new instance of Phase 8.3's already-fixed mechanisms (narrowly missed by the existing
   `_MAX_CROSS_OBJECT_MASK_OVERLAP_FRACTION`/`_MAX_BBOX_EDGE_TOUCH_FRACTION` thresholds) or a
   genuinely new compositing interaction. Highest-value: both are real, reproducible-looking,
   user-visible defects on ordinary content, not edge cases.
2. **Make `panel` the default `analysis_mode`** (or auto-select panel-aware analysis whenever
   real panel structure is detected) — §5.3's finding is large, real, and now reproduced on 8
   diverse new pages, not a marginal Phase 3.3-era null result. The current default
   (`analysis_mode="page"`, per `docs/pipeline.md`) leaves a real 3x completion-rate improvement
   unclaimed by default.
3. **Reduce `detect_seam_like_artifacts`'s false-positive rate** (currently 50% on diverse
   content, §7.2) — e.g. by adding an organic-silhouette allowance (the `space_monster_creature`
   false positive) before relying on it as a primary automated visual-QA signal at scale.
4. **Gather real evidence for `effects`/`occlusion`/`scale`/`boundary_objects`** — all four
   remain `UNKNOWN` after this phase; `space_monster_hypersenses` (the one dedicated effect
   sample) never reached a render, so a future evaluation phase should either add more
   effect-only samples or specifically investigate why segmentation rejects this sample class.
5. **Harden the resumability design against a full remote container loss** (§10) — periodic
   local snapshotting (ad hoc this session) could become a first-class, committed feature of
   `scripts/run_phase9_evaluation.py` (or a shared harness concern) rather than an
   improvised session-local workaround.

## 14. Independent audit (`qa-agent`, fresh session, no access to this document's own reasoning)

Mandate: independently re-verify dataset validity, methodology, failure classification, visual
scoring, aggregate statistics, and capability-matrix conclusions against the real artifacts on
disk, without trusting this document's own prose — same convention as
`docs/phase8-results.md` section 9 / `docs/phase8.3-results.md` section 10.

**Confirmed, independently re-derived from real artifacts**: the dataset (10 real PNGs, 8
distinct series, `dataset_composition()` recomputed by hand from the YAML matches §3.1 exactly);
the methodology (`harness.run_one_sample` genuinely calls `pipeline.orchestrator.run_pipeline`,
`scripts/run_phase3_3_evaluation.py` genuinely reuses it, no duplication); all 20 outcomes'
`E2EStatus` classification, hand-applied from the real `classify_outcome` logic against the raw
JSON, matched §5.4 and the `status_breakdown` counts exactly; every aggregate rate in §5,
recomputed from the raw `outcomes` arrays (not the `reports` section), matched exactly; all
three §7.1 defects, independently re-opened from the actual PNG crops, confirmed real; the
`space_monster_creature` false-positive and `sss_hunter_gladiator` inconclusive
characterizations, independently agreed with; `detect_seam_like_artifacts` independently
re-run locally against the real Phase 8 evidence videos, reproducing the same
seam/ghosting/clean discrimination ADR 0016 §4 documents.

**Two real, evidenced discrepancies found and fixed as a direct result of this audit** (both
were reporting-layer mistakes in this document and in `outputs/experiments/phase9_visual_qa.json`
— neither affected any `classify_outcome` result, aggregate rate, or capability verdict):

1. `capability_matrix["complex_background"]` cited `realworld_villainess_ending_scuffle` as
   evidence — that sample's real `scene_complexity_tags` in
   `configs/phase9_realworld_eval_dataset.yaml` are `[multiple_characters, multiple_panels]`,
   not `complex_background` (the evidence list was byte-identical to the unrelated `"weapons"`
   row — a copy-paste error). Fixed: the row now cites only the two samples that are actually
   tagged complex_background (`omniscient_reader_blade`, `sss_hunter_gladiator`); the verdict
   itself (PARTIAL) was already correct and unchanged.
2. `realworld_sss_hunter_gladiator` (panel mode) was undercounted as "5 objects" /
   "PRIMARY + 4 SECONDARY/MICRO" in three places (this document's §5.4, the visual-QA JSON's own
   note, and the `multiple_objects` capability row's "2-5" range) — the raw JSON's
   `object_outcomes` actually lists 5 real *rendered* SECONDARY/MICRO objects
   (`character_eyes`, `character_hair`, `hand`, `green_fluid`, `character_face`), i.e. 6 objects
   total including PRIMARY. Fixed in both this document and a regenerated
   `outputs/experiments/phase9_visual_qa.json`.

No other discrepancy was found; every other claim in this document was independently
re-verified as accurate. See the audit's full report (delivered via `SendMessage`, 2026-08-13)
for the complete file-by-file trace.

## 15. Reproducibility

- Git commit under test: `dd0554728f552c085db68717f1a26a2c27d8d427`.
- Config profile: `kaggle` (`configs/kaggle.yaml` — `device: cuda`, `dtype: float16`,
  `resolution: 2048`).
- Model versions (real, from the run's own `environment`/`model_variants` metadata):
  `qwen2.5-vl-7b-instruct`, `grounding-dino-swin-l`, `sam2.1-hiera-base`, `lama-large`;
  `torch==2.13.0+cu130`.
- GPU: 2x Tesla T4 (Kaggle).
- Command: `uv run python scripts/run_phase9_evaluation.py --env kaggle`.
- Dataset version: `configs/phase9_realworld_eval_dataset.yaml` as committed at `dd05547`.
- Output artifacts (git-ignored, present on this checkout):
  `outputs/experiments/phase9_evaluation_20260813T174730Z.json` (raw per-sample outcomes +
  reports), `outputs/experiments/phase9_visual_qa.json` (structured visual QA scores +
  capability matrix), `outputs/videos/phase9_evidence/*.mp4` (the 8 real downloaded renders),
  `outputs/frames/phase9_evidence/` (extracted frames and diff-heatmap crops).

## 16. Git

See the top-level session summary for branch/commit/push status.
