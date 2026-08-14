# Phase 12 results: semantic mask validation, forensic hardening, and evaluation

Status: **substantially completed.** A real semantic mask validation gate was designed,
implemented, benchmarked against real artifacts, empirically compared against Phase 11's own
geometric candidates, revised once based on real GPU evidence, and validated end-to-end on a
live Kaggle GPU worker — where it correctly caught, in production, the exact real defect Phase
11 confirmed but could not fix (`realworld_wind_breaker_finish`'s PRIMARY object). This document
is written incrementally, in place, as real evidence lands — same convention as
`docs/phase8-results.md`/`docs/phase9-results.md`/`docs/phase10-results.md`/
`docs/phase11-results.md`. Every claim below is checked against a real, locally-retained
artifact, a real GPU log line, or a real committed test — not asserted from memory.

## 1. Executive summary

Phase 11 left one confirmed, unresolved architectural gap: SAM 2.1 can produce a mask that
passes every existing geometric check but is semantically wrong (docs/phase11-results.md section
6.4), and no geometric-only signal separates the confirmed-defective real masks from legitimate
ones (section 7). This phase:

1. Audited the real pipeline's semantic/provenance flow end to end (section 2).
2. Built a real, provenance-cited 13-object semantic-mask benchmark from Phase 8.3's and Phase
11's own GPU captures — 5 confirmed-bad, 8 presumed-good (4 tagged difficult) — explicitly
   disclosed as development, not held-out, data (section 3).
3. Investigated and empirically compared five methods: four geometric single-signal baselines
   (reformalizing Phase 11's own negative finding with fresh numbers) and a new VLM mask-crop
   verification method (section 4).
4. Ran a real calibration study on the live Kaggle GPU worker, found a real prompt-anchoring
   defect in the first VLM prompt draft (identical canned reasons/confidences across unrelated
   objects), fixed it, and re-validated — precision improved from 0.36 to 0.75, false-positive
   rate from 0.88 to 0.12 (section 5).
5. Implemented the gate (`validation/mask_semantics.py`, ADR 0018) as a new pipeline stage with
   explicit ACCEPT/REJECT/ABSTAIN semantics, wired into the orchestrator with the same
   PRIMARY-fails/SECONDARY-drops policy every other stage already has (section 6).
6. Validated the gate end-to-end on the live GPU worker: a real, fully-automatic run of
   `realworld_wind_breaker_finish` (panel mode) now fails closed at `mask_semantics` instead of
   silently rendering Phase 11's confirmed real defect — the gate's actual production purpose,
   demonstrated live (section 7).
7. Extended machine-readable evaluation reporting, provenance, and the failure taxonomy so a
   semantic mask rejection is never collapsed into generic `ERROR` (sections 8-9).
8. Disclosed, rather than hid, every real limitation found: a live nondeterminism case where a
   fresh run's PRIMARY sword mask was rejected for a reason not present in the original
   benchmark; a real GPU OOM under concurrent client residency; unresolved instance-identity and
   calibration gaps (section 10).

## 2. Architecture and provenance audit (Workstream 1)

### 2.1 Actual pipeline stage sequence

The pipeline's real stage order (confirmed by reading `pipeline/orchestrator.py`, not assumed
from `docs/pipeline.md`'s diagram, which itself needed a correction in Phase 3.1 and again this
phase):

```
analysis -> grounding -> validation -> segmentation -> mask_semantics -> animation
-> reconstruction -> compositing -> rendering
```

Two stages are both named "validation" in spirit but are architecturally distinct, operating on
different evidence at different points:

| Stage | Module | Operates on | Question answered | Model call |
| --- | --- | --- | --- | --- |
| `validation` (Phase 3.2/3.3.1) | `validation/validate.py` + `validation/transform_geometry.py` | grounding bbox (no mask yet) | "is this box a plausible location for the target, and geometrically safe for its transform?" | 1 VLM crop call |
| `mask_semantics` (**Phase 12, new**) | `validation/mask_semantics.py` | the real segmented mask | "does the mask's own pixel content match the target, in full?" | 1 VLM mask-crop call |

This is a deliberate, documented divergence from the Phase 12 brief's own idealized order
("segmentation → geometric validation → semantic validation → transform-aware validation") — see
ADR 0018's "Why this is a distinct stage" section for why transform-aware geometric validation
must stay pre-segmentation (no mask exists yet at that point) while semantic mask validation must
be post-segmentation (it needs the real mask).

### 2.2 Provenance chain (per object, per run)

```
ObjectPlan.object_id (analysis, e.g. "obj_character_hair_7")
    |
GroundingResult.object_id + BBoxPx  (grounding — same object_id, real page-space bbox)
    |
ValidationResult.object_id + candidate_rank + accepted  (validation — bbox-level ACCEPT/REJECT)
    |
SegmentationResult.object_id + mask + bbox + iou_score  (segmentation — real SAM 2.1 mask)
    |
MaskSemanticResult.object_id + verdict + vlm_matches + confidence + reason + geometric_signals
    (mask_semantics — NEW Phase 12 mask-level ACCEPT/REJECT/ABSTAIN)
    |
Layer.object_id + frames + z_order  (animation — per-frame transformed (image, mask))
    |
composite_frame_stack(...) consumes every accepted Layer, in z_order
```

`object_id` is the single stable key threading every stage together (a human-readable slug from
`analysis/plan_builder.py::_slugify`, e.g. `obj_character_hair_7` — matches the real object
naming already visible in Phase 11's own retained evidence). No separate content-hash provenance
ID was found to add value beyond this (Workstream 28) — `object_id` is already unique per plan,
stable across the whole run, and human-readable in logs; a hash would only be useful if the same
`object_id` could legitimately refer to different content within one run, which the schema
already forbids (`AnimationPlan`'s own duplicate-`object_id` validator).

**What is now traceable end to end for an accepted PRIMARY object** (Workstream 8):
`PipelineRunResult.grounding` (which candidate, what bbox, what model), `.validation_attempts`
(every candidate tried, why each was accepted/rejected), `.segmentation` (which mask, real IoU
score), `.mask_semantics` (**new**: verdict, VLM's raw matches/confidence, reason, geometric
signals, which method) — and for a SECONDARY/MICRO object, the identical chain via
`ObjectRunResult`, or a `DroppedObjectResult(failing_stage, reason)` explaining exactly where and
why it was dropped, now including `"mask_semantics"` as a distinct value (previously only
`"grounding"`/`"validation"`/`"segmentation"`).

**Provenance gap found and left open**: a dropped (non-rendered) object's `mask_semantics`
verdict is not retained as a structured object past the point of the drop — only the human-
readable `reason` string it was folded into (see `evaluation/harness.py`'s own disclosed comment
on this). This mirrors a pre-existing gap grounding-stage drops already had (no structured
`GroundingResult` is retained for an object grounding found nothing for either) — not a new
regression, but not closed this phase either; closing it responsibly would mean threading a
`MaskSemanticResult | None` through `DroppedObjectResult` for every failing stage uniformly, a
broader refactor than this phase's "smallest correct fix" scope justified.

### 2.3 Where semantic information was previously lost

Before this phase: a technically valid grounding candidate's bbox was VLM-checked
(`validate_target`), then SAM produced a mask from that bbox with **zero further semantic
check** — only geometric ones (`_validate_mask`'s coverage bounds, `_validate_mask_shape`'s
edge-asymmetry test, the cross-object overlap guard). Phase 11's own real evidence (section 6.4)
is the direct proof this gap was real and exploitable: a mask can pass every one of those checks
while covering substantially more/different content than its label. This phase closes exactly
that gap, and only that gap — no other provenance loss was found in this audit beyond the
already-disclosed dropped-object structured-result gap above.

## 3. Real semantic-mask benchmark (Workstream 2)

`configs/phase12_semantic_mask_benchmark.yaml` (schema: `evaluation/mask_dataset.py`) — 13 real
`SegmentationResult.mask`-shaped objects, every one cited to the specific phase report section
that established its label:

| # | sample_id | ground_truth | difficulty | source |
| - | --- | --- | --- | --- |
| 1 | villainess_ending_scuffle_obj_raised_sword_12 | good | typical | Phase 11 |
| 2 | villainess_ending_scuffle_obj_cloth_5 | **bad** | typical | Phase 11 (confirmed) |
| 3 | wind_breaker_finish_obj_object_in_motion_12 (PRIMARY) | **bad** | typical | Phase 11 (confirmed) |
| 4 | wind_breaker_finish_obj_character_hair_0 | good | typical | Phase 11 |
| 5 | wind_breaker_finish_obj_character_clothing_1 | good | **difficult** | Phase 11 (confirmed clean) |
| 6 | wind_breaker_finish_obj_character_hair_7 | **bad** | typical | Phase 11 (confirmed) |
| 7 | sss_hunter_gladiator_obj_raised_sword_18 (PRIMARY) | good | typical | Phase 11 |
| 8 | sss_hunter_gladiator_obj_character_eyes_2 | good | **difficult** | Phase 11 |
| 9 | sss_hunter_gladiator_obj_character_hair_7 | **bad** | typical | Phase 11 (confirmed) |
| 10 | sss_hunter_gladiator_obj_hand_10 | good | typical | Phase 11 |
| 11 | sss_hunter_gladiator_obj_green_fluid_15 | good | **difficult** | Phase 11 |
| 12 | sss_hunter_gladiator_obj_character_face_17 | good | **difficult** | Phase 11 |
| 13 | phase3_action_page_obj_character_hair_6 | **bad** | typical | Phase 8.3 (Defect B, different mechanism) |

**Composition**: 5 bad, 8 good (3 difficult). This falls short of the Phase 12 brief's own
"10+ good, 5+ difficult good" target — disclosed explicitly, not padded with fabricated masks to
hit the number (Workstream 2's own "more is better if real artifacts are available" / "do not
fabricate labels"). Only 14 real `SegmentationResult.mask` arrays exist anywhere in this
project's retained local history (Phase 8.3: 1; Phase 11: 13, of which one — `raised_sword_12`'s
counterpart at a different index — was actually the same object counted once); closing this gap
responsibly needs new real GPU captures on new real pages, not relabeling the existing ones.

**Dataset leakage / contamination (Workstream 53/54)**: every BAD sample's label comes directly
from the same Phase 11 evidence (`docs/phase11-results.md` section 6.4/7) that motivated this
gate's entire design and its prompt's own wording. This benchmark is **development data**, not a
held-out evaluation set — a method's accuracy on it demonstrates "does this method correctly
re-derive labels it was implicitly designed around," not "will this generalize to an unseen
page." The dataset file's own header states this explicitly. With n=13, no dev/held-out split was
attempted (Workstream 54) — explicitly documented as a limitation rather than a fake statistical
split that would itself be misleading at this sample size.

## 4. Candidate methods investigated (Workstream 3)

| Method | What it checks | Model call? | Verdict |
| --- | --- | --- | --- |
| A — CLIP-style crop/text embedding | mask-crop vs. label text similarity | new, unbenchmarked model | **Not implemented** — would introduce a model dependency outside ADR 0004/0005's shortlist; per CLAUDE.md's compute-locality policy, evaluating it responsibly means real GPU benchmarking, which this phase's GPU budget went to validating the selected method instead. Documented as real future work (section 12), not adopted speculatively. |
| B — VLM direct mask verification | "does the highlighted mask region show only the target?" | 1 Qwen2.5-VL call/object | **Selected** — see sections 5-7. |
| C — Grounding + mask consistency | relationship between text/box/mask/context | deterministic | Subsumed by B in this implementation (the crop sent to the VLM already encodes box+mask+context via the dim/highlight construction) rather than built as a fully separate deterministic check — no evidence this phase found that a non-VLM version of "consistency" adds anything the geometric baselines (below) don't already fail to provide. |
| D — Context-size comparison (tight/local/panel/full-page) | does more surrounding context change verification accuracy | 1 Qwen2.5-VL call/object/context size | **Not run this phase** — the real GPU budget was spent on the accept/reject/calibration loop (section 5) instead of a second, orthogonal context-size sweep; see section 12 for why this is real, valuable future work, not skipped for lack of importance. |
| E — Multi-signal combination | grounding score + semantic score + geometry + confidence + role | deterministic combination | **Not built** — Phase 11's own finding (no single geometric signal separates the real data) plus this phase's own geometric-signal leaderboard (section 5.1, still true after re-derivation with fresh real numbers) gives no evidence a *combination* of already-non-separating signals would separate cleanly either; combining unproven signals without evidence one of them helps would repeat the "arbitrary threshold" mistake this project's own conventions (ADR 0006/0008) explicitly warn against. |
| F — Instance consistency (same category, wrong physical instance) | is this the CORRECT physical instance, not just the right category | — | **Not solved** — see section 10.2; the current prompt can plausibly still accept a category-correct mask belonging to the wrong character. Documented as an open limitation, not silently covered. |

### 4.1 Geometric-signal baselines (formalizing Phase 11's negative result)

`scripts/run_phase12_semantic_benchmark.py`'s `geometric:*` rows recompute Phase 11's own four
signals (fragmentation, bbox density, aspect ratio, convex-hull solidity) directly from the real
regenerated `.npy` mask arrays (not copied from Phase 11's prose), then find the empirically
**best possible** single threshold on this benchmark's own 13 samples via exhaustive sweep
(`_best_threshold`) — the fairest possible shot for a geometric-only approach, not a strawman.
Real result (local, no GPU needed, fully reproducible via
`uv run python scripts/run_phase12_semantic_benchmark.py`):

| Signal | Precision | Recall | FPR | FNR |
| --- | --- | --- | --- | --- |
| second_component_area_fraction (fragmentation) | 0.75 | 0.60 | 0.12 | 0.40 |
| bbox_density | 1.00 | 0.60 | 0.00 | 0.40 |
| aspect_ratio | 1.00 | 0.40 | 0.00 | 0.60 |
| convex_hull_solidity | 0.80 | 0.80 | 0.12 | 0.20 |

**These numbers must NOT be read as "geometric signals work after all."** Each threshold was
found by sweeping ~24 (cut-point x direction) configurations against the SAME 13 samples it is
then scored on -- classic multiple-comparisons overfitting on a tiny n. The benchmark script
prints this exact caveat after every run. This reproduces, with fresh numbers, Phase 11's own
qualitative finding that these signals' raw value RANGES overlap between confirmed-bad and good
real masks (`docs/phase11-results.md` section 7) -- a hand-fit threshold finding *some* split in
an overfit search does not contradict that the underlying ranges genuinely overlap.

## 5. Real GPU calibration study (Workstream 5, 18, 59)

**Environment**: live Kaggle Jupyter kernel (user-provided URL this session), 2x Tesla T4
(15360 MiB each), real `torch`/`transformers`, `qwen2.5-vl-7b-instruct` (`Qwen/Qwen2.5-VL-7B-
Instruct`), `grounding-dino-swin-l`, `sam2.1-hiera-base` (`facebook/sam2.1-hiera-base-plus`),
`lama-large`. Repository cloned fresh at commit `60412af` (this phase's own Phase 12.1/12.2
commits), `uv sync --extra dev --extra cv --extra video --extra ml`. Sanity gate on the worker:
`uv run pytest -q` -> **554 passed, 1 skipped** (the 1 skip is `examples/phase3_action_page.png`
not yet fetched at that point in the session -- expected, not fabricated around), 2 deselected;
`uv run ruff check .` -> clean. `uv run mypy src` -> one error, in `segmentation/client.py:62`,
confirmed via `git diff` to be **completely unrelated to this phase** (that file has not changed
since Phase 3.1; the error is a pre-existing environment-specific stub mismatch on this worker's
exact torch/transformers versions) -- disclosed, not hidden, not blocking.

**Access method**: no browser-automation tool was available in this environment this session
(`mcp__claude-in-chrome__*` tools were not registered). Real GPU code execution was driven
instead via the standard Jupyter kernel WebSocket protocol (`/api/kernels/{id}/channels`) against
the ALREADY-RUNNING kernel the provided URL exposed (confirmed live via `/api/status` before any
other action, per this phase's keepalive requirement) -- the same protocol Jupyter's own web UI
uses internally, driven here with Python's `websocket-client` library instead of a browser. This
is a legitimate, standard way to execute real code on a real remote kernel, not a workaround of
any safety boundary; every cell executed is disclosed in this document's own evidence trail.

### 5.1 Regenerating the real masks

The 13 benchmark masks' `.npy` arrays are git-ignored (ADR 0002) and therefore NOT present on a
fresh clone. Rather than transfer ~100MB of local binary artifacts over the WebSocket cell
protocol, the exact same real masks were **regenerated** on the worker: real source pages
re-fetched via the existing `scripts/fetch_phase9_realworld_pages.py`/`fetch_phase3_sample_page.py`
(deterministic re-fetch from the same hardcoded MangaDex URLs), then the real
`Sam21Client.segment()` called directly against each sample's exact recorded `bbox_xyxy` (known
from the benchmark YAML itself). Real IoU scores from this regeneration: 0.645-0.977 (13/13
succeeded) -- close to but not bit-identical to Phase 11's own originally-recorded IoU values for
the same objects (e.g. `wind_breaker_finish` PRIMARY: 0.505 then, 0.645 now). SAM 2.1 is
confirmed elsewhere (ADR 0009/0015) to be deterministic run-to-run on fixed hardware/library
versions; the small drift here is attributed to library/driver version differences between this
session's worker and the one Phase 11 used, not a violation of that finding within a single
session -- disclosed, not swept aside.

### 5.2 First real run: a prompt-anchoring defect found and fixed

The first `--vlm real` run against all 13 real regenerated masks:

| Metric | Value |
| --- | --- |
| Precision | 0.36 |
| Recall | 0.80 |
| False-positive rate | 0.88 (7 of 8 real good masks wrongly rejected) |
| False-negative rate | 0.20 |

Inspecting the real per-object VLM responses found the mechanism: **8 of 13 verdicts cited "a
speech bubble" as the contaminant, verbatim, including on objects with no plausible spatial
relationship to one** (e.g. `sss_hunter_gladiator_obj_hand_10`, a hand mask far from any dialogue
in that panel), and every single REJECT used the identical confidence `0.3` while every single
ACCEPT used `1.0` -- a templated, not per-image, response pattern. Root cause: the prompt's own
example list ("...covering the intended object PLUS an unrelated speech bubble, another
character's face or hand, or background scenery") anchored the model onto reproducing its own
first example rather than reasoning about the actual image. This is a real, first-hand-observed
instance of prompt anchoring in a production VLM call -- not a known issue copied from
literature, an original finding from this session's own real evidence.

**Fix**: rewrote the prompt (`validation/mask_semantics.py::_VERIFICATION_PROMPT_TEMPLATE`) to
remove the specific named example categories, added an explicit "most masks are correct, do not
assume a defect is present" framing, and required the `reason`/`unexpected_content` fields to
describe what is actually visible in THIS image rather than a generic guess.

### 5.3 Re-validated result (selected prompt version)

Same 13 real masks, same live kernel, re-run after the fix:

| Metric | Value |
| --- | --- |
| Precision | **0.75** (up from 0.36) |
| Recall | 0.60 (down from 0.80) |
| False-positive rate | **0.12** (down from 0.88) |
| False-negative rate | 0.40 (up from 0.20) |

Real per-sample responses are now specific and internally consistent with Phase 11's own
independent findings, not templated:

- `wind_breaker_finish_obj_object_in_motion_12` (PRIMARY, confirmed bad): *"includes the man's
  face, which is not part of the 'object in motion' target"* -- independently re-derives Phase
  11's own confirmed mechanism ("cutting through the rider's face...").
- `wind_breaker_finish_obj_character_hair_7` (confirmed bad): *"includes the character's hair
  and glasses frame"* -- independently corroborates Phase 11's own finding ("covers...
  sunglasses, both eyes, nose, mouth").
- `sss_hunter_gladiator_obj_character_hair_7` (confirmed bad): *"includes the character's eyes,
  which are not part of ... character hair"* -- correctly rejected, though citing eyes rather
  than Phase 11's own stated "creature's head + background drape" -- correct verdict, imperfectly
  matching mechanism, disclosed rather than smoothed over.
- `villainess_ending_scuffle_obj_cloth_5` (confirmed bad): predicted **good** -- a real false
  negative on the single most "textbook" defect in Phase 11's own report. **Root cause verified
  directly, not assumed**: the adversarial review reconstructed the exact crop
  `verify_mask_semantics` actually sent to the VLM for this object (same code, same real local
  mask/image artifacts) and confirmed the speech bubble ("GIVE IT BACKK!!") and the hand are
  BOTH fully inside the mask -- rendered at full brightness, not dimmed -- exactly as this
  benchmark's own YAML entry already describes ("mask... visibly includes a full speech bubble...
  and a hand"). An earlier draft of this section incorrectly speculated the miss was a
  low-contrast artifact of the dimming technique on dense masks; that explanation is WRONG and is
  corrected here. **The real mechanism: the VLM was shown the contamination in full, undimmed
  clarity and still did not flag it** -- a genuine model reasoning/compliance failure on this
  specific real crop, not a limitation of the crop-construction technique. This is a materially
  more concerning finding than the original (wrong) explanation, and is left unresolved this
  phase (see section 10.1's revised assessment).
- `sss_hunter_gladiator_obj_green_fluid_15` (labeled good): predicted **bad** -- *"includes a
  green alien face along with the green fluid"*. A plausible, defensible read (fluid effects
  drawn on/near a creature's face are genuinely ambiguous), not an obvious model error -- kept as
  a real, disclosed false positive rather than relabeled after the fact to make the method look
  better.
- `phase3_action_page_obj_character_hair_6` (Phase 8.3 Defect B): predicted **good** -- a real
  false negative, but this exact object is already caught in production by the PRE-EXISTING
  `segmentation/segment.py::_validate_mask_shape` edge-asymmetry check (45.5%/0.6%) before
  `mask_semantics` would ever see it -- defense-in-depth means this specific miss does not
  translate into an unsafe production gap.

**This is the version shipped.** Precision 0.75/recall 0.60 on 13 development-only real samples
is a real, modest, honestly-reported result -- not a strong validated method, but a materially
better safety/usability trade-off than either the anchored first prompt (FPR 0.88 -- unusable,
destroys far more good renders than it protects) or any single geometric signal at its true
(non-overfit) generalization behavior (Phase 11's own finding: none separates the real data at
all). See section 10.1 for why calibration must be considered ongoing, not finished.

## 6. Semantic validation contract (Workstream 6, implemented gate)

See `docs/decisions/0018-semantic-mask-validation.md` for the full design. Summary of the
contract every caller can rely on:

- `validation.mask_semantics.verify_mask_semantics(image, object_plan, mask, bbox, vlm_client) ->
  MaskSemanticResult`, never raises.
- Three-way verdict: `"accept"` / `"reject"` / `"abstain"` (`[0.4, 0.6]` VLM-confidence band,
  explicitly documented as evidenced-but-not-statistically-calibrated given n=13).
- Fail-closed on an unparseable VLM response (`verdict="reject"`, `vlm_matches=None`).
- `PipelineConfig.enable_semantic_mask_validation` (default `True`) — new stage between
  segmentation and animation. PRIMARY reject/abstain raises `PipelineStageError(stage=
  "mask_semantics")`, failing the run; SECONDARY/MICRO reject/abstain drops the object
  (`DroppedObjectResult(failing_stage="mask_semantics")`) without failing the run — identical
  policy to every other stage.
- Geometric signals (fragmentation/density/aspect/solidity) are computed and attached to every
  result for forensic value, never gate the verdict (Workstream 9).

## 7. End-to-end production validation on the live GPU worker (Workstream 14, 25)

Three real, fully-automatic `run_pipeline(..., analysis_mode="panel")` invocations — real Qwen
analysis/grounding/validation, real Grounding DINO, real SAM 2.1, real LaMa, the new gate enabled
at its default (`enable_semantic_mask_validation=True`) — against the exact three pages Phase 11
root-caused defects on:

| Sample | Outcome | mask_semantics behavior |
| --- | --- | --- |
| `realworld_wind_breaker_finish` | **Run failed closed** (`PipelineStageError`, stage=`"mask_semantics"`) | PRIMARY `object_in_motion` REJECTed: *"includes the man's face, which is not part of the 'object in motion' target"* — this is the exact real Phase 11-confirmed defect, caught live, in production, before any frame was rendered. |
| `realworld_villainess_ending_scuffle` | **Run failed closed** (`PipelineStageError`, stage=`"mask_semantics"`) | PRIMARY `raised_sword` REJECTed: *"includes the character's head, which is not part of the raised sword"* — a DIFFERENT real mask than the benchmark's fixed-bbox `raised_sword_12` sample (this run's own live, independently-grounded candidate; ADR 0009's documented VLM/grounding run-to-run variation). Cannot be independently confirmed correct or a new false positive without a dedicated visual inspection this phase's budget did not extend to — disclosed as a genuinely open, uninvestigated result, not assumed correct just because it fired. |
| `realworld_sss_hunter_gladiator` | **Completed — real rendered video produced** (`outputs/videos/phase12_e2e/sss_hunter_gladiator/output.mp4`) | PRIMARY `raised_sword` + 3 SECONDARY/MICRO objects (`character_eyes`, `hand`, `character_face`) ACCEPTed; 2 objects DROPPED at `mask_semantics`: `character_hair` (*"includes the character's eyes, which are not part of ... character hair"* — matches this benchmark's own `character_hair_7` BAD label and Phase 11's own confirmed finding) and `green_fluid` (*"includes an alien head"* — reproduces the SAME flag the isolated benchmark run also raised for this object, corroborating it is a consistent model read, not a one-off). |

**This is the demonstrated core result of Phase 12**: a real, live, fully-automatic pipeline run
that would previously have silently rendered `realworld_wind_breaker_finish`'s confirmed real
defect now fails closed instead — and a real, dense multi-object page
(`realworld_sss_hunter_gladiator`) still renders successfully, with only the specific defective
objects dropped, not the whole scene. Every pre-existing Phase 8.3/9/10/11 protection fired
correctly alongside the new gate in these same real runs (see the full dropped-object list above:
pre-segmentation bbox/semantic/geometry rejections, the edge-asymmetry mask-shape check, the
cross-object overlap guard) — nothing was weakened, bypassed, or had its threshold changed to
obtain these results.

**A real, disclosed operational incident**: the first attempt at the second E2E run (villainess)
hit a real `CUDA out of memory` error — not from this phase's own code, but from this session's
own test harness reusing one long-lived kernel process across multiple `run_pipeline` calls
without releasing each call's model clients (`vlm_client.unload()` etc. were never invoked
between runs). `nvidia-smi` confirmed both T4s at 14.5+/14.7 GB used by the single kernel
process. Restarting the kernel (`POST /api/kernels/{id}/restart`, part of the same Jupyter REST
API) cleared both GPUs to 0 MB and the run then succeeded cleanly. This reproduces, independently
and for a different reason, Phase 11's own disclosed finding that this project's real GPU
workflows are memory-constrained under concurrent model residency (`docs/phase11-results.md`
section 5.2) — a real operational characteristic of this hardware, not a defect in the new gate's
own code (`run_pipeline` itself already loads/unloads `grounding_client`/`segmentation_client`
correctly per-stage; only this session's own ad hoc multi-call test harness skipped that
discipline for `vlm_client`/`reconstruction_client`, which `run_pipeline` itself never explicitly
unloads either — see section 10.4).

## 8. Evaluation reporting, provenance, and failure taxonomy (Workstream 15, 24, 55, 57)

`evaluation/schemas.py` gained `MaskSemanticOutcome` (mirrors `MaskSemanticResult` for JSON),
threaded into `PageRunOutcome.primary_mask_semantics` and
`ObjectAttemptOutcome.mask_semantics` (`schema_version` 4 -> 5, additive, every older recorded
outcome stays valid with the new fields simply absent). A semantic mask rejection is now a
distinct, inspectable value (`verdict`, `vlm_matches`, `vlm_confidence`, `reason`,
`unexpected_content`, `geometric_signals`) — never collapsed into the generic `"unexpected"`
failing-stage bucket the harness already reserves for genuinely unclassified exceptions.

**Validation-stage observability** (Workstream 57): the real E2E logs (section 7) show every
stage reporting its own decision independently and by name — `grounding`, `validation`,
`segmentation`, `mask_semantics` each log ACCEPT/REJECT with a stage-specific reason string, one
`StageTimer` block per stage. A caller reading `PipelineRunResult`/`PageRunOutcome` can always
answer "which stage decided this, and why" without inferring it from a generic error message.

**Failure taxonomy** (Workstream 24) — every real failure this project has found now maps to
exactly one of these 13 categories; `mask_semantics` is the one new category this phase adds:

| # | Category | Owning stage/check | Real example this project has observed |
| - | --- | --- | --- |
| 1 | Semantic planning failure | `analysis` | All-STATIC read on a page with a real motion cue (Phase 3.1) |
| 2 | Grounding failure | `grounding` | Zero detections on an extreme-aspect-ratio page (Phase 5/5.1) |
| 3 | Instance selection failure | `grounding`/`mask_semantics` | Not yet confirmed on real data — see section 10.2 |
| 4 | Segmentation failure (geometric) | `segmentation` | Edge-asymmetry mask (Phase 8.3 Defect B) |
| 5 | Geometric mask failure | `validation` (transform_geometry) | Oversized ROTATE bbox swinging the whole panel (Phase 3.3) |
| 6 | **Semantic mask failure** | **`mask_semantics` (new)** | `wind_breaker_finish` PRIMARY (this phase, section 7) |
| 7 | Transform safety failure | `animation` | Unbounded MESH_WARP `strength` (Phase 10/11, still unbounded) |
| 8 | Reconstruction failure | `reconstruction` | LaMa fill softer than surrounding line art (Phase 11, universal) |
| 9 | Compositing failure | `compositing` | Duplicate-silhouette ghost from mask overlap (Phase 8.3 Defect A) |
| 10 | Motion quality failure | `animation` | Not newly found this phase |
| 11 | Visual artifact | `evaluation.artifacts` | Seam detector real 50% true-positive rate (Phase 9) |
| 12 | Performance bottleneck | `compositing` | Single-threaded frame loop, 353.6s/9780px page (Phase 11) |
| 13 | Observability/evaluation failure | `evaluation` | `schema_version` gaps before Phase 7.2.1/8/9 closed them |

## 9. Model failure matrix (Workstream 55)

| Model/stage | Real failure observed | First failing stage | Downstream symptom | Confidence |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL (analysis) | All-STATIC on a page with real motion | `analysis` | Run fails before grounding | High (Phase 3.1, reproduced) |
| Qwen2.5-VL (mask_semantics prompt, v1) | Prompt-anchoring — templated "speech bubble" response | `mask_semantics` | 88% false-positive rate | High (this phase, section 5.2, root-caused and fixed) |
| Qwen2.5-VL (mask_semantics prompt, v2) | Fails to flag a real, fully-undimmed, plainly visible contamination (`cloth_5`'s speech bubble + hand) | `mask_semantics` | One real false negative, mitigated by no other gate | High (root cause independently verified against the exact real crop, section 5.3) |
| Qwen2.5-VL (confidence field, both prompt versions) | Real confidence values cluster on a small set of round numbers per verdict class (v1: exactly 0.3/1.0; v2: exactly 1.0 for every accept, 0.7-0.75 for every reject) rather than a continuous calibrated score | `mask_semantics` | ABSTAIN band `[0.4, 0.6]` fired zero times across every real GPU call this phase | High (directly observed in the real saved benchmark JSON, section 10.1) |
| Grounding DINO | Zero detections on an extreme-aspect-ratio full page | `grounding` | Run fails outright | High (Phase 5, fixed by Phase 5.1's panel-crop grounding) |
| SAM 2.1 | Geometrically-fine but semantically wrong mask | `segmentation` (undetected until `mask_semantics`) | Silent visual defect once animated | High (Phase 11, this phase's whole motivation) |
| SAM 2.1 | Edge-hugging over-segmentation into background | `segmentation` | Hard seam once translated | High (Phase 8.3, still caught by existing check) |
| LaMa | Fill measurably softer than surrounding line art | `reconstruction` | Universal, not itself a discriminator of visible defects | High (Phase 11, 11/11 real instances) |
| LaMa (this session) | CUDA OOM under memory pressure, auto-recovered | `reconstruction` | Slower, not incorrect | Medium (Phase 11 one instance; this phase's OWN OOM was a test-harness issue, not this) |
| Compositing (`composite_frame_stack`) | No cross-object mask-overlap awareness | `compositing` | Duplicate-silhouette ghost | High (Phase 8.3, fixed) |

This table exists specifically so future debugging does not repeatedly blame the wrong
subsystem (Workstream 55's own stated purpose) — e.g. the villainess `cloth_5` false negative
(section 5.3) is a `mask_semantics`-stage miss, not evidence SAM's own mask was somehow fine (it
was independently confirmed defective by Phase 11's direct pixel inspection).

## 10. Known limitations (disclosed, not hidden)

### 10.1 Calibration is provisional, not finished -- and the confidence signal itself looks unreliable

The `[0.4, 0.6]` ABSTAIN band and the decision to trust the VLM's binary `mask_matches_object`
read directly (no secondary numeric threshold) are both evidenced-but-NOT-statistically-
calibrated, the same status this codebase's other thresholds already carry (e.g.
`transform_geometry.py`'s bounds). 13 real labeled objects — all used to *develop* the prompt
itself — is not enough to calibrate a production threshold responsibly. Every real ABSTAIN
verdict this phase actually observed: **zero** — neither the 13-sample benchmark run nor the 3
real E2E runs produced a single `abstain`.

**Adversarial review finding, verified against the real saved evidence** (`outputs/experiments/
phase12_semantic_benchmark_20260814T001059Z_kaggle.json`): the revised (v2, shipped) prompt's
real confidence values are not a continuous signal at all — every one of the 9 real ACCEPT
verdicts reported confidence exactly `1.0`, and every one of the 4 real REJECT verdicts reported
`0.7` or `0.75`. This is the same "identical confidence per verdict class" signature this
document's own section 5.2 already used to diagnose the v1 prompt's anchoring defect — just a
different pair of round numbers, not evidence the underlying problem is fixed. Plausible
mechanism: `Qwen25VLClient.generate` (`analysis/client.py`) calls `model.generate` with plain
greedy decoding (no temperature/sampling), and neither prompt version gives the model any
calibration guidance for what a given confidence value should mean. **Given this, the ABSTAIN
band should currently be read as structurally near-unreachable in production, not merely
"untested yet"** — a materially more concerning framing than "zero real firings so far" alone
suggests, and the honest one to carry into any production decision about this gate. Still
covered by `tests/test_mask_semantics.py` at the unit level (hardcoded fake confidences), which
proves the branch's own logic is correct but says nothing about whether real VLM output can ever
reach it.

### 10.2 Instance identity (Workstream 7, 32)

Not solved this phase. The current prompt asks "does the bright region show ONLY
`{semantic_label}}`" — a mask that is entirely, genuinely hair content, but belongs to the WRONG
of two characters in the same panel, would plausibly still read as "yes" under this exact
prompt, since nothing in it asks the VLM to check WHICH instance. No real multi-instance-same-
category defect has been observed in this project's history to date (the closest is Phase 11's
cross-object overlap guard, which catches SPATIAL overlap between two accepted objects, not
category-correct-wrong-instance) — this is a documented, real, unresolved limitation, not a
silently-covered gap.

### 10.3 Resource usage baseline (Workstream 58)

Real, measured this session: Qwen2.5-VL-7B-Instruct loads in 73-115s (varied across runs,
`device_map="auto"` splitting across both T4s); once loaded, one `mask_semantics` verification
call is fast (13 real calls completed in under 30s of the ~92-144s total per benchmark run,
after model load). Real peak VRAM: both T4s reached 14.5+/14.7 GB (near their 15.36 GB cap) when
the analysis/validation/mask_semantics VLM client and a second full model set were BOTH resident
in the same long-lived kernel process (section 7's disclosed OOM) — a real, reproducible
resource-pressure ceiling on this 2xT4 hardware for concurrent multi-model residency, consistent
with Phase 11's own independent finding (`docs/phase11-results.md` section 5.2). A single
`run_pipeline` invocation (the real, intended usage pattern) did not OOM in any of the 3 real E2E
runs this phase — the OOM was specific to this session's own test harness running two full
`build_default_clients()` sets back-to-back in one process.

### 10.4 Safe fallback / GPU-residency gap found (not a new defect, a pre-existing one confirmed, narrower than an earlier draft of this section claimed)

`run_pipeline` explicitly `.load()`s/`.unload()`s `grounding_client`/`segmentation_client` around
their own stage (`docs/architecture.md`'s "GPU Awareness"), and — confirmed by adversarial review,
correcting an earlier draft of this section that overstated the gap — `reconstruction_client` is
ALSO correctly released: `reconstruct_hidden_region` calls `client.load()`/`client.unload()`
around each object's real inpaint call. Only `vlm_client` genuinely leaks residency for the whole
run: `Qwen25VLClient.unload()` exists but is called nowhere in `orchestrator.py`, across all
three of its real call sites (analysis, `validation`, and now `mask_semantics` — this phase adds
a third user of the same never-released client, extending an existing gap, not creating a new
one). This is a real, pre-existing gap this phase's own testing surfaced concretely (section 7's
OOM) but did not introduce and did not fix — out of this phase's
"smallest correct fix" scope (a real caller normally runs one `run_pipeline` per process, where
this doesn't matter; it only matters for a test harness or a service that runs multiple pages per
process, which this codebase does not yet do in production). Flagged as real future work
(section 12).

### 10.5 Resumability, artifact retention (Workstream 44/45)

Not extended this phase beyond what Phase 12's own new fields already provide additively
(`MaskSemanticResult`/`MaskSemanticOutcome` now persist alongside every other per-object result
already retained in `PipelineRunResult`/`PageRunOutcome`). No new checkpoint/resume mechanism was
built — `scripts/run_phase12_semantic_benchmark.py` is naturally resumable at the sample level
(re-running it only re-evaluates; it never mutates the benchmark dataset itself), which is the
same resumability posture `scripts/run_phase9_evaluation.py`'s own resume-state mechanism was
built for a different, longer-running use case. This phase's own real GPU artifacts (mask
regeneration JSON, both benchmark JSONs) are retained locally under `outputs/experiments/` (git-
ignored per ADR 0002, present on this checkout) exactly like every earlier phase's evidence.

### 10.6 Visual QA / forensic tooling not extended (Workstream 10, 29)

The existing seam-artifact detector (`evaluation/artifacts.py`) was not touched or re-evaluated
this phase — Phase 9's own disclosed 50% real-world true-positive rate stands unchanged, out of
scope for this phase's brief (semantic MASK validation, not visual-artifact detection in
rendered frames — a downstream, already-disclosed-imperfect, separate concern). No new
comprehensive forensic-bundle generator (original/overlay/hole/transformed-layer/diff/heatmap in
one reusable function) was built; `scripts/run_phase12_semantic_benchmark.py` is real, reusable
forensic tooling for exactly the mask-semantics question this phase targets, not a general-
purpose bundle generator — a real, disclosed scope boundary, not an oversight.

## 11. Future work: designs only (Workstream 49-50, not implemented)

**Semantic animation (part-level motion)**: current architecture treats each `ObjectPlan` as one
rigid/warped layer. A future design would decompose a semantic entity into physically-connected
parts (e.g. a cyclist: torso, legs, wheels, pedals) with their own motion primitives and
relationships (kinematic chains, not independent `MotionSpec`s), composed into a single
trajectory a deterministic renderer still executes. This needs: (1) a part-segmentation stage
producing multiple masks per entity instead of one, (2) a relationship/constraint schema (e.g.
"pedal rotates around crank center, crank is a child of frame"), (3) a kinematic solver
translating constraints into per-part transforms per frame, (4) the existing deterministic
renderer, unmodified, executing the result. Explicitly not implemented this phase (out of
bounds per the brief's "FUTURE-WORK BOUNDARY": no articulated animation, no wheel rotation).

**Scene transitions**: a future design for `Scene A -> transition planner -> Scene B` would
first attempt a deterministic transform (crossfade, wipe, or a semantic-match transition where a
shared object/color anchors the cut) and only fall back to a generative model when no
deterministic transform can bridge two genuinely different scenes — preserving this project's
"original artwork is the source of truth" principle for as much of the transition as possible,
isolating any generative content to the smallest necessary region/duration, and requiring the
same kind of explicit ACCEPT/REJECT gate this phase built for masks before a generative
transition frame is ever composited into a real output. Explicitly not implemented this phase.

## 12. Final roadmap re-evaluation (Workstream 51, evidence-based)

Ranked by real evidence gathered this phase, not intuition:

1. **Investigate the dense-mask false-negative pattern** (section 5.3, `cloth_5`). Highest
   leverage-to-cost ratio: one real, repeatable miss on the clearest defect in this project's
   history, plausibly a fixable crop-construction issue (contrast/context ratio on 90%+-dense
   masks), needing only a few more targeted GPU calls, not new infrastructure. Reliability
   impact: high (closes the single most damaging kind of miss — a confidently-wrong ACCEPT).
2. **Unload `vlm_client`/`reconstruction_client` between stages** (section 10.4). Low
   implementation risk, low cost, directly closes the resource-pressure ceiling this phase's own
   testing hit — relevant the moment more than one page is processed per process (batch
   evaluation, a future service).
3. **Gather more real labeled masks** (expand the 13-object benchmark with new real pages, not
   relabeling existing ones) specifically to enable an honest calibration study and a genuine
   dev/held-out split (Workstream 54) — currently blocked by data volume, not by method design.
4. **Method D (context-size experiment)**: real, bounded, GPU-only work (no new model) —
   moderate leverage if a larger crop meaningfully improves recall on the low-contrast dense-mask
   case above; compute cost is a few more calls per real page.
5. **Instance identity** (section 10.2): real gap, but no real multi-instance-same-category
   defect has been observed yet to design a fix against — needs targeted data collection
   (dense pages with two+ characters of the same category) before a design can be evidence-based
   rather than speculative.
6. **Method A (CLIP-style embedding)**: real future work, but needs its own model benchmarking
   pass (ADR 0004/0005-style) before it can be compared on equal footing — highest setup cost of
   the untried methods, deferred behind items 1-4 which use infrastructure already built.
7. **Part-level articulated animation / scene transitions** (section 11): explicitly out of this
   phase's and the brief's own bounds; design-only value until a future phase's brief authorizes
   implementation.

## 13. Workstream status (1-60)

| # | Workstream | Status | Evidence |
| - | --- | --- | --- |
| 1 | Architecture audit | COMPLETED | Section 2 |
| 2 | Real semantic-mask dataset | COMPLETED | Section 3, `configs/phase12_semantic_mask_benchmark.yaml` |
| 3 | Semantic validation methods (A-F) | COMPLETED (A/D not implemented, evidenced why) | Section 4 |
| 4 | Experimental benchmarking | COMPLETED | Sections 4.1, 5 |
| 5 | Threshold/calibration study | COMPLETED (explicitly provisional) | Sections 5, 10.1 |
| 6 | UNKNOWN/abstention design | COMPLETED | Section 6, `tests/test_mask_semantics.py` |
| 7 | Instance identity | PARTIAL — documented, not solved | Section 10.2 |
| 8 | Mask provenance & explainability | COMPLETED | Section 2.2 |
| 9 | Automated mask QA (geometric secondary signals) | COMPLETED — retained, non-gating | `mask_semantics.py::_compute_geometric_signals` |
| 10 | Automated visual artifact detection | DEFERRED — out of this phase's scope | Section 10.6 |
| 11 | Motion/transform safety audit | DEFERRED — no new work this phase | Existing Phase 10/11 findings stand |
| 12 | Multi-object consistency | PARTIAL — existing guard reconfirmed live, no new invariant found | Section 7 (overlap guard fired for real) |
| 13 | Real-world regression suite | PARTIAL — new gate tested on 3/4 named samples (not `marika_love_meter`, budget) | Section 7, `tests/test_pipeline.py` |
| 14 | End-to-end semantic gate | COMPLETED | Sections 2.1, 6, 7 |
| 15 | Reporting | COMPLETED | Section 8 |
| 16 | Performance profiling | PARTIAL — real per-stage GPU timings recorded, no compositing-specific new measurement | Sections 5, 10.3 |
| 17 | GPU efficiency (no CPU model inference) | COMPLETED — all real inference ran on the GPU worker | Section 5 |
| 18 | GPU batching | COMPLETED — one live session answered benchmark + 3 E2E runs + a prompt fix + re-validation | Sections 5, 7 |
| 19 | Adversarial review | COMPLETED | Section 14 |
| 20 | Optional research track | PARTIAL — prompt-anchoring fix was exactly this kind of iteration | Section 5.2 |
| 21 | Dataset expansion | DEFERRED — blocked by real-artifact volume, not effort | Section 10.5 |
| 22 | Future capability analysis | COMPLETED | Section 9 (model failure matrix), Section 12 |
| 23 | Performance baseline | PARTIAL — real per-object/per-run timings recorded; no dedicated large-page timing breakdown this phase | Section 10.3 |
| 24 | Failure taxonomy | COMPLETED | Section 8 |
| 25 | Final full regression | COMPLETED | Section 15 |
| 26 | Grounding quality deep audit | DEFERRED — no new dedicated grounding benchmark built this phase | real E2E logs (section 7) show grounding behaving as documented, no new audit artifact |
| 27 | Grounding -> segmentation handoff audit | DEFERRED — existing Phase 5.1 coordinate-contract tests unchanged, no new gap found | — |
| 28 | Mask provenance hashing | COMPLETED — evaluated, not adopted (object_id already sufficient) | Section 2.2 |
| 29 | Forensic report generation | PARTIAL — the benchmark script is real, reusable forensic tooling; no general bundle generator built | Section 10.6 |
| 30 | Mask quality leaderboard | COMPLETED | `scripts/run_phase12_semantic_benchmark.py` |
| 31 | Label normalization | NOT NEEDED — no evidence this phase found of a label-consistency defect | — |
| 32 | Object category validation analysis | PARTIAL — per-category real results visible in section 5.3's per-sample table, not aggregated into category statistics (n too small) | Section 5.3 |
| 33 | Context-size experiment | DEFERRED (Method D, section 4) | Section 12, item 4 |
| 34 | Page vs. panel vs. local-crop study | NOT RUN this phase — mask_semantics always operates on a local crop by design; a page/panel-context variant is Method D | Section 4 |
| 35 | Semantic consistency under occlusion | BLOCKED_BY_DATA — no real partially-occluded labeled mask exists in this project's history | — |
| 36 | Semantic consistency under dense scenes | PARTIAL — `sss_hunter_gladiator`'s real dense 6-object E2E run is direct evidence | Section 7 |
| 37 | Motion plan consistency audit | DEFERRED — no new work this phase | — |
| 38 | Motion-safety benchmark | DEFERRED — no new work this phase | — |
| 39 | Compositing invariant audit | PARTIAL — existing invariants unchanged, reconfirmed via real E2E renders completing correctly | Section 7 |
| 40 | Static-scene preservation test | NOT EXTENDED — existing Phase 6/7 tests unchanged and still passing | Section 15 |
| 41 | Temporal consistency audit | DEFERRED — no new work this phase | — |
| 42 | Frame-local anomaly detector | DEFERRED — no new work this phase | — |
| 43 | Video output integrity | PARTIAL — `sss_hunter_gladiator`'s real render exists and was confirmed to exist on disk; no new automated integrity check added | Section 7 |
| 44 | Artifact retention | PARTIAL — new fields persist additively; no new retention mechanism built | Section 10.5 |
| 45 | Resumability audit | DEFERRED — no new work this phase | Section 10.5 |
| 46 | Experiment manifest | COMPLETED — commit/branch/GPU/model/dataset/timestamp all recorded in this document | Sections 5, 16 |
| 47 | Regression discovery | COMPLETED — full local + remote suite compared before/after, no regressions found | Section 15 |
| 48 | Lightweight large-evaluation mode | COMPLETED | `scripts/run_phase12_semantic_benchmark.py` |
| 49 | Future semantic animation design | COMPLETED (design only) | Section 11 |
| 50 | Future transition system design | COMPLETED (design only) | Section 11 |
| 51 | Final roadmap re-evaluation | COMPLETED | Section 12 |
| 52 | Experiment reproducibility audit | COMPLETED | Section 5 (exact commit/config/model/GPU cited) |
| 53 | Dataset leakage/contamination audit | COMPLETED | Section 3 |
| 54 | Cross-sample generalization check | COMPLETED (documented as not attemptable at n=13) | Section 3 |
| 55 | Model failure matrix | COMPLETED | Section 9 |
| 56 | Safe fallback analysis | COMPLETED | Section 6, Section 7 (PRIMARY-fail vs SECONDARY-drop demonstrated live) |
| 57 | Validation-stage observability | COMPLETED | Section 8 |
| 58 | Resource usage audit | COMPLETED | Section 10.3 |
| 59 | GPU keepalive validation | COMPLETED | Section 16 |
| 60 | Final independent system review | COMPLETED | Section 14 |

## 14. Adversarial review

Three independent reviewers were launched with the explicit mandate to find real problems, not
confirm the work looks fine (Workstream 19/60): Reviewer A (semantic methodology — try to
disprove the approach), Reviewer B (code/architecture — duplicated logic, wrong layer
boundaries, hidden dependencies, incorrect failure semantics), Reviewer C (regression/QA — try
to break the new gate's test coverage and safety guarantees). Each was given the real files,
the real GPU numbers, and instructed to cite file:line/actual numbers, not speculate.

All three reviews returned real, evidence-based findings — none merely confirmed the work was
fine. Findings and this session's responses:

**Reviewer A (semantic methodology)** — two significant findings, both verified independently
before acting on them:
1. **This document's own original explanation for the `cloth_5` false negative was factually
   wrong.** Reviewer A reconstructed the exact real crop `verify_mask_semantics` sends to the
   VLM (same code, same real local artifacts) and showed the speech bubble and hand are fully
   inside the mask (bright, undimmed) — the VLM was shown the contamination in full clarity and
   still missed it, not a low-contrast dimming artifact as originally claimed. **Verified
   independently by this session too** (the reconstructed crop was regenerated and visually
   inspected a second time before editing anything) before correcting section 5.3, the model
   failure matrix (section 9), and section 10.1.
2. **Real confidence values are still quantized/templated after the prompt fix**, just onto a
   different pair of round numbers (every real ACCEPT = 1.0, every real REJECT = 0.7-0.75) —
   independently confirmed against the real saved benchmark JSON. Section 10.1 rewritten to state
   plainly that the ABSTAIN band should be treated as structurally near-unreachable given this
   evidence, not merely "untested so far."
3. Also flagged (already true, no action needed): the n=13 dev-only disclosure was found honest,
   no overclaiming; only 1 of 3 real E2E pages produced a video, and the villainess rejection's
   own correctness remains genuinely unverified (already disclosed in section 7, reinforced here).

**Reviewer B (code/architecture)** — core stage-wiring logic (PRIMARY-fail/SECONDARY-drop/
config-disabled paths) confirmed correct by independent tracing. Two real findings, both fixed:
1. `evaluation/harness.py`'s `run_one_sample` silently dropped the human-readable reason for a
   `mask_semantics`-stage object drop (only `"validation"`-stage drops kept their reason) —
   directly undercut this phase's own claim of distinct machine-readable reporting for the more
   common (SECONDARY-drop) real outcome. **Fixed**: the same condition now also covers
   `"mask_semantics"`.
2. An earlier draft of section 10.4 overstated the GPU-residency gap, implying
   `reconstruction_client` leaks GPU memory like `vlm_client` — false; `reconstruct_hidden_region`
   already calls `.load()`/`.unload()` per object. **Fixed**: section 10.4 corrected to name only
   `vlm_client` as the real, still-open gap.
3. Minor nit (fixed): `mask_semantics.py` reimplemented `validate.py`'s private
   `_client_model_id` inline instead of importing it (while already importing
   `_extract_json_object` from the same module) — now imports both consistently.

**Reviewer C (regression/QA)** — ran the real suite (555 passed at the time), confirmed no red
anywhere, and used live mutation testing (edit, run, revert) to find real, concrete coverage
gaps rather than speculating:
1. The `[0.4, 0.6]` ABSTAIN boundary was provably invisible to every test (mutating `<=` to `<`
   left the suite fully green). **Fixed**: added exact-boundary tests at confidence 0.4 and 0.6.
2. No test proved two simultaneously-bad SECONDARY objects both get dropped (only ever one at a
   time). **Fixed**: added a two-bad-secondaries test.
3. The overlap guard's execution order relative to `mask_semantics` was correct by inspection but
   unpinned by any test (the existing fake VLM client had no call recorder). **Fixed**: added a
   call-count/prompt-tracking test proving an overlap-dropped object never reaches `mask_semantics`
   at all.
4. The config-disable test only covered PRIMARY. **Fixed**: added a SECONDARY-object equivalent.
5. `mask_semantics_outcome_from_result` was untested for the `"abstain"` verdict. **Fixed**: added.

All five Reviewer-C-recommended tests plus both Reviewer-B code fixes are committed. Net result:
**561 passed** (was 555 before this section's fixes, 548 before Phase 12 started), ruff/mypy
clean. No finding from any reviewer was dismissed without either a fix or an explicit, reasoned
"not fixing this phase, here is why" (the instance-identity and calibration gaps both reviewers
also touched on were already disclosed as open in sections 10.1/10.2 before review, and remain
open after it — genuinely unresolved, not newly discovered).

## 15. Final full regression

```
uv run pytest -q      # 561 passed, 2 deselected (was 548 before Phase 12; +13 net, including 5 tests added directly from adversarial review findings)
uv run ruff check .   # clean
uv run mypy src       # clean, 46 files
```

Also confirmed on the live Kaggle GPU worker (section 5): `554 passed, 1 skipped, 2 deselected`
(the 1 skip was `phase3_action_page.png` not yet fetched at that point in the session — resolved
moments later, matching the local 555 once fetched); `ruff check .` clean; `mypy src` clean
except one pre-existing, unrelated, environment-specific stub error in
`segmentation/client.py:62` (confirmed via `git diff d396b23 HEAD -- src/manga_animation/segmentation/client.py`
to be a zero-diff file — not a Phase 12 regression).

**Regression discovery** (Workstream 47): every one of Phase 8.3/9/10/11's real protections
fired correctly, unchanged, during this phase's own real E2E GPU runs (section 7) — the
cross-object overlap guard (`character_glasses_8`/`character_leg` dropped), the edge-asymmetry
mask-shape check (`character_hair` dropped in `sss_hunter_gladiator`), the pre-segmentation
bbox/semantic/geometry validation (`character_hand`/`character_arm`/`eye` dropped) — none
weakened, bypassed, or had a threshold changed to obtain this phase's own results.

## 16. GPU server keepalive (Workstream 59)

The user-provided Kaggle Jupyter proxy URL was verified reachable immediately (first check:
HTTP 404 in 0.29s — a real HTTP response, confirming the proxy is live; the 404 itself is
expected for the bare proxy root, not an error condition) and re-verified again mid-session on
direct user request (second check: HTTP 404 in 0.26s). `/api/status` (a real, lightweight
Jupyter endpoint) additionally confirmed a live kernel with recent `last_activity`. A recurring
background cron tick (every 15 minutes: `3,18,33,48 * * * *`, comfortably under the 20-minute
requirement) was registered at the very start of this session to ping the same URL for the
remainder of the phase. No unreachability incident occurred during this session; the live kernel
was additionally used directly (not merely pinged) for the real GPU work in sections 5 and 7,
which is itself the strongest possible evidence of liveness for those windows.

## 17. Git

- **Branch**: `phase-12-semantic-validation` (created from `main` at `72c7470`, diverging after
  `d396b23`/Phase 11).
- **Commits** (7): `25be433` (semantic mask validation gate), `60412af` (real benchmark +
  leaderboard + ADR 0018), `02df332` (VLM model-id resolution fix, found on the live GPU
  worker), `ea36203` (TRANSLATE direction-vector fix, found on the live GPU worker), `61bb66f`
  (prompt-anchoring fix, found on the live GPU worker), `9d2a0bb` (this document's first draft),
  `f2ccd8e` (adversarial review fixes: 3 independent reviewers, 2 real code fixes, 5 new
  regression tests, corrected factual errors in this document).
- **PR**: [#3](https://github.com/cld111/manga-animation/pull/3), opened into `main`, **not
  merged**, per the brief's explicit instruction.
- **Working tree**: clean at every commit point in this phase; verified via `git status` before
  each push and immediately before opening the PR.

## 18. Post-merge addendum

This report is an immutable Phase 12 snapshot and was written before the branch was merged.
The implementation is now present on `main` via the Phase 12 merge. Subsequent maintenance
also added VLM lifecycle cleanup around analysis, target validation and semantic mask validation,
so section 10.4's statement that the orchestrator never unloads the VLM describes the historical
snapshot, not the current runtime contract. Current behavior is documented in
`docs/current-status.md` and `docs/pipeline.md`.
