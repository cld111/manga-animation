# manga-animation

Turn a single manga page into a short (~3-5s), seamlessly looping animation —
with the minimum amount of visually justified motion needed to express the
action already present in the artwork.

This is **not** an image-to-video generation system. The original manga
artwork is the source of truth: unrelated regions stay pixel-identical, and
only semantically justified objects (a flag, hair, a falling object, an
outstretched hand, cloth, an eye blink...) receive deterministic, kinematic
motion, composited back onto the original image.

## Project status

**Phase 1 — Engineering foundation — complete.** No ML models are integrated yet. This
phase established the project skeleton, configuration system, logging
foundation, the Animation Plan schema, `.claude/` agents and skills, and the
test suite Phase 2+ builds on. See [`docs/pipeline.md`](docs/pipeline.md)
for the planned pipeline and [`docs/decisions/`](docs/decisions) for why it's
structured this way.

**Phase 2 — Model benchmarking & selection — accepted.** The candidate
shortlist and benchmark methodology are written up in
[`docs/decisions/0004-phase2-model-candidates.md`](docs/decisions/0004-phase2-model-candidates.md),
with the machine-readable shortlist in
[`configs/benchmark_candidates.yaml`](configs/benchmark_candidates.yaml) and the
(model-agnostic, no-GPU-required) timing/reporting harness in
[`src/manga_animation/benchmarking`](src/manga_animation/benchmarking). Actual benchmark
runs — loading each candidate and measuring it — happen on the remote Kaggle/Jupyter GPU
worker per [ADR 0003](docs/decisions/0003-remote-compute-workers.md); no model weights are
downloaded or run locally. Reproducible adapter code for every shortlisted candidate lives
in [`scripts/phase2_kaggle_benchmark.py`](scripts/phase2_kaggle_benchmark.py); local,
non-GPU feasibility checks for the `deterministic-animation` and `video-rendering` stages
live in [`scripts/phase2_cv_feasibility.py`](scripts/phase2_cv_feasibility.py) and
[`scripts/phase2_video_feasibility.py`](scripts/phase2_video_feasibility.py). Per-stage
status (PRIMARY/FALLBACK/PENDING) is tracked in
[ADR 0005](docs/decisions/0005-phase2-model-selection.md).

**Phase 3.1 — First end-to-end vertical slice — real run completed, open gaps documented.**
Every stage (analysis, grounding, segmentation, reconstruction, animation, compositing,
rendering) is implemented for real in `src/manga_animation/<stage>` and wired together in
[`src/manga_animation/pipeline/orchestrator.py`](src/manga_animation/pipeline/orchestrator.py).
A real manga page was run through the complete pipeline on the remote Kaggle GPU worker,
producing a genuine, seamlessly-looping H.264 video — see
[`docs/phase3-results.md`](docs/phase3-results.md) for the full write-up, including two real
bugs found and fixed (a resolution-driven VLM OOM, a grounding/segmentation dtype mismatch)
and two open gaps carried into Phase 3.2 (automatic VLM operation still returns all-STATIC
on every real page tested so far; the one successful render used a human-authored fallback
object and exhibits a real grounding-accuracy defect, not just a mechanical one).

**Phase 3.2 — Semantic reliability: VLM targeting + grounding-target validation —
implemented, real end-to-end validation run pending.** Addresses both Phase 3.1 gaps directly.
The analysis prompt now recognizes panel/page-level effect lines and pose-implied motion (not
only deformation drawn on the object itself), and no longer discards SECONDARY/MICRO reads
when ranking candidates (`src/manga_animation/analysis/plan_builder.py`). A new explicit
**target validation stage** (`src/manga_animation/validation`) sits between grounding and
segmentation: grounding now returns ranked candidates instead of only its best guess
(`grounding.ground_object_candidates`), and each candidate gets an explicit ACCEPT/REJECT with
structured diagnostics before segmentation ever runs on it — a technically valid detection is
no longer automatically trusted as semantically correct. See
[ADR 0006](docs/decisions/0006-grounding-target-validation.md) for the real calibration
evidence behind this design (why a simple confidence threshold can't separate Phase 3.1's real
wrong detection from a real correct one at a similar score) and
[`docs/phase3.2-results.md`](docs/phase3.2-results.md) for real end-to-end results once the
remote validation run completes.

**Phase 3.3 — Panel-aware analysis + evaluation framework — implemented, real end-to-end
comparison run completed.** Addresses Phase 3.2's own flagged gap ("no real panel/scene
splitting before the VLM call"). A deterministic, model-free gutter-detection panel splitter
(`src/manga_animation/analysis/panels.py`) — no new model dependency, see
[ADR 0007](docs/decisions/0007-panel-aware-analysis.md) for why a classical CV approach was
chosen over a learned detector — feeds a new panel-aware analysis path
(`analyze_page_panels`) that runs one VLM call per detected panel instead of one per page, with
a defined page-level fallback when detection finds no usable structure. Page-level analysis
(`analyze_page`) is completely unchanged and remains the default
(`run_pipeline(..., analysis_mode="page" | "panel")`). A new evaluation framework
(`src/manga_animation/evaluation`) adds a real, honest 5-page dataset across 4 series
(`configs/phase3_3_eval_dataset.yaml`) and reproducible metrics (every rate reported with its
sample-count denominator). The real comparison result, run on the remote GPU worker: panel-aware
analysis is real, tested, and produces at least one materially different VLM read on a real
page, but showed **no measurable end-to-end reliability improvement** over page-level analysis
on this small dataset — reported honestly as a null result, not reframed as a win. See
[`docs/phase3.3-results.md`](docs/phase3.3-results.md) for the full real results, including a
newly-found real visual defect (an oversized grounding box on a `rotate` transform) and a
significant new VLM cross-session nondeterminism finding. Two focused follow-ups before Phase 4
started: an evaluation-oracle-integrity fix (ground truth made immutable/versioned/independent
of VLM predictions — see [ADR 0009](docs/decisions/0009-evaluation-ground-truth-integrity.md))
and a baseline-cleanup pass (`_check_regression` corrected, uncertain samples explicitly
excluded from binary metrics, two independently-verified positive controls added) — both
documented in [`docs/phase3.3-results.md`](docs/phase3.3-results.md)'s later sections.

**Phase 4 — Layer decomposition — implemented (deterministic, unit-tested; not yet run against
real models on a live GPU worker).** `analysis/plan_builder.py` no longer forces every object
except the chosen PRIMARY to STATIC — a decision the VLM itself marks SECONDARY/MICRO now keeps
real motion, and `pipeline/orchestrator.py` grounds/validates/segments/animates every non-STATIC
object, not just one (PRIMARY keeps its exact pre-Phase-4 hard-fail policy; a SECONDARY/MICRO
object that fails just drops out of the render). A new `Layer` type
(`pipeline/types.py`) and `compositing.composite_frame_stack` generalize single-object alpha
compositing to N simultaneously-animated layers in a deterministic z-order (PRIMARY on top).
See [ADR 0010](docs/decisions/0010-multi-object-layer-decomposition.md) for the full design,
including why this necessarily absorbed most of what the phase table below used to list
separately under "Phase 5" — a layer decomposition that only ever handles one object isn't
really decomposing anything; the two were never separable in practice.

**Hidden-region reconstruction hardening — one real, confirmed bug found and fixed.**
`reconstruction._compute_hole_mask` was computing the complement of the wrong set (`original &
~UNION(frames)`, "never covered by any frame" — instead of `original & ~INTERSECTION(frames)`,
"not covered by every frame"), which is mathematically guaranteed to return an empty hole
whenever any single sampled frame fully reproduces the original mask — which frame index 0
always does, for every `cycle`-mode motion's rest pose. Confirmed on real data: run against
`examples/sample_page_01.png`'s actual hair region, the old formula found **zero** hole pixels;
the fix finds **70,343** (62% of the mask), and rendering a mid-swing frame without the fix
shows a real, visible ghosting defect the fix's hole exactly covers. See ADR 0010's "Revision"
section for the full audit, the real-data comparison, and which other candidate failure modes
(degenerate/empty masks, boundary-touching masks, disconnected holes, thin regions,
cross-object hole safety) were checked and found already correct. Text/line-art-safe inpainting
*content* remains a real-model-quality question outside deterministic code's reach — documented,
not claimed as verified.

No live Kaggle/Jupyter worker was available during this phase, so the pipeline logic above has
been validated with deterministic fake-client tests (plus the one real-image hole/compositing
check above, using a placeholder fill in place of real LaMa inference) — real-model validation is
real, disclosed future
work, not claimed here.

**Phase 5 — Secondary/micro motion, multi-object plans — software substantially delivered as
part of Phase 4; real-page VLM evidence now obtained, real end-to-end render still not
observed.** A repository-level scope audit confirmed Phase 5's own defined scope (this row)
was already implemented by ADR 0010's Phase 4 work — no new production code was needed. Added
three regression tests (`tests/test_pipeline.py`) proving object identity survives grounding →
segmentation → animation → reconstruction without cross-associating two simultaneously-animated
objects, verified against a deliberately introduced mask/motion-swap bug to confirm they'd
actually catch it. Separately, with the project owner's live Kaggle T4 session, `analyze_page`
was run against 5 real pages (3 attempts each, real `qwen2.5-vl-7b-instruct`): 2/5 pages
reproducibly (3/3 attempts) produced a genuine simultaneous PRIMARY + SECONDARY/MICRO plan —
resolving ADR 0010's "no real page has ever produced one" open question. Running the full
pipeline against both of those pages, however, failed before any SECONDARY/MICRO object was
reached: both share a PRIMARY `weapon` object that Grounding DINO fails to localize correctly
in this art style (one page: 0 detections above threshold, reproducing
`docs/phase3.2-results.md`'s original finding on the same page; the other: 3 candidates, all
rejected by target validation) — a real, pre-existing grounding limitation, not a Phase 4/5
multi-object defect. See ADR 0010's "Revision (Phase 5 audit)" section for the full real-run
table and honest gap characterization.

**Phase 5.1 — panel-aware grounding: the `phase3_action_page.png` grounding failure above is
root-caused and fixed; a real multi-object-eligible end-to-end render is now observed for the
first time.** A follow-up live investigation found the failure was never weapon-specific: the
full 720x5062 page produced 0 Grounding DINO candidates for *every* tested category (weapon,
hair, person, character, face alike), while the same categories scored strongly (0.34-0.76) on
real, already-computed panel crops from `analysis/panels.py::detect_panels` — a preprocessing/
scale effect on this extreme-aspect-ratio page, not a model or lexical limitation. Grounding DINO
now runs against an object's real panel crop when one is known (full-page fallback unchanged for
pages with no real panel structure), translating detections back to full-page coordinates before
anything downstream sees them — see [ADR 0011](docs/decisions/0011-panel-aware-grounding.md) for
the full design, coordinate contract, and test coverage. Real GPU validation: the same PRIMARY
`weapon`/`ROTATE` object that previously failed at grounding on `phase3_action_page.png` now
grounds (score 0.43), passes real semantic validation (Qwen2.5-VL, confidence 0.95) and real
transform-geometry validation, segments (real SAM 2.1, IoU 0.82), reconstructs, and renders a
real, seamless-loop-verified `output.mp4` — the first real end-to-end render this project has
produced for a page with genuine multi-object potential. `eval_weapon_effects.png`'s own
PRIMARY failure is unaffected by this fix (its panel detector already returns a whole-page
fallback panel, so its grounding crop is unchanged) and its existing semantic/geometric
rejections were reconfirmed byte-identical under the new architecture — no validation was
weakened to obtain the improvement above.

**Phase 6 — seamless-loop hardening + local-region rendering at scale — implemented, locally
verified.** Two independent fixes. First: `AnimationPlan` previously accepted
`loop_mode="once_hold"` under `loop.seamless=True`, a combination that always produces a
visible jump at the loop boundary (`once_hold` holds its end state rather than returning to
rest); the schema now rejects it, and the pre-existing `cycle`/non-integer-speed error message
no longer misleadingly suggests `once_hold` as an equivalent fix to `ping_pong`. Second (the
higher-risk half): `animation/transforms.py`, `compositing/__init__.py`, and
`reconstruction/__init__.py`'s `_compute_hole_mask` previously ran their actual CV work
(`cv2.warpAffine`/`cv2.remap`, alpha blending, hole-mask accumulation) over the **whole page**
every frame regardless of how small the animated object was — a real gap against
`docs/architecture.md`'s "Local Modification" principle. All three now restrict that work to
the relevant region, verified bit-exact (mesh_warp/opacity/reconstruction/compositing) or
within an explicitly justified, bounded `±1`-uint8 floating-point tolerance (the affine
`warpAffine` path only, confined to inside the moving object's own footprint, never the static
region) against a kept-verbatim copy of the old full-page implementation, across small/large/
edge/corner/extreme-aspect-ratio/off-page/distant-pivot/multi-object/overlapping-layer cases.
Real measurements: the raw interpolation cost itself now scales cleanly with the animated
region (25x-109x faster in isolation, growing with page size, for a fixed small object,
completely decoupled from page pixel count). A post-Phase-6 closure audit independently
verified this but found the initially-reported end-to-end speedup (~1.15x-2x) was bottlenecked
almost entirely (91-97%) by an unrelated, avoidable redundancy — `generate_transformed_layer`
recomputing the object's bbox via a full-page scan on every single frame instead of reusing the
one segmentation already computed — not, as first reported, by the small full-page placement
array `Layer`'s contract still requires allocating every frame. With that hoisted (an optional
`object_bbox_px` parameter, now fed from `SegmentationResult.bbox`), end-to-end speedup reaches
~20x-60x, close to the raw-interpolation numbers above. See
[ADR 0012](docs/decisions/0012-phase6-seamless-loop-and-local-rendering.md) for the full design
and [`docs/phase6-results.md`](docs/phase6-results.md) for the complete evidence — including the
audit correction — all captured locally (no GPU/remote work needed — every fix here is
deterministic CPU/OpenCV/NumPy code).

Planned phases:

| Phase | Scope |
|---|---|
| 1 | Engineering foundation: repo, config, schema, tests, docs, agents/skills |
| 2 | Model benchmarking & selection (VLM, grounding, segmentation, inpainting) |
| 3.1 | First end-to-end vertical slice: one real page through every stage |
| 3.2 | VLM targeting reliability + explicit grounding-target validation gate |
| 3.3 | Panel-aware analysis + reproducible evaluation framework |
| 4 | Layer decomposition — **implemented**; hidden-region reconstruction hardening — **one real bug found and fixed** (`_compute_hole_mask`), other candidate failure modes audited and found already correct; real-model validation still pending |
| 5 | Secondary/micro motion, multi-object plans — **software substantially delivered as part of Phase 4** (see above); real-page VLM evidence of genuine multi-object plans **now obtained** (2/5 real pages, 3/3 reproducible); Phase 5.1 root-caused and fixed the grounding limitation blocking both real multi-object pages' PRIMARY object on `phase3_action_page.png` (panel-aware Grounding DINO cropping, ADR 0011) and observed a real, complete end-to-end render for the first time; `eval_weapon_effects.png`'s separate, unrelated PRIMARY rejection (candidate-correctness, not scale) is unaffected and unresolved |
| 6 | Seamless-looping/rendering hardening at scale — **implemented**: once_hold+seamless schema gap closed (ADR 0012); animation/compositing/reconstruction localized to the animated region, verified bit-exact/bounded-tolerance against the old full-page implementation, with real local performance evidence |
| 7 | End-to-end QA, evaluation, regression testing — **substantially complete**: deterministic regression layer implemented (multi-object encode/decode-verified regression, whole-pipeline determinism, panel-aware regression on `phase3_action_page.png`'s real geometry, a defensive `panel_bbox_px` check, opt-in Phase 6 performance-regression protection); evaluation reporting extended from PRIMARY-only to SECONDARY/MICRO objects (ADR 0013, closing ADR 0010's explicitly-deferred item); real-model evaluation on a live Kaggle GPU worker produced **the first real, fully automatic, successfully rendered multi-object output this project has observed** (3 real pages, up to 5 simultaneously-animated real objects on one page, real LaMa reconstruction, real decode-verified seamless loop) — closing the gap ADR 0010's "Revision (Phase 5 audit)" and ADR 0011 both left open ("no real page has yet produced a successfully rendered multi-object output"); real LaMa visual QA performed for the first time (previously only placeholder-fill-validated) with a genuinely clean result — see [`docs/phase7-results.md`](docs/phase7-results.md) for the complete evidence |

## Architecture overview

```text
Manga page
    -> Panel / scene analysis
    -> VLM semantic understanding
    -> Structured Animation Plan
    -> Object grounding
    -> Precise segmentation
    -> Layer decomposition
    -> Optional hidden-region reconstruction
    -> Deterministic / kinematic animation
    -> Secondary motion
    -> Original-image compositing
    -> Seamless loop
    -> H.264 video
```

Full principles are documented in [`docs/architecture.md`](docs/architecture.md).
The two load-bearing rules:

1. **The local project is the canonical source of truth.** Kaggle/Jupyter GPU
   servers are ephemeral remote compute workers, never the only copy of the
   code.
2. **Static is a valid result.** If there's no visually justified reason for
   an object to move, the system should prefer leaving it static over
   inventing motion.

## Hardware

Developed against:

- **Local:** Apple Silicon (M1 Max, 32GB unified memory), macOS, arm64 — no
  NVIDIA GPU. PyTorch here means CPU or MPS backend, not CUDA.
- **Remote (as needed):** Kaggle T4 / L4 or other Jupyter GPU workers, CUDA.

Because local and remote hardware differ (MPS vs. CUDA, memory budgets,
available dtypes), every hardware-sensitive parameter (`device`, `dtype`,
`batch_size`, `resolution`, `model_variant`, `num_workers`, ...) is
configuration-driven — see [`src/manga_animation/core/config.py`](src/manga_animation/core/config.py)
and [`configs/`](configs) — never hardcoded in pipeline code.

## Local setup

Requires Python 3.11+. This project uses [`uv`](https://docs.astral.sh/uv/)
for environment and dependency management.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the environment and install base + dev dependencies
uv sync --extra dev

# activate it (optional — `uv run` works without activating)
source .venv/bin/activate
```

Optional dependency groups (added as later phases need them):

```bash
uv sync --extra cv      # OpenCV, for segmentation/compositing/animation stages
uv sync --extra video   # ffmpeg-python wrapper (the `ffmpeg` binary itself is a system dependency)
uv sync --extra ml      # torch, for VLM/grounding/segmentation model stages
```

> `ffmpeg` (the binary) and, later, CUDA/NVIDIA drivers on GPU workers are
> **system** dependencies, not installed by `uv`/`pip`. Phase 1 does not
> require `ffmpeg` to be installed locally; it will be needed starting with
> the video-rendering stage.

## Development workflow

```bash
# run tests
uv run pytest

# lint
uv run ruff check .

# type-check
uv run mypy src

# format
uv run ruff format .
```

Configuration lives in [`configs/`](configs) as YAML, loaded and validated
through pydantic models in `src/manga_animation/core/config.py`. Don't scatter
`device="cuda"` / magic numbers through pipeline code — add a field to the
config schema instead.

## Testing

```bash
uv run pytest -v
```

Phase 1 tests cover configuration validation, Animation Plan schema
validation/serialization, deterministic seed behavior, loop-parameter
validation, and package imports. They deliberately avoid tests that only
assert "the class exists" — see `tests/`.

## Remote GPU workflow (Kaggle / Jupyter)

Kaggle/Jupyter GPU servers are **ephemeral remote compute workers**, never
the canonical copy of the project. Workflow:

```text
local: edit code -> git commit -> git push
remote: git pull -> run experiments (GPU-bound work only)
remote: git commit/push  (only if source files changed on the remote)
local: git pull
```

Do not hand-copy files to/from the remote as a substitute for git. Avoid
editing the same files on both sides at once.

If a task requires connecting to a Kaggle/Jupyter server, the assistant will
ask you for the server URL explicitly rather than guessing or reusing a
possibly-stale one.

## Repository layout

```text
src/manga_animation/   # application code (empty stage packages until Phase 2+)
tests/                 # pytest suite
configs/                # YAML configuration files
docs/                   # architecture, pipeline, schema docs, ADRs
.claude/agents/         # specialist Claude Code agents for this project
.claude/skills/         # domain-specific Claude Code skills
scripts/                # one-off developer scripts
examples/               # example inputs/usage (populated in later phases)
outputs/                # git-ignored generated artifacts (videos, frames, debug)
```

## License

MIT (see `pyproject.toml`).
