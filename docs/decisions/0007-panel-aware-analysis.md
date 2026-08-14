# 7. Deterministic gutter-based panel detection for panel-aware analysis (Phase 3.3)

Status: Accepted

## Context

`docs/pipeline.md` has always documented "Panel / scene analysis" as a distinct step before
VLM understanding, but it was never implemented — `analysis/plan_builder.py` always treats
the entire page as one `PanelPlan` covering `(0, 0, 1, 1)`. Both Phase 3.2's real end-to-end
run (`docs/phase3.2-results.md`, "Remaining limitations") and Phase 3.1's original finding
(`docs/phase3-results.md`, a 720x5062px page) confirmed this is a real gap: a long page's
motion-relevant region can be a small fraction of the page after the page-level resize
`_resized_for_vlm` applies to stay within `PipelineConfig.resolution`, which shrinks fine
motion cues (a single blade, a strand of hair) proportionally to whatever else shares the
page.

Per the Phase 3.3 brief, before adding any model the existing codebase and real sample pages
were inspected for whether a *deterministic* panel detector is sufficient:

- `opencv-python-headless` is a base project dependency in `pyproject.toml`, already imported
  at module level in `animation/transforms.py` — no separate runtime extra is needed for
  classical image processing.
- All three real sample pages this project has ever used (`examples/sample_page_01.png`,
  `examples/sample_page_02.png`, `examples/phase3_action_page.png` — visually inspected for
  this decision) are digital, full-color, single-column **webtoon-style** pages: panels are
  separated by wide, near-uniform-color (usually white) horizontal gutters, not drawn with
  the traditional dense, irregular multi-column grid of a printed manga page. A gutter is
  trivially and reliably detectable as a run of near-uniform image rows — no learned model
  is needed to find *this specific, real, evidenced kind of boundary*.
- No labeled panel-detection dataset exists in this project (or is proposed by the brief) to
  train or calibrate a learned detector against. Introducing one now would repeat exactly the
  mistake `docs/decisions/0006-grounding-target-validation.md` already argued against for a
  different stage: an uncalibrated component is not obviously better than a documented,
  inspectable deterministic rule, and would add a new model dependency (weights, GPU
  residency, license) for a problem the real evidence doesn't show needs one.
- `docs/architecture.md`'s "Deterministic First" principle already states the project's
  default: prefer deterministic CV wherever the problem is expressible that way. Gutter
  detection via recursive background-uniformity splitting ("XY-cut") is a standard,
  well-understood comic/document-segmentation technique that is directly expressible this
  way.

## Decision

Add a **deterministic, model-free panel detector** (`analysis/panels.py::detect_panels`) as
an independent stage/abstraction, decoupled from Grounding DINO, SAM 2.1, and the animation
engine — none of those stages change or become panel-aware:

1. **Algorithm**: recursive gutter-based splitting ("XY-cut"). Convert the page to grayscale;
   find horizontal runs of near-uniform rows (a "gutter") above a minimum thickness; split the
   page into vertical bands at those gutters; within each band (if tall enough to plausibly
   contain a side-by-side layout), recurse once on columns to find vertical gutters. This
   directly supports "very tall manga pages" (the motivating Phase 3.1 case) since a tall page
   with several stacked panels splits into proportionally-sized bands, each analyzed at a
   normal aspect ratio instead of being crushed by one page-wide resize.
2. **Structured output**: `PanelCandidate` (`pipeline/types.py`) — `id`, `bbox` (`BBoxPx`,
   already page-space by construction), `crop` (the pixels for exactly `bbox`, expanded by a
   small context margin *before* being recorded as `bbox` — so `crop.shape == bbox` extent is
   a checked invariant, not two independently-tracked regions), `confidence`, `source`
   (`"gutter_xy_cut"` or `"fallback_full_page"`), `metadata` (gutter thickness/uniformity,
   recursion depth, band/column index — diagnostic, not load-bearing).
3. **Confidence is explicit, not assumed**: a panel bounded by real detected gutters on both
   sides scores higher than one bounded by the page edge (no gutter evidence on that side); a
   degenerate/too-small image returns zero panels; a page with no internal gutters at all
   returns exactly one candidate spanning the page (a valid splash-page read per the
   `manga-analysis` skill, not a failure) at reduced confidence.
4. **Coordinate mapping is a first-class, tested concern**: `pipeline/types.py` gains
   `bbox_px_to_normalized`/`normalized_bbox_to_px`, converting between a detector's pixel-space
   `BBoxPx` and the schema's page-normalized `BBox`. The recursive splitter itself always
   accumulates a parent region's page-space origin into every child region before returning it
   (never returns crop-local-only coordinates), so every `PanelCandidate.bbox` this module
   returns is already canonical page space — `analysis/plan_builder.py`'s panel-aware path only
   has to normalize it once for `PanelPlan.bbox`. No downstream stage (grounding, validation,
   segmentation, animation) needs to know or care whether a target's panel came from page-level
   or panel-level analysis — they already only ever consumed page-space pixels/normalized
   `BBox`es, unchanged by this phase.
5. **Page-level fallback is defined, not silent**: `analysis/plan_builder.py::analyze_page_panels`
   (panel-aware entry point) falls back to the existing, unmodified `analyze_page` (full-page
   VLM call) whenever `detect_panels` returns zero panels, or when every detected panel's VLM
   read is STATIC or unusable and the page-level path might still find something (see that
   function's docstring for the exact fallback trigger). `analyze_page` itself is completely
   unchanged — Phase 3.2's behavior remains an always-available path
   (`run_pipeline(..., analysis_mode="page")`). **No longer `run_pipeline`'s default as of Phase
   10** (see `docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md`) —
   real Phase 9/10 evidence found panel-level analysis substantially more reliable; this
   historical note is left as originally written above it, not edited to pretend the default
   never changed.

## Consequences

- New types: `PanelCandidate`, `PanelSource` (`pipeline/types.py`).
- New module: `analysis/panels.py` (the detector — no VLM/torch import, pure numpy/opencv,
  consistent with `analysis/client.py`'s existing "heavy ML stays out of anything that must be
  importable without the `ml` extra" rule).
- `analysis/plan_builder.py` gains `analyze_page_panels`, sitting alongside the unchanged
  `analyze_page`; both build the same `AnimationPlan` shape (multiple `PanelPlan`s is already
  legal per the existing schema — no schema change needed).
- `pipeline/orchestrator.py::run_pipeline` exposes
  `analysis_mode: Literal["page", "panel"] = "panel"` — panel-level is the current default
  since Phase 10; page-level remains available when passed explicitly.
- Grounding still runs over the *whole page image* exactly as before, per the brief's explicit
  "do not tightly couple panel detection to Grounding DINO" instruction — a panel-aware
  analysis run changes *which candidates the analysis stage considers and where they're
  attributed in `plan.panels`*, not what pixels grounding/segmentation ever see.

## Open questions / limitations

- All real evidence behind this decision is single-column, webtoon-style, full-color pages —
  the only kind of real page this project has ever fetched (see
  `docs/decisions/../phase3-results.md`, `phase2-benchmark-results.md`). Whether gutter-based
  XY-cut generalizes to a traditional, densely-arranged multi-column printed manga grid (panels
  of unequal size, diagonal/non-rectangular panel borders, panels that bleed into each other
  with no gutter at all) is **untested** — no such page exists in this project's real sample
  set. This is flagged, not silently assumed to generalize.
- A panel border drawn *within* a scene (e.g. an inset reaction-face box overlapping a larger
  background panel) is a real manga convention this detector does not specifically recognize —
  it would either be missed (if it doesn't create a real uniform-row gutter) or mis-split (if
  it does). No real evidence in this project's sample pages exercises this case either way.
- The confidence score is a documented, evidenced heuristic (gutter-boundary presence/absence),
  not a calibrated probability — same status as every other uncalibrated-but-evidenced
  threshold already in this codebase (`pipeline/types.py`'s coverage-fraction bounds,
  `validation/validate.py`'s margin fraction).
