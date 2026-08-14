# Phase 13 results: panel extraction and independent scene animation

Status: **PARTIALLY COMPLETED.** The local panel-first architecture, deterministic scene crops,
independent outputs, manifest, failure isolation and regression coverage are implemented. Real
GPU validation is blocked by the supplied Jupyter proxy returning `404`; no GPU result is claimed.

## Architecture change

`run_page_panels` detects panels, creates bounded scene crops, and invokes the existing
`run_pipeline` once per crop. The scene crop is the actual grounding, segmentation, animation,
compositing and video canvas. The strict logical panel rectangle is retained separately.

## Panel detection and scene crop

The existing deterministic XY-cut detector is reused. Its stable reading-order candidates expose
the tight logical rectangle through `PanelCandidate.panel_bbox`. `derive_scene_crop_bbox` adds an
8% bounded context margin, clamps to page bounds, and limits expansion toward a neighboring panel
at the gutter midpoint. Touching detector cuts receive only a small context allowance.

## Safety and statuses

Grounding bboxes materially crossing another logical panel are rejected without splitting or
synchronizing objects. Each panel is independently `PASS`, `STATIC`, `REJECTED` or `ERROR`.
The manifest is written after each panel and records bboxes, crop path, output path, failure data,
and measured crop/runtime/frame metrics.

## Validation

Local tests pass for crop containment, page-edge and adjacent-panel behavior, all-STATIC plans,
coordinate translation, cross-panel rejection, multiple panel videos, manifest contents, failure
isolation and resumability behavior. Fake-client end-to-end rendering produced one H.264 output
per panel. Real example pages were used for local detector/crop inspection; targeted native-
resolution GPU validation remains pending.

## Performance and limitations

The manifest records per-panel crop dimensions, pixel counts, frame counts and runtime, plus page
crop-pixel totals and elapsed time. No speedup is claimed before real GPU measurements. The crop
margin is a deterministic heuristic and the existing XY-cut detector remains limited on complex
traditional layouts and flat-color content.

## Git

Branch: `phase-13-panel-animation`. Commit and PR details are recorded after local verification.
