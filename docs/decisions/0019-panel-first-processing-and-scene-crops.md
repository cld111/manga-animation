# 19. Panel-first processing and scene crops

Status: Accepted

## Context

Panel-aware analysis already improved grounding reliability, but `run_pipeline` still rendered a
whole page and used panels only as model context. A manga panel boundary is not necessarily an
object boundary: visible scene content can extend beyond the strict panel rectangle. Cropping
the final canvas to that rectangle would cut such content, while an unrestricted expansion would
import neighboring panels.

## Decision

`run_page_panels` is the production page entry point. Deterministic panel detection creates one
stable `PanelUnit` per panel. Each unit has separate logical `panel_bbox` and processing/output
`scene_crop_bbox` geometry, a persisted crop, an explicit status, and an optional video.

The scene crop expands by a bounded fraction of the logical panel and is clipped by page bounds
and nearby panel geometry. A gutter gap is shared only up to its midpoint; when detector logical
boxes touch at a cut, only a small bounded context allowance is used. This is a conservative
context heuristic, not object ownership inference.

The existing stage pipeline is reused once per crop. A candidate that materially overlaps both
its current logical panel and another logical panel is rejected. The object is not split or
synchronized across outputs. Panel failures are isolated and written to a machine-readable page
manifest; completed PASS/STATIC records can be reused on a later invocation.

## Consequences

- One page can produce `panel_001.mp4`, `panel_002.mp4`, and so on, each at its scene-crop
  resolution.
- Existing geometric, semantic-mask, transform-aware, overlap and loop safety gates remain in
  the shared pipeline.
- The scene-crop margin is deterministic and testable but not yet calibrated against a broad
  real overflow dataset.
- Final page-video assembly and scene transitions remain out of scope.
