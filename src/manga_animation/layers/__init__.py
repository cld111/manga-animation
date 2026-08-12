"""Layer decomposition of segmented objects for independent animation (Phase 4).

Implemented, but thinner than this package's own name once suggested it would be -- see
`docs/decisions/0010-multi-object-layer-decomposition.md`. The actual pieces:

- The `Layer` type itself lives in `manga_animation.pipeline.types`, alongside every other
  cross-stage result contract (`GroundingResult`, `SegmentationResult`, `ReconstructionResult`,
  ...) -- keeping it there follows this project's own existing convention rather than carving
  out an exception for this one type.
- The actual decomposition -- turning each animated `ObjectPlan` + its `SegmentationResult`
  into a `Layer` (one `(image, mask)` pair per frame, plus a `z_order`) -- is a thin assembly
  step inside `pipeline.orchestrator.run_pipeline`'s animation stage, not substantial
  standalone logic that earns its own module. `animation.generate_transformed_layer` (unchanged
  since Phase 3.1) still does the real per-frame transform math.
- Multi-layer compositing lives in `compositing.composite_frame_stack`.

This module intentionally stays empty. It exists so `docs/pipeline.md`'s stage diagram still
names a real, findable home for "where is layer decomposition" even though the answer turned
out to be "distributed across the two modules above," not "a new package."
"""

from __future__ import annotations
