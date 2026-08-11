---
name: evaluation
description: Concrete, numeric checks for whether pipeline output actually satisfies the project's guarantees — artwork/static-region preservation, loop quality, motion/temporal quality, artifact detection, and reproducibility. Load when writing or running QA checks on generated output, or reviewing whether a result is acceptable.
---

# Evaluation

Concrete, numeric ways to check the guarantees in `docs/architecture.md` actually hold for
a given output — not subjective "does it look okay" review. Owned in spirit by `qa-agent`,
usable by anyone checking pipeline output.

## Artwork / static-region preservation

The most important check. For every frame, outside every animated object's (post-transform)
mask, pixels must match the source image:

- Compute a per-pixel diff (or MSE) restricted to the inverse of the union of all animated
  masks for that frame. This should be exactly zero for uncompressed intermediate frames,
  and near-zero (bounded by codec quantization, not by compositing error) after H.264
  encoding.
- A nonzero diff *inside* the expected static region means a compositing bug (a transform
  or mask leaked beyond its intended area) — treat this as a hard failure, not a quality
  nit.

## Loop quality

- **Frame-0 vs. frame-N continuity**: diff frame 0 against a frame regenerated at
  `t = duration_s` (not literally "the last rendered frame," which is at `t = duration_s -
  1/fps`) — this is the actual seam the loop constraint promises. Small diffs are expected
  only where `crossfade_frames > 0` was used to blend it away.
- **Visual seam check on playback**: looping the rendered video back-to-back should show no
  perceptible pop/jump at the wrap point — automatable via the frame-0/frame-N diff above,
  but worth a manual spot-check too when reviewing a new model/technique.

## Motion and temporal quality

- **Smoothness**: frame-to-frame displacement of an animated object's centroid (or mean
  optical flow within its mask) should vary smoothly according to its `MotionSpec.easing`
  curve — a sudden discontinuity between adjacent frames indicates a sampling bug (see the
  `cv-animation` skill's "edge popping" note), not real motion.
- **Amplitude sanity**: measured pixel displacement of an object across the loop should be
  in the same ballpark as `MotionSpec.amplitude` implies for its `transform_kind` and the
  object's actual size — a large mismatch suggests a unit/scale bug (amplitude is
  normalized; verify it was denormalized against the right reference, panel diagonal vs.
  object bbox vs. full page).

## Object consistency

- An animated object's mask area shouldn't fluctuate wildly frame-to-frame (beyond what
  `scale`/`mesh_warp` legitimately implies) — large unexplained area swings suggest mask
  regeneration inconsistency rather than a stable tracked layer.
- Nothing should appear/disappear abruptly mid-loop outside of an intentional
  `loop_mode="once_hold"` or `"ping_pong"` transition.

## Artifact detection

- Check mask-edge regions specifically (not whole-frame metrics, which dilute a small
  visible seam into an acceptable-looking average) for halo/ghosting from feathering or
  interpolation mismatches (see `cv-animation`).
- Compare a masked-out region's background against the *original* image at that location —
  any visible remnant of the pre-transform object position indicates a hidden-region
  reconstruction gap or a compositing order bug.

## Reproducibility

- Same input image + same `AnimationPlan` + same `PipelineConfig.seed` should produce
  byte-identical (or, across nondeterministic-but-seeded GPU ops, numerically very close)
  output on a re-run — see `manga_animation.core.set_global_seed`. A regression test that
  runs the same input twice and diffs the output is the cheapest way to catch an
  accidentally-nondeterministic stage (e.g. an unseeded random call, or a
  dict/set-iteration-order dependency).
