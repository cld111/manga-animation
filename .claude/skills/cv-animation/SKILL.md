---
name: cv-animation
description: Implementation guidance for deterministic OpenCV/NumPy motion — applying MotionSpec transforms to masked layers, mesh warping, and compositing back over the original image without disturbing unaffected pixels. Load when implementing or reviewing the CV transform/compositing code itself.
---

# CV animation

Implementation guidance for the deterministic transform + compositing stage. Principle
reference: "Deterministic First" and "Local Modification" in `docs/architecture.md`. This
skill is about *how to implement it correctly*, not about choosing motion parameters
(that's the `animation-planning` skill).

## Per-frame transform pipeline

For each animated `ObjectPlan`, per output frame `t`:

1. **Sample the motion curve** at `t` from `MotionSpec` (`amplitude`, `phase`, `speed`,
   `easing`, `timing`) — produce a single scalar/vector displacement for this frame, not a
   whole-sequence array up front, so timing (`delay_s`, `duration_s`, `loop_mode`) is
   easy to reason about per-frame.
2. **Resolve `pivot`** from its normalized `reference` frame (`object_bbox`/`panel`/`page`)
   to actual pixel coordinates in *this* image, at *this* resolution — never bake a pixel
   pivot into code, since resolution varies by `PipelineConfig.resolution`.
3. **Build the transform** (affine matrix for `translate`/`rotate`/`scale`/`shear` via
   `cv2.getRotationMatrix2D`/`cv2.warpAffine`; a displacement field for `mesh_warp` via
   `cv2.remap`; a plain alpha multiply for `opacity`) around the resolved pivot.
4. **Apply the transform to the masked layer only** — extract the object's pixels via its
   mask, transform that extracted layer (plus its mask, with matching interpolation), never
   the full frame.
5. **Composite back** over the original (untouched) image using the *transformed* mask as
   the alpha channel.

## Preserving original pixels

This is the hard constraint `qa-agent` checks: every pixel outside a transformed object's
(post-transform) mask must be bit-identical to the source image, every frame.

- Never write to the full frame buffer and then "patch in" the original background — build
  each frame by compositing transformed layers *onto a fresh copy* of the untouched source,
  so there's no way for an unrelated pixel to be touched by accident.
- Interpolation at mask edges (`cv2.INTER_LINEAR`/`cv2.INTER_CUBIC`) will blend
  object-colored and background-colored pixels right at the boundary — feather the alpha
  mask edge deliberately (a small Gaussian blur on the mask, not the image) rather than
  letting hard-edge aliasing or uncontrolled bleed appear.
- Don't upscale/downscale the full frame as a side effect of transforming one small object
  — only the extracted layer's local canvas should change size/warp, then it's composited
  back at the original frame's resolution.

## Mesh warping specifics

- Build the displacement field only over the object's mask region (plus a small margin for
  the feathered edge) — don't compute a full-frame `remap` when only a small area moves.
- Keep the warp field smooth (e.g. derived from a low-frequency sinusoidal or a small
  control-point grid) — high-frequency/noisy displacement fields read as glitchy rather
  than as cloth/hair motion.

## Common artifacts to check for

- **Double-transformation**: a child object accidentally inheriting its parent's transform
  *and* applying its own on top when it should compose them intentionally (or vice versa —
  losing the parent's motion entirely). Decide explicitly whether a child's transform is
  relative to its parent's already-transformed position or independent, and be consistent.
- **Mask/layer misalignment**: the transformed mask and transformed layer pixels must use
  identical transform parameters and interpolation — any mismatch shows as a halo.
- **Edge popping between frames**: if per-frame transforms are computed independently
  without a shared continuous parameterization (e.g. re-deriving `easing` output instead of
  sampling one continuous function of `t`), adjacent frames can jump — always sample from
  one continuous motion function across the whole frame range.
