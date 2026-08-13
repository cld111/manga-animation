"""Phase 9: one automated visual-artifact signal, added only because it was validated against

real evidence first -- not a speculative addition to grow the metric count (the Phase 9 brief's
own explicit warning). `docs/phase8.3-results.md` root-caused a real "hard vertical seam" defect
(`verified_action_1`/`phase3_action_page`, section 7) to a SAM 2.1 mask over-segmenting into
adjacent background along one straight edge of its own tight bbox -- an asymmetric edge-touch
signature (one side hugged, the opposite side clean) already confirmed on real downloaded mask
data (Phase 8.3 section 3.1: 45.5% vs 0.6%/2.2-20.2%) and fixed at the segmentation stage
(`segmentation.segment._validate_mask_shape`).

This module re-derives the SAME geometric signature independently, from final composited RGB
pixels only (no access to segmentation masks) -- a second, black-box QA layer over the actual
rendered output, not a replacement for the production-stage gate. Empirically validated during
Phase 9 curation directly against the real, locally-retained pre-fix defect evidence
(`outputs/videos/phase8_evidence/panel_phase3_action_page.mp4`,
`outputs/videos/phase8_evidence/panel_verified_action_1.mp4`,
`outputs/videos/phase8_evidence/panel_verified_action_2.mp4` -- git-ignored generated
artifacts, not committed, see `docs/phase9-results.md` for the exact numbers this validation
produced): the real seam defect's largest changed-region component showed 82-92% edge-touch on
one side vs. 4.7-5.7% on the opposite side; the real, *different* "duplicate silhouette" defect
and a clean control both stayed under 21% on every edge -- a wide, real margin, not a
theoretical one.

**Deliberately narrow claim**: this catches the axis-aligned/rigid-edge "seam" defect class
only. It does NOT detect the unrelated "duplicate silhouette" (ghosting) defect class (validated
negative above) or any other visual-quality problem -- see `LOOP CONTINUITY != GENERAL VISUAL
QUALITY` in the `evaluation` skill; this is one more narrowly-scoped signal, not a general
artifact detector, and does not replace human visual QA (Phase 9 brief section 9).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_DIFF_THRESHOLD = 10
"""Per-channel summed abs-diff above which a pixel counts as "changed" -- generous enough to

ignore H.264 quantization noise on an unchanged background (see `rendering.encode`'s own
codec-tolerance framing) while still catching a real, visible moved/leaked region."""

_MIN_COMPONENT_AREA_PX = 200
"""Ignore connected components smaller than this -- compression artifacts/antialiasing produce

tiny scattered changed-pixel specks that are never a real animated object or a real leaked
region; below this size there usually isn't enough of a boundary to measure "straightness" of
in the first place."""

_DILATION_KERNEL_PX = 15
"""Merges nearby fragments of one real animated region's diff into a single connected component

(e.g. antialiased/thin edges that would otherwise fragment into several tiny components) --
evidenced against the real defect videos above, where the true single largest object-level
diff region is what carries the asymmetric-edge signature; without this, the signature can get
lost across many small fragments instead of one dominant component.
"""

_HUGGED_EDGE_THRESHOLD = 0.3
"""A component edge counts as "hugged" when this fraction of its own rows/columns touch that

edge -- reuses the exact value `segmentation.segment._validate_mask_shape` already established
and independently reviewed (Phase 8.3, ADR 0015) for the equivalent check on raw segmentation
masks. The real defect video's worst component measured 82-92% here (frame0-vs-frame24 and
frame0-vs-frame95); the real non-defective components (the different ghosting defect and the
clean control) never exceeded 21%."""

_OPPOSITE_EDGE_CEILING = 0.15
"""The geometrically opposite edge must stay below this for a "hugged" edge to count as a real

asymmetric seam rather than a genuinely rectangular object (which would show BOTH opposite edges
hugged together, e.g. a banner) -- mirrors `_validate_mask_shape`'s own asymmetry refinement,
added after that check's independent review found a false-positive risk on legitimately
rectangular real objects (Phase 8.3 section 10). The real seam defect's opposite edge measured
4.7-5.7%; comfortably under this ceiling with real margin."""

_DEFAULT_SAMPLE_COUNT = 12
"""How many frames (roughly evenly spaced, always including frame 0's comparison partners

across the sequence) get diffed against frame 0 -- a black-box QA check has no access to a
sample's actual motion curve/peak-displacement timing, so this samples broadly across the whole
loop rather than assuming where the peak is. Kept small (not every frame) since this runs as a
cheap post-hoc CPU check on top of an already-encoded video, not inside the render loop itself.
"""


@dataclass(frozen=True, slots=True)
class ChangedRegionShape:
    """One connected component of the changed-pixel region between two frames, and its own

    per-edge "hugged" fraction -- the raw geometric evidence `axis_aligned_seam_suspected`
    (on `SeamArtifactReport`) is computed from.
    """

    bbox_px: tuple[int, int, int, int]  # x, y, w, h, in the diff image's own pixel coordinates
    area_px: int
    left_edge_fraction: float
    right_edge_fraction: float
    top_edge_fraction: float
    bottom_edge_fraction: float

    @property
    def has_asymmetric_edge(self) -> bool:
        """True if some edge is hugged while its geometrically opposite edge is not -- the

        real, evidenced seam signature (see this module's docstring). Checked on both axes
        independently (a vertical seam hugs left-or-right; a horizontal one hugs top-or-bottom).
        """
        horizontal = (
            max(self.left_edge_fraction, self.right_edge_fraction) >= _HUGGED_EDGE_THRESHOLD
            and min(self.left_edge_fraction, self.right_edge_fraction) < _OPPOSITE_EDGE_CEILING
        )
        vertical = (
            max(self.top_edge_fraction, self.bottom_edge_fraction) >= _HUGGED_EDGE_THRESHOLD
            and min(self.top_edge_fraction, self.bottom_edge_fraction) < _OPPOSITE_EDGE_CEILING
        )
        return horizontal or vertical


def detect_changed_region_shapes(
    frame_a: np.ndarray, frame_b: np.ndarray
) -> list[ChangedRegionShape]:
    """Every real (post-dilation, post-min-area) connected component of the changed-pixel

    region between two same-shape RGB frames, largest area first.
    """
    diff = cv2.absdiff(frame_a, frame_b).astype(np.int32).sum(axis=2)
    raw_mask = (diff > _DIFF_THRESHOLD).astype(np.uint8)
    mask = cv2.dilate(raw_mask, np.ones((_DILATION_KERNEL_PX, _DILATION_KERNEL_PX), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    shapes: list[ChangedRegionShape] = []
    for label in range(1, count):  # label 0 is the background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < _MIN_COMPONENT_AREA_PX:
            continue
        x, y, w, h = (int(stats[label, i]) for i in range(4))
        component = labels[y : y + h, x : x + w] == label
        shapes.append(
            ChangedRegionShape(
                bbox_px=(x, y, w, h),
                area_px=area,
                left_edge_fraction=_edge_touch_fraction(component, axis=1, near_zero=True),
                right_edge_fraction=_edge_touch_fraction(component, axis=1, near_zero=False),
                top_edge_fraction=_edge_touch_fraction(component, axis=0, near_zero=True),
                bottom_edge_fraction=_edge_touch_fraction(component, axis=0, near_zero=False),
            )
        )
    return sorted(shapes, key=lambda s: -s.area_px)


def _edge_touch_fraction(component: np.ndarray, *, axis: int, near_zero: bool) -> float:
    """Fraction of lines perpendicular to `axis` whose nearest changed pixel touches the

    component's own bbox edge on that side. `axis=1` scans rows (checks the left/right edges of
    each row); `axis=0` scans columns (checks the top/bottom edges of each column).
    """
    length = component.shape[1 - axis]
    touching = 0
    lines_with_pixels = 0
    for i in range(length):
        line = component[i, :] if axis == 1 else component[:, i]
        idx = np.flatnonzero(line)
        if idx.size == 0:
            continue
        lines_with_pixels += 1
        edge_idx = idx.min() if near_zero else (line.size - 1 - idx.max())
        if edge_idx <= 2:
            touching += 1
    return touching / lines_with_pixels if lines_with_pixels else 0.0


@dataclass(frozen=True, slots=True)
class SeamArtifactReport:
    """Whether any sampled frame, diffed against frame 0, shows a changed region with the real,

    evidenced asymmetric-edge "seam" signature (see this module's docstring). A black-box,
    post-hoc, whole-video signal -- complements, does not replace, `LoopMetrics` (which only
    ever compares the wrap transition, never a mid-cycle frame -- the exact gap
    `docs/phase8-results.md` section 9 documented) and human visual QA.
    """

    frames_compared: int
    seam_suspected: bool
    worst_component: ChangedRegionShape | None
    """The changed-region component with the strongest asymmetric-edge signal across every

    sampled frame, regardless of whether it crossed the `seam_suspected` threshold -- useful for
    inspecting a near-miss, not just a pass/fail bit."""


def detect_seam_like_artifacts(
    frames: list[np.ndarray], *, sample_count: int = _DEFAULT_SAMPLE_COUNT
) -> SeamArtifactReport | None:
    """Samples up to `sample_count` frames roughly evenly spaced across `frames` (excluding

    frame 0 itself) and diffs each against frame 0, looking for the real, evidenced seam
    signature (`ChangedRegionShape.has_asymmetric_edge`) on the SINGLE LARGEST changed-region
    component of each comparison. `None` when `frames` has fewer than 2 entries (nothing to
    diff).

    Deliberately only the largest component, not every component above `_MIN_COMPONENT_AREA_PX`
    -- real validation against the Phase 8 evidence videos (this module's docstring) found that
    scanning every component produces a false positive on the real "duplicate silhouette"
    (ghosting) defect video: a small, unrelated sliver component (e.g. a thin text/UI-adjacent
    diff fragment) can pass the asymmetric-edge check by chance even though the video's own
    dominant animated-object component correctly does not. The real seam signature was always
    carried by the single largest changed-region component in every validated case; restricting
    to it removes the false positive without weakening real detection.
    """
    if len(frames) < 2:
        return None
    n = len(frames)
    step = max(1, n // sample_count)
    sample_indices = sorted({i for i in range(step, n, step)} | {n - 1})

    worst: ChangedRegionShape | None = None
    seam_suspected = False
    for i in sample_indices:
        shapes = detect_changed_region_shapes(frames[0], frames[i])
        if not shapes:
            continue
        largest = shapes[0]
        if worst is None or _asymmetry_score(largest) > _asymmetry_score(worst):
            worst = largest
        if largest.has_asymmetric_edge:
            seam_suspected = True

    return SeamArtifactReport(
        frames_compared=len(sample_indices), seam_suspected=seam_suspected, worst_component=worst
    )


def _asymmetry_score(shape: ChangedRegionShape) -> float:
    """A single ranking number for "how close to the seam signature is this component" --

    used only to pick a representative `worst_component` for reporting, not to decide
    `seam_suspected` itself (that stays the evidenced threshold check).
    """
    horizontal = max(shape.left_edge_fraction, shape.right_edge_fraction) - min(
        shape.left_edge_fraction, shape.right_edge_fraction
    )
    vertical = max(shape.top_edge_fraction, shape.bottom_edge_fraction) - min(
        shape.top_edge_fraction, shape.bottom_edge_fraction
    )
    return max(horizontal, vertical)
