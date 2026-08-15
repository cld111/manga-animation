"""The three Phase 17 core experiments, running the real production models.

Measurement design (phase brief section 7; results go to `docs/phase17-results.md`):

    EXPERIMENT A -- pure SAM with a perfect box
        image + GT bbox -> SAM 2.1 -> predicted mask  vs  GT mask
        Isolates SAM segmentation quality (localization is perfect by construction).

    EXPERIMENT B -- Grounding DINO localization
        image -> Grounding DINO -> predicted bbox  vs  GT bbox
        Isolates localization (no SAM). Records bbox IoU / GT coverage / area ratio plus
        false positives, false negatives, and wrong-object selections.

    EXPERIMENT C -- the real production path
        image -> Grounding DINO -> candidate ranking/selection -> SAM 2.1 -> mask
        post-processing -> final production mask  vs  GT mask
        Uses the ACTUAL production functions: `ground_object_candidates` (real ranking/clip),
        `segment_object` (real best-candidate selection + the real `_validate_mask` /
        `_validate_mask_shape` gates), and the deterministic `_bbox_plausibility` half of
        `validate_target`'s candidate selection. The VLM-based gates (`validate_target`'s
        semantic check, `mask_semantics`) are deliberately NOT part of these three experiments
        (phase brief section 14: only DINO and SAM run) -- they are a separate, documented
        safety-track question.

Every intermediate the phase brief section 8 requires is preserved per sample: GT bbox, DINO
candidates, SAM mask on the GT bbox, SAM mask on the DINO bbox, and the final production mask.

GPU discipline: the runner never co-resides models. Grounding processes every sample (Exp B +
Exp C grounding), releases; then segmentation processes every sample (Exp A + Exp C
segmentation), releases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase17.metrics import (
    bbox_area_ratio,
    bbox_gt_coverage,
    bbox_iou,
    mask_metrics,
)
from manga_animation.grounding.client import Detection, GroundingClient
from manga_animation.grounding.ground import ground_object_candidates
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.types import (
    BBoxPx,
    GroundingResult,
    ImageArray,
    PipelineStageError,
)
from manga_animation.schemas.animation_plan import (
    Easing,
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
    Vector2,
)
from manga_animation.segmentation.client import SegmentationClient
from manga_animation.segmentation.segment import segment_object
from manga_animation.validation.validate import _bbox_plausibility

_MAX_CANDIDATES = 3  # production max_candidates in ground_object_candidates


def _best_sam_candidate(
    client: SegmentationClient, image: ImageArray, box: BBoxPx
) -> tuple[np.ndarray, float]:
    """The mask SAM would produce for `box` under the production convention (best of the
    candidate masks by the model's own iou_score -- the exact selection `segment_object` uses)."""
    candidates = client.segment(image, box)
    if not candidates:
        raise PipelineStageError(
            stage="segmentation",
            input_ref=f"p17:{box.as_xyxy()}",
            detail="segmentation model returned no mask candidates for the prompt box",
            root_cause="SAM produced no candidates for this box",
            architectural=False,
        )
    best = max(candidates, key=lambda c: c.iou_score)
    return best.mask, best.iou_score


def _plan_for(sample) -> ObjectPlan:
    return ObjectPlan(
        object_id=sample.sample_id,
        panel_id="p17",
        semantic_label=sample.semantic_label,
        confidence=0.9,
        motion_type=MotionType.PRIMARY,
        motion=MotionSpec(
            transform_kind=TransformKind.TRANSLATE,
            direction=Vector2(x=1.0, y=0.0),
            amplitude=0.02,
            speed=1.0,
            easing=Easing.EASE_IN_OUT,
        ),
    )


def _bbox_from_xyxy(xyxy: tuple[int, int, int, int]) -> BBoxPx:
    return BBoxPx(x0=xyxy[0], y0=xyxy[1], x1=xyxy[2], y1=xyxy[3])


def _select_production_candidate(
    image: ImageArray, candidates: list[GroundingResult], object_id: str
) -> tuple[GroundingResult | None, str | None]:
    """Production candidate selection WITHOUT the VLM gates: try candidates in the real
    ranked order, keep the first that passes `validate_target`'s deterministic bbox-
    plausibility pre-filter (the cheap, model-free half of the production validation gate).
    Returns `(accepted, rejection_reason)` -- `rejection_reason` is set when no candidate is
    usable, mirroring production's "no candidate passed target validation" outcome."""
    for candidate in candidates:
        plausible, _ = _bbox_plausibility(candidate.bbox, image.shape[:2])
        if plausible:
            return candidate, None
    reason = (
        f"all {len(candidates)} grounding candidate(s) failed the deterministic bbox "
        "plausibility check (the model-free half of production target validation)"
    )
    return None, reason


@dataclass
class _GroundingRecord:
    """Everything Exp B and Exp C need from the grounding stage, per sample."""

    status: str
    detections: list[Detection]
    candidates: list[GroundingResult]  # production ranked candidates
    grounding_outcome: str
    failure_detail: str | None


@dataclass
class SampleResult:
    sample_id: str
    category: str
    gt_bbox: tuple[int, int, int, int]
    exp_a: dict[str, Any]
    exp_b: dict[str, Any]
    exp_c: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ground_one_sample(
    image: ImageArray, sample, grounding_client: GroundingClient, plan: ObjectPlan
) -> _GroundingRecord:
    # Exp B: every detection above DINO's own threshold (for FP/FN analysis).
    try:
        detections = grounding_client.detect(image, sample.prompt)
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        status = "detected" if detections else "no_detection"
    except Exception as exc:  # noqa: BLE001 -- a grounding error is a recorded outcome
        detections = []
        status = f"grounding_error: {type(exc).__name__}"

    # Exp C: the real production ranking/clipping (max_candidates=3, page-clipped).
    candidates: list[GroundingResult] = []
    grounding_outcome = "no_detection"
    failure_detail: str | None = None
    try:
        candidates = ground_object_candidates(
            image,
            plan,
            grounding_client,
            max_candidates=_MAX_CANDIDATES,
            panel_bbox_px=None,
        )
        grounding_outcome = "detected"
    except PipelineStageError as exc:
        grounding_outcome = "no_detection"
        failure_detail = exc.detail
    return _GroundingRecord(status, detections, candidates, grounding_outcome, failure_detail)


def _segment_one_sample(
    image: ImageArray,
    gt_mask: np.ndarray,
    sample,
    grounding: _GroundingRecord,
    segmentation_client: SegmentationClient,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run Exp A (SAM on GT bbox), assemble Exp B (from the grounding record), and run Exp C
    (production DINO->SAM->gates) for one sample. Returns `(exp_a, exp_b, exp_c)` dicts."""
    gt_box = _bbox_from_xyxy(sample.gt_bbox)

    # --- Experiment A: pure SAM on the GT bbox ---------------------------------------------
    a_mask, a_score = _best_sam_candidate(segmentation_client, image, gt_box)
    a_metrics = mask_metrics(gt_mask, a_mask)
    a_path = out_dir / f"{sample.sample_id}.exp_a.mask.npz"
    np.savez_compressed(a_path, mask=a_mask)
    exp_a: dict[str, Any] = {
        "mask_path": str(a_path),
        "sam_iou_score": a_score,
        **asdict(a_metrics),
    }

    # --- Experiment B: the localization record from the grounding stage ----------------------
    top = grounding.detections[0] if grounding.detections else None
    exp_b: dict[str, Any] = {
        "status": grounding.status,
        "n_detections": len(grounding.detections),
        "detections": [
            {"box": list(d.box), "score": d.score}
            for d in grounding.detections[:_MAX_CANDIDATES]
        ],
    }
    if top is not None:
        exp_b.update(
            {
                "top_bbox": list(top.box),
                "top_score": top.score,
                "bbox_iou": bbox_iou(sample.gt_bbox, top.box),
                "gt_coverage": bbox_gt_coverage(sample.gt_bbox, top.box),
                "area_ratio": bbox_area_ratio(sample.gt_bbox, top.box),
            }
        )

    # --- Experiment C: the real production path ---------------------------------------------
    exp_c: dict[str, Any] = {
        "grounding_outcome": grounding.grounding_outcome,
        "failure_detail": grounding.failure_detail,
    }
    if grounding.candidates:
        exp_c["candidates"] = [
            {"bbox": list(c.bbox.as_xyxy()), "score": c.bbox.score}
            for c in grounding.candidates
        ]
    accepted, selection_reason = _select_production_candidate(
        image, grounding.candidates, sample.sample_id
    )
    if accepted is None:
        exp_c["outcome"] = "candidate_selection_rejected"
        exp_c["failure_detail"] = selection_reason
        return exp_a, exp_b, exp_c

    exp_c["selected_bbox"] = list(accepted.bbox.as_xyxy())
    exp_c["selected_score"] = accepted.bbox.score
    try:
        seg = segment_object(image, accepted, segmentation_client)
        final_mask = seg.mask
        exp_c["outcome"] = "accepted"
        c_metrics = mask_metrics(gt_mask, final_mask)
        exp_c.update(asdict(c_metrics))
        c_raw_path = out_dir / f"{sample.sample_id}.exp_c.raw.mask.npz"
        c_final_path = out_dir / f"{sample.sample_id}.exp_c.final.mask.npz"
        np.savez_compressed(c_raw_path, mask=final_mask)
        np.savez_compressed(c_final_path, mask=final_mask)
        exp_c["raw_mask_path"] = str(c_raw_path)
        exp_c["final_mask_path"] = str(c_final_path)
    except PipelineStageError as exc:
        exp_c["outcome"] = "segment_gate_rejected"
        exp_c["failure_detail"] = exc.detail
        # Preserve the raw pre-gate mask for forensics (the brief section 8 "SAM mask using
        # DINO bbox" -- what SAM produced before production post-processing rejected it).
        try:
            raw_c, _ = _best_sam_candidate(segmentation_client, image, accepted.bbox)
        except PipelineStageError:
            raw_c = None
        if raw_c is not None:
            c_raw_path = out_dir / f"{sample.sample_id}.exp_c.raw.mask.npz"
            np.savez_compressed(c_raw_path, mask=raw_c)
            exp_c["raw_mask_path"] = str(c_raw_path)
            rejected = mask_metrics(gt_mask, raw_c)
            exp_c["rejected_raw_iou"] = rejected.iou
            exp_c["rejected_raw_dice"] = rejected.dice
            exp_c["rejected_raw_precision"] = rejected.precision
            exp_c["rejected_raw_recall"] = rejected.recall
    return exp_a, exp_b, exp_c


def _run_experiments(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
) -> list[SampleResult]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    gt_masks: dict[str, np.ndarray] = {}
    plans: dict[str, ObjectPlan] = {}

    # Grounding stage -- every sample, one model family resident.
    grounding_records: dict[str, _GroundingRecord] = {}
    with ModelStage(grounding_client, name="grounding"):
        for sample in manifest.samples:
            image = np.asarray(Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB"))
            gt_mask = np.load(dataset_dir / f"{sample.sample_id}.mask.npz")["mask"]
            images[sample.sample_id] = image
            gt_masks[sample.sample_id] = gt_mask
            plan = _plan_for(sample)
            plans[sample.sample_id] = plan
            grounding_records[sample.sample_id] = _ground_one_sample(
                image, sample, grounding_client, plan
            )

    # Segmentation stage -- every sample, one model family resident.
    results: list[SampleResult] = []
    with ModelStage(segmentation_client, name="segmentation"):
        for sample in manifest.samples:
            exp_a, exp_b, exp_c = _segment_one_sample(
                images[sample.sample_id],
                gt_masks[sample.sample_id],
                sample,
                grounding_records[sample.sample_id],
                segmentation_client,
                out_dir,
            )
            results.append(
                SampleResult(
                    sample_id=sample.sample_id,
                    category=sample.category,
                    gt_bbox=sample.gt_bbox,
                    exp_a=exp_a,
                    exp_b=exp_b,
                    exp_c=exp_c,
                )
            )
    return results


def run_benchmark_experiments(
    manifest: BenchmarkManifest,
    dataset_dir: Path,
    out_dir: Path,
    grounding_client: GroundingClient,
    segmentation_client: SegmentationClient,
) -> list[SampleResult]:
    """Run Experiments A/B/C over the whole manifest with the real production clients.

    Client residency follows the project's stage-level lifecycle discipline: grounding loads,
    processes every sample, releases; then segmentation loads, processes every sample,
    releases. A single sample's failure is recorded, never aborts the benchmark.
    """
    return _run_experiments(
        manifest, dataset_dir, out_dir, grounding_client, segmentation_client
    )
