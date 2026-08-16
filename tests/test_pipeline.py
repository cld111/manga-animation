"""Behavioral tests for src/manga_animation/pipeline/orchestrator.py (Phase 18.3 flow).

Exercises the real stage-wiring code with fake model clients (no torch/transformers/GPU
needed) plus the REAL cv2/animation/compositing code and (when available) the REAL ffmpeg
encode. The Phase 18.3 architecture under test: grounding (DINO) -> segmentation (SAM) ->
the pipeline's single VLM stage (object description: one image + ALL its candidate bboxes in
one call) -> deterministic animation planning -> animation (SAM mask + description motion) ->
reconstruction -> compositing -> rendering.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from manga_animation.core.config import PipelineConfig, load_config
from manga_animation.grounding.client import Detection
from manga_animation.pipeline.orchestrator import (
    DEFAULT_ANIMATION_LABELS,
    PipelineRunResult,
    _candidate_source,
    run_pipeline,
)
from manga_animation.pipeline.panels import run_page_panels
from manga_animation.pipeline.types import PipelineStageError
from manga_animation.schemas.animation_plan import MotionType
from manga_animation.segmentation.client import MaskCandidate

pytestmark = pytest.mark.filterwarnings("ignore")


def _resolve_test_ffmpeg() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        return None


requires_ffmpeg = pytest.mark.skipif(
    _resolve_test_ffmpeg() is None,
    reason="no ffmpeg binary resolvable (neither system nor imageio-ffmpeg)",
)


# --- fakes --------------------------------------------------------------------------------


_OBJECT_DESCRIPTION_PROMPT_MARKER = "proposed animation candidate"
"""Distinctive substring of object_description/prompt.py's prompts (both the single-candidate
and the batch variant), so fakes can dispatch on it."""


def _fake_batch_response(
    n_boxes: int,
    *,
    accepted: set[int] | None = None,
    assessment: str = "pass",
    matches: bool = True,
    animatable: bool = True,
    motion_kind: str = "sway",
    direction: str | None = None,
    confidence: float = 0.9,
) -> str:
    """A schema-valid batch answer: one description per box_index."""
    accepted = accepted if accepted is not None else set(range(n_boxes))
    entries = []
    for i in range(n_boxes):
        ok = i in accepted
        entries.append(
            {
                "box_index": i,
                "bbox_assessment": assessment if ok else "reject",
                "object_identity": "fake_object",
                "matches_semantic_label": matches if ok else False,
                "animatable": animatable if ok else False,
                "movable_parts": ["fake movable part"] if ok else [],
                "static_parts": ["fake static part"] if ok else [],
                "motion_kind": motion_kind if (ok and animatable) else None,
                "direction": direction if (ok and motion_kind == "drift") else None,
                "amplitude_band": "moderate" if ok else None,
                "speed_band": "slow" if ok else None,
                "pivot_hint": "center" if ok else None,
                "constraints": ["fake constraint"] if ok else [],
                "neighbor_conflicts": [] if ok else ["fake conflict"],
                "confidence": confidence if ok else 0.1,
                "reason": "fake object description response",
            }
        )
    return json.dumps(entries)


class BatchVLMClient:
    """Answers ONLY the object-description batch prompt (the pipeline's single VLM stage)
    with one description per candidate box. Records the number of calls and the images seen."""

    def __init__(
        self,
        *,
        accepted: set[int] | None = None,
        assessment: str = "pass",
        matches: bool = True,
        animatable: bool = True,
        motion_kind: str = "sway",
        direction: str | None = None,
        confidence: float = 0.9,
        unparseable: bool = False,
    ):
        self._accepted = accepted
        self._assessment = assessment
        self._matches = matches
        self._animatable = animatable
        self._motion_kind = motion_kind
        self._direction = direction
        self._confidence = confidence
        self._unparseable = unparseable
        self.call_count = 0
        self.seen_images: list[tuple[int, int]] = []
        self.seen_prompts: list[str] = []

    def generate(self, image, prompt: str) -> str:
        if _OBJECT_DESCRIPTION_PROMPT_MARKER not in prompt:
            raise AssertionError(
                "the Phase 18.3 pipeline must call the VLM ONLY at the object-description "
                "stage -- unexpected prompt: " + prompt[:80]
            )
        self.call_count += 1
        self.seen_images.append((image.width, image.height))
        self.seen_prompts.append(prompt)
        if self._unparseable:
            return "not json at all"
        n_boxes = prompt.count("<|box_start|>")
        return _fake_batch_response(
            n_boxes,
            accepted=self._accepted,
            assessment=self._assessment,
            matches=self._matches,
            animatable=self._animatable,
            motion_kind=self._motion_kind,
            direction=self._direction,
            confidence=self._confidence,
        )

    def unload(self) -> None:
        pass


class RecordingVLMClient(BatchVLMClient):
    """Like `BatchVLMClient` but lets a test override the batch answer completely."""

    def __init__(self, responder):
        self._responder = responder
        self.call_count = 0

    def generate(self, image, prompt: str) -> str:
        self.call_count += 1
        return self._responder(image, prompt)

    def unload(self) -> None:
        pass


class FailingGroundingClient:
    model_id = "fake-grounding-dino"

    def load(self) -> None:
        pass

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return []  # nothing above threshold

    def unload(self) -> None:
        pass


class FakeGroundingClient:
    """Returns one or more ranked candidate boxes for every prompt."""

    model_id = "fake-grounding-dino"

    def __init__(
        self,
        box: tuple[int, int, int, int] = (10, 10, 60, 90),
        *,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ):
        self.boxes = boxes if boxes is not None else [box]
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        return [
            Detection(label="banner", score=0.9 - 0.1 * i, box=b) for i, b in enumerate(self.boxes)
        ]

    def unload(self) -> None:
        self.unloaded = True


class MultiObjectFakeGroundingClient:
    """Returns a different box per semantic label, matched by substring against the grounding
    prompt (`grounding/ground.py::_prompt_from_label`). A label with no entry detects nothing."""

    model_id = "fake-grounding-dino"

    def __init__(self, boxes_by_label: dict[str, tuple[int, int, int, int]]):
        self._boxes_by_label = {
            label.replace("_", " "): box for label, box in boxes_by_label.items()
        }
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        for label, box in self._boxes_by_label.items():
            if label in text_prompt:
                return [Detection(label=label, score=0.9, box=box)]
        return []

    def unload(self) -> None:
        self.unloaded = True


class RecordingGroundingClient:
    """Like `MultiObjectFakeGroundingClient`, but records every `detect()` call."""

    model_id = "fake-grounding-dino"

    def __init__(self, boxes_by_label: dict[str, tuple[int, int, int, int]]):
        self._boxes_by_label = {
            label.replace("_", " "): box for label, box in boxes_by_label.items()
        }
        self.calls: list[tuple[tuple[int, ...], str]] = []
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.calls.append((image.shape, text_prompt))
        for label, box in self._boxes_by_label.items():
            if label in text_prompt:
                return [Detection(label=label, score=0.9, box=box)]
        return []

    def unload(self) -> None:
        self.unloaded = True


def _region_mask(h: int, w: int, box) -> np.ndarray:
    """A full-image-shape uint8 0/255 mask: a diamond inscribed in `box` (a shape a real SAM
    output could plausibly have -- `segmentation/segment.py::_validate_mask_shape` rejects
    solid rectangles)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = (box.x0 + box.x1 - 1) / 2.0, (box.y0 + box.y1 - 1) / 2.0
    ax, ay = max((box.x1 - box.x0) / 2.0, 1e-9), max((box.y1 - box.y0) / 2.0, 1e-9)
    yy, xx = np.mgrid[box.y0 : box.y1, box.x0 : box.x1]
    local = (np.abs(xx - cx) / ax + np.abs(yy - cy) / ay) <= 1.0
    mask[box.y0 : box.y1, box.x0 : box.x1] = local.astype(np.uint8) * 255
    return mask


class FakeSegmentationClient:
    model_id = "fake-sam2.1"

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[0], image.shape[1]
        return [MaskCandidate(mask=_region_mask(h, w, box), iou_score=0.9)]

    def unload(self) -> None:
        pass


class FakeReconstructionClient:
    def load(self) -> None:
        pass

    def inpaint(self, image, hole_mask):
        return image

    def unload(self) -> None:
        pass


@pytest.fixture
def page_path(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    img = Image.new("RGB", (120, 160), (240, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 60, 90], fill=(180, 30, 30))
    img.save(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(duration_s=0.5, fps=8)


@pytest.fixture
def two_panel_page_path(tmp_path: Path) -> Path:
    """A real, detectably-two-panel page (construction style from tests/test_panels.py)."""
    def noise_block(h: int, w: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = noise_block(300, 300, seed=21)
    page[500:900, 0:300] = noise_block(400, 300, seed=22)
    path = tmp_path / "two_panel_page.png"
    Image.fromarray(page).save(path)
    return path


# --- end-to-end pipeline -------------------------------------------------------------------


@requires_ffmpeg
def test_run_pipeline_end_to_end_produces_a_playable_video(page_path: Path, config, tmp_path: Path):
    """The whole Phase 18.3 flow on one image: DINO box -> SAM mask -> ONE Qwen call (full
    image + the box) -> accepted description -> animation with the SAM mask -> real MP4."""
    result = run_pipeline(
        page_path,
        config,
        labels=["character"],
        vlm_client=BatchVLMClient(motion_kind="sway"),
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )

    assert isinstance(result, PipelineRunResult)
    assert result.render.output_path.exists()
    assert result.render.frame_count == config.fps * config.duration_s
    assert result.primary_object.motion_type == MotionType.PRIMARY
    assert result.primary_object.semantic_label == "character"
    assert result.primary_object.motion is not None
    assert result.primary_object.motion.transform_kind.value == "mesh_warp"  # sway mapping
    assert result.object_description is not None
    assert result.object_description.accepted is True
    assert result.segmentation.mask.shape[:2] == (160, 120)  # full-source-image mask


@requires_ffmpeg
def test_run_pipeline_makes_exactly_one_vlm_call_for_all_candidates(
    page_path: Path, config, tmp_path: Path
):
    """Phase 18.3 input contract: the VLM sees the image and ALL candidate bboxes in ONE
    call -- never one call per candidate."""
    vlm = BatchVLMClient(motion_kind="sway")
    result = run_pipeline(
        page_path,
        config,
        labels=["character", "flag_banner", "weapon"],
        vlm_client=vlm,
        grounding_client=FakeGroundingClient(boxes=[(10, 10, 60, 90), (30, 40, 80, 120)]),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert result.render.output_path.exists()
    assert vlm.call_count == 1
    # The prompt lists every candidate box with its coordinates (the "[i] image 0 ..." lines).
    listed_boxes = sum(1 for line in vlm.seen_prompts[0].splitlines() if line.startswith("["))
    assert listed_boxes == len(result.dropped_objects) + len(result.secondary_objects) + 1


class CountingSegmentationClient:
    """Fake SAM that counts every `segment()` call and remembers the boxes it was asked to
    segment -- used to prove the Phase 18.4 contract: SAM runs ONLY on accepted bboxes."""

    model_id = "fake-sam2.1"

    def __init__(self):
        self.calls: list[tuple[int, int, int, int]] = []

    def load(self) -> None:
        pass

    def segment(self, image, box) -> list[MaskCandidate]:
        self.calls.append((box.x0, box.y0, box.x1, box.y1))
        mask = _region_mask(image.shape[0], image.shape[1], box)
        return [MaskCandidate(mask=mask, iou_score=0.9)]

    def unload(self) -> None:
        pass


@requires_ffmpeg
def test_run_pipeline_sam_segments_only_accepted_bboxes(
    page_path: Path, config, tmp_path: Path
):
    """Phase 18.4 ordering (DINO -> Qwen -> SAM): of two grounded bboxes, only the one with
    an accepted action description is ever handed to SAM."""
    sam = CountingSegmentationClient()
    result = run_pipeline(
        page_path,
        config,
        labels=["character", "flag_banner"],
        vlm_client=RecordingVLMClient(
            lambda image, prompt: _fake_batch_response(
                sum(1 for line in prompt.splitlines() if line.startswith("[")),
                accepted={0},
                motion_kind="sway",
            )
        ),
        grounding_client=FakeGroundingClient(boxes=[(10, 10, 60, 90), (30, 40, 80, 120)]),
        segmentation_client=sam,
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert result.render.output_path.exists()
    assert result.primary_object.semantic_label == "character"
    # Exactly one SAM call, for the accepted bbox (the first grounded one).
    assert sam.calls == [(10, 10, 60, 90)]


@requires_ffmpeg
def test_run_pipeline_rejected_candidate_never_reaches_sam_or_render(
    page_path: Path, config, tmp_path: Path
):
    """A VLM-rejected bbox is never segmented and never animated: only the accepted
    candidate's mask participates in the final plan."""
    sam = CountingSegmentationClient()
    result = run_pipeline(
        page_path,
        config,
        labels=["character", "flag_banner"],
        vlm_client=RecordingVLMClient(
            lambda image, prompt: _fake_batch_response(
                sum(1 for line in prompt.splitlines() if line.startswith("[")),
                accepted={1},
                motion_kind="flow",
            )
        ),
        grounding_client=FakeGroundingClient(boxes=[(10, 10, 60, 90), (30, 40, 80, 120)]),
        segmentation_client=sam,
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert sam.calls == [(30, 40, 80, 120)]  # only the accepted bbox was segmented
    assert result.primary_object.semantic_label == "character"
    # 2 labels x 2 boxes = 4 candidates in the batch; 1 accepted, 3 rejected at description.
    assert len(result.dropped_objects) == 3
    assert all(d.failing_stage == "object_description" for d in result.dropped_objects)


@requires_ffmpeg
def test_run_pipeline_batches_multiple_candidates_of_one_image_in_one_call(
    page_path: Path, config, tmp_path: Path
):
    """Two candidates on the same image -> exactly one VLM call covering both."""
    vlm = RecordingVLMClient(
        lambda image, prompt: _fake_batch_response(
            sum(1 for line in prompt.splitlines() if line.startswith("[")),
            accepted={0},
            motion_kind="sway",
        )
    )
    result = run_pipeline(
        page_path,
        config,
        labels=["character", "flag_banner"],
        vlm_client=vlm,
        grounding_client=FakeGroundingClient(boxes=[(10, 10, 60, 90), (30, 40, 80, 120)]),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert vlm.call_count == 1
    assert result.render.output_path.exists()
    assert result.primary_object.semantic_label == "character"


@requires_ffmpeg
def test_run_pipeline_vlm_receives_the_full_image_not_a_crop(
    page_path: Path, config, tmp_path: Path
):
    """The task brief's critical input contract at the pipeline level: the VLM call receives
    the FULL page (resized to the analysis resolution), never a crop of the candidate."""
    vlm = BatchVLMClient(motion_kind="sway")
    run_pipeline(
        page_path,
        config,
        labels=["character"],
        vlm_client=vlm,
        grounding_client=FakeGroundingClient(),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    seen_w, seen_h = vlm.seen_images[0]
    # The bbox is (10,10,60,90) on a 120x160 page; the VLM image is the whole page after the
    # 28px patch-grid rounding (120x160 -> 112x168) -- never a crop of the candidate.
    assert seen_w >= 112 and seen_h >= 160
    assert "Pixel coordinates" in vlm.seen_prompts[0]


def test_run_pipeline_fails_closed_when_no_candidate_is_accepted(
    page_path: Path, config, tmp_path: Path
):
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            labels=["character"],
            vlm_client=BatchVLMClient(accepted=set()),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "object_description"
    assert "no grounded candidate was accepted" in excinfo.value.detail
    assert not (tmp_path / "out" / "output.mp4").exists()


def test_run_pipeline_fails_closed_on_unparseable_vlm_output(
    page_path: Path, config, tmp_path: Path
):
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            labels=["character"],
            vlm_client=BatchVLMClient(unparseable=True),
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "object_description"


def test_run_pipeline_drops_an_identity_conflict_candidate(page_path: Path, config, tmp_path: Path):
    """The deterministic identity backstop: a pass+matches+animatable read naming a speech
    bubble is rejected even though every soft signal says accept."""
    vlm = RecordingVLMClient(
        lambda image, prompt: json.dumps(
            [
                {
                    "box_index": 0,
                    "bbox_assessment": "pass",
                    "object_identity": "speech_bubble",
                    "matches_semantic_label": True,
                    "animatable": True,
                    "movable_parts": ["tail"],
                    "static_parts": [],
                    "motion_kind": "sway",
                    "direction": None,
                    "amplitude_band": "subtle",
                    "speed_band": "slow",
                    "pivot_hint": "center",
                    "constraints": [],
                    "neighbor_conflicts": [],
                    "confidence": 0.95,
                    "reason": "fake",
                }
            ]
        )
    )
    with pytest.raises(PipelineStageError) as excinfo:
        run_pipeline(
            page_path,
            config,
            labels=["character"],
            vlm_client=vlm,
            grounding_client=FakeGroundingClient(),
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    assert excinfo.value.stage == "object_description"


@requires_ffmpeg
def test_run_pipeline_multi_object_uses_ranked_acceptance(page_path: Path, config, tmp_path: Path):
    """Two accepted candidates: the higher-confidence one becomes PRIMARY, the other SECONDARY,
    and both animate."""
    vlm = RecordingVLMClient(
        lambda image, prompt: _fake_batch_response(
            sum(1 for line in prompt.splitlines() if line.startswith("[")),
            accepted={0, 1},
            motion_kind="sway",
            confidence=0.95,
        )
    )
    result = run_pipeline(
        page_path,
        config,
        labels=["character", "flag_banner"],
        vlm_client=vlm,
        grounding_client=FakeGroundingClient(boxes=[(10, 10, 60, 90), (30, 40, 80, 120)]),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert result.render.output_path.exists()
    assert len(result.secondary_objects) == 1
    assert result.primary_object.motion_type == MotionType.PRIMARY
    assert result.secondary_objects[0].object_plan.motion_type == MotionType.SECONDARY
    assert result.secondary_objects[0].object_plan.motion is not None


@requires_ffmpeg
def test_run_pipeline_grounding_prompt_uses_the_label(
    page_path: Path, config, tmp_path: Path
):
    grounding = RecordingGroundingClient({"character": (10, 10, 60, 90)})
    run_pipeline(
        page_path,
        config,
        labels=["character"],
        vlm_client=BatchVLMClient(motion_kind="sway"),
        grounding_client=grounding,
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "out",
    )
    assert len(grounding.calls) == 1
    assert "character" in grounding.calls[0][1]


def test_run_pipeline_default_labels_are_used(page_path: Path, config, tmp_path: Path):
    grounding = RecordingGroundingClient({})
    with pytest.raises(PipelineStageError):
        run_pipeline(
            page_path,
            config,
            vlm_client=BatchVLMClient(),
            grounding_client=grounding,
            segmentation_client=FakeSegmentationClient(),
            reconstruction_client=FakeReconstructionClient(),
            out_dir=tmp_path / "out",
        )
    grounded_labels = {p.split(".")[0].strip().replace(" ", "_") for _, p in grounding.calls}
    assert grounded_labels == set(DEFAULT_ANIMATION_LABELS)


@requires_ffmpeg
def test_run_page_panels_processes_each_panel_with_one_vlm_residency(
    two_panel_page_path: Path, config, tmp_path: Path
):
    vlm = BatchVLMClient(motion_kind="sway")
    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=vlm,
        grounding_client=FakeGroundingClient(box=(20, 20, 90, 140)),
        segmentation_client=FakeSegmentationClient(),
        reconstruction_client=FakeReconstructionClient(),
        out_dir=tmp_path / "videos",
    )
    # Every panel either rendered or was rejected at a safe gate; the VLM was called once per
    # eligible panel (the single object-description stage), never more.
    for panel in result.panels:
        assert panel.status in ("PASS", "REJECTED")
    assert vlm.call_count <= len(result.panels)


# --- config / candidate resolution ---------------------------------------------------------


def test_candidate_source_resolves_from_manifest():
    config = PipelineConfig(model_variants={"vlm": "qwen2.5-vl-7b-instruct"})
    assert "qwen" in _candidate_source("vlm", config).lower()


def test_config_defaults_have_the_phase_18_3_flag():
    config = load_config("default")
    assert config.enable_object_description_validation is True
