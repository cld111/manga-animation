"""Targeted tests for Phase 14's stage-level model lifecycle.

Two layers are protected here, without duplicating the existing behavioral coverage in
tests/test_pipeline.py:

1. `ModelStage` (src/manga_animation/pipeline/lifecycle.py) -- the context manager that owns
   one model client's residency. The GPU evidence this phase records (docs/phase14-results.md)
   showed the old unload path (`set model = None; empty_cache()` with no `gc.collect()`) left
   Qwen's ~16 GiB resident until an opportunistic cyclic-GC, racing the next load into a CUDA
   OOM. `ModelStage` must release deterministically on normal exit AND on exception.
2. Stage-level `run_page_panels` -- each model client is loaded exactly once per page (per
   stage), not once per panel, and a panel failing in an early stage cannot poison later
   panels or prevent the stage's cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_animation.core.config import PipelineConfig
from manga_animation.grounding.client import Detection
from manga_animation.pipeline.lifecycle import ModelStage
from manga_animation.pipeline.orchestrator import DEFAULT_ANIMATION_LABELS
from manga_animation.pipeline.panels import run_page_panels, run_pages
from manga_animation.segmentation.client import MaskCandidate

pytestmark = pytest.mark.filterwarnings("ignore")

# Distinctive substrings of the real stage prompts, so fakes can tell analysis from
# validation from mask_semantics without threading extra parameters (same convention as
# tests/test_pipeline.py).
_VALIDATION_PROMPT_MARKER = "Does the image above show"
_MASK_SEMANTICS_PROMPT_MARKER = "Does the bright region show"
_OBJECT_DESCRIPTION_PROMPT_MARKER = "proposed animation candidate"


def _fake_object_description_response() -> str:
    return json.dumps(
        {
            "bbox_assessment": "pass",
            "object_identity": "fake_object",
            "matches_semantic_label": True,
            "animatable": True,
            "movable_parts": ["fake movable part"],
            "static_parts": ["fake static part"],
            "motion_kind": "sway",
            "direction": None,
            "amplitude_band": "moderate",
            "speed_band": "slow",
            "pivot_hint": "center",
            "constraints": ["fake constraint"],
            "neighbor_conflicts": [],
            "confidence": 0.9,
            "reason": "fake object description response",
        }
    )


# --- ModelStage unit tests -------------------------------------------------------------------


class TrackingClient:
    """A client with visible load/unload counters (DINO/SAM/LaMa shape: explicit load())."""

    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1


class LazyVLMClient:
    """The VLM's real shape: no `load()` (it lazy-loads inside generate()); only unload()."""

    def __init__(self):
        self.unload_calls = 0

    def generate(self, image, prompt: str) -> str:
        return "{}"

    def unload(self) -> None:
        self.unload_calls += 1


def test_model_stage_loads_on_entry_and_unloads_on_normal_exit():
    client = TrackingClient()
    with ModelStage(client, name="stage"):
        assert client.load_calls == 1
        assert client.unload_calls == 0
    assert client.unload_calls == 1


def test_model_stage_still_unloads_when_the_stage_body_raises():
    client = TrackingClient()
    with pytest.raises(ValueError, match="boom"):
        with ModelStage(client, name="stage"):
            raise ValueError("boom")
    # The exception propagated, but the model was still released -- a failed panel must not
    # leave its model resident to poison the next panel (Phase 14 acceptance criterion 4).
    assert client.unload_calls == 1


def test_model_stage_handles_a_client_with_no_load_method():
    client = LazyVLMClient()
    with ModelStage(client, name="analysis"):
        assert client.unload_calls == 0  # lazy VLM is not force-loaded by the stage
    assert client.unload_calls == 1


def test_model_stage_handles_a_client_with_neither_load_nor_unload():
    with ModelStage(object(), name="trivial"):  # must not raise on enter/exit
        pass


def test_model_stage_rejects_reentrant_entry():
    stage = ModelStage(TrackingClient(), name="stage")
    stage.__enter__()
    try:
        with pytest.raises(RuntimeError):
            stage.__enter__()
    finally:
        stage.__exit__(None, None, None)


def test_model_stage_enter_load_failure_does_not_poison_the_stage_object():
    """Phase 15 adversarial review (HIGH/MEDIUM): Python does not call `__exit__` when
    `__enter__` raises, so a failed client.load() must still reset the re-entrancy guard and
    attempt the deterministic release. Otherwise the stage object stays permanently
    'active' and any partial CUDA blocks a failed load allocated stay resident."""
    client = TrackingClient()

    def explode() -> None:
        raise RuntimeError("simulated CUDA OOM during load")

    client.load = explode  # type: ignore[method-assign]
    stage = ModelStage(client, name="stage")
    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        stage.__enter__()
    # The guard was reset, so a later, successful entry is possible instead of a permanent
    # "entered while already active" poisoning.
    stage.auto_load = False
    stage.__enter__()
    assert stage._active
    stage.__exit__(None, None, None)
    assert not stage._active


def test_model_stage_unload_failure_does_not_mask_the_stage_exception():
    """Phase 15 adversarial review (HIGH): a raising client.unload() must not replace the
    stage body's own exception (the root cause would be lost)."""
    class ExplodingUnload(TrackingClient):
        def unload(self) -> None:
            raise RuntimeError("simulated empty_cache failure")

    with pytest.raises(ValueError, match="body boom"):
        with ModelStage(ExplodingUnload(), name="stage"):
            raise ValueError("body boom")


def test_model_stage_unload_failure_still_fails_a_successful_stage():
    """Phase 15 adversarial review (HIGH): when the stage body succeeds but cleanup raises,
    the failure is still surfaced (fail-closed) rather than silently swallowed."""
    class ExplodingUnload(TrackingClient):
        def unload(self) -> None:
            raise RuntimeError("simulated empty_cache failure")

    with pytest.raises(RuntimeError, match="simulated empty_cache failure"):
        with ModelStage(ExplodingUnload(), name="stage"):
            pass


def test_model_stage_unload_failure_still_runs_release_device_memory():
    """Phase 15 adversarial review (HIGH): the deterministic release must run even when the
    client's unload() raises -- a failed unload must not skip the Phase 14 leak protection."""
    from manga_animation.pipeline import lifecycle as lifecycle_module

    class ExplodingUnload(TrackingClient):
        def unload(self) -> None:
            raise RuntimeError("simulated unload failure")

    calls = []

    def fake_release(device: str | None = None) -> None:
        calls.append(device)

    original = lifecycle_module.release_device_memory
    lifecycle_module.release_device_memory = fake_release
    try:
        with pytest.raises(RuntimeError, match="simulated unload failure"):
            with ModelStage(ExplodingUnload(), name="stage"):
                pass
    finally:
        lifecycle_module.release_device_memory = original
    assert calls == [None]  # release ran despite the unload failure


# --- stage-level run_page_panels -------------------------------------------------------------


class CountingGroundingClient:
    model_id = "fake-grounding-dino"

    def __init__(self, fail_first_n: int = 0, raise_on_call: int | None = None):
        self.load_calls = 0
        self.unload_calls = 0
        self.detect_calls = 0
        self.fail_first_n = fail_first_n
        self.raise_on_call = raise_on_call

    def load(self) -> None:
        self.load_calls += 1

    def detect(self, image, text_prompt: str) -> list[Detection]:
        self.detect_calls += 1
        if self.detect_calls == self.raise_on_call:
            raise RuntimeError("fake unexpected GPU error")
        if self.detect_calls <= self.fail_first_n:
            return []
        h, w = image.shape[:2]
        # Well inset from the crop edges and smaller than 35% of the reference region so the
        # transform-aware geometry check (mesh_warp bounds: 2% edge margin, 35% max area)
        # accepts it.
        return [Detection(label="object", score=0.9, box=(w // 4, h // 4, 3 * w // 4, 3 * h // 4))]

    def unload(self) -> None:
        self.unload_calls += 1


class CountingSegmentationClient:
    model_id = "fake-sam2.1"

    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def segment(self, image, box) -> list[MaskCandidate]:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[box.y0 : box.y1, box.x0 : box.x1] = 255
        return [MaskCandidate(mask=mask, iou_score=0.9)]

    def unload(self) -> None:
        self.unload_calls += 1


class CountingReconstructionClient:
    def __init__(self):
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def inpaint(self, image, hole_mask):
        return image

    def unload(self) -> None:
        self.unload_calls += 1


class StageLevelVLMClient:
    """Answers ONLY the object-description batch prompt (the pipeline's single VLM stage in
    the Phase 18.3 architecture) with one pass description per candidate box, and counts
    unloads (its real shape: no load())."""

    def __init__(self):
        self.unload_calls = 0

    def generate(self, image, prompt: str) -> str:
        if _OBJECT_DESCRIPTION_PROMPT_MARKER not in prompt:
            raise AssertionError(
                "the Phase 18.3 pipeline must call the VLM ONLY at the object-description "
                "stage -- unexpected prompt: " + prompt[:80]
            )
        n_boxes = sum(1 for line in prompt.splitlines() if line.startswith("["))
        return json.dumps(
            [
                {
                    "box_index": i,
                    "bbox_assessment": "pass",
                    "object_identity": "character",
                    "matches_semantic_label": True,
                    "animatable": True,
                    "movable_parts": ["hair"],
                    "static_parts": ["face"],
                    "motion_kind": "sway",
                    "direction": None,
                    "amplitude_band": "moderate",
                    "speed_band": "slow",
                    "pivot_hint": "center",
                    "constraints": ["keep the face static"],
                    "neighbor_conflicts": [],
                    "confidence": 0.9,
                    "reason": "fake object description response",
                }
                for i in range(n_boxes)
            ]
        )

    def unload(self) -> None:
        self.unload_calls += 1


def _noise_block(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def two_panel_page_path(tmp_path: Path) -> Path:
    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = _noise_block(300, 300, seed=21)
    page[500:900, 0:300] = _noise_block(400, 300, seed=22)
    path = tmp_path / "two_panel_page.png"
    Image.fromarray(page).save(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(duration_s=0.5, fps=8)


def _requires_ffmpeg():
    import shutil

    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_loads_each_model_once_for_the_whole_page(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """Phase 14 stage-level lifecycle: a model is loaded once for all eligible panels, not
    once per panel. The old per-panel path loaded Grounding DINO / SAM / LaMa once per panel
    and entered the VLM analysis/validation/mask_semantics stages once per panel.
    """
    grounding = CountingGroundingClient()
    segmentation = CountingSegmentationClient()
    reconstruction = CountingReconstructionClient()
    vlm = StageLevelVLMClient()

    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=segmentation,
        reconstruction_client=reconstruction,
        out_dir=tmp_path / "videos",
    )

    assert len(result.panels) == 2
    assert all(panel.status == "PASS" for panel in result.panels)
    # One load/unload per model for the whole page (the stage owns residency).
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1
    assert segmentation.load_calls == 1
    assert segmentation.unload_calls == 1
    assert reconstruction.load_calls == 1
    assert reconstruction.unload_calls == 1
    # VLM: exactly ONE stage in the Phase 18.3 architecture (object description), releasing
    # once -- not per panel and not for any analysis/validation/mask_semantics stage.
    assert vlm.unload_calls == 1


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_early_panel_failure_does_not_poison_later_panels_or_cleanup(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """Phase 14 acceptance criterion 4 (Phase 18.3 flow): a panel whose grounding detects
    nothing for EVERY candidate label must not prevent panel 2 from processing, and the
    stage's model must still be released exactly once.
    """
    n_labels = len(DEFAULT_ANIMATION_LABELS)
    grounding = CountingGroundingClient(fail_first_n=n_labels)
    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=StageLevelVLMClient(),
        grounding_client=grounding,
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )

    assert [panel.status for panel in result.panels] == ["REJECTED", "PASS"]
    assert result.panels[0].failure_stage == "object_description"
    assert grounding.detect_calls == 2 * n_labels  # panel 2 still got grounded for every label
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_object_description_rejection_keeps_panels_rejected(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """When the pipeline's single semantic stage (object description) rejects every
    candidate, each panel must stay REJECTED -- never rendered into a PASS by a later
    stage. This is the fail-closed policy of the Phase 18.3 flow.
    """
    class RejectingVLM(StageLevelVLMClient):
        def generate(self, image, prompt: str) -> str:
            n_boxes = sum(1 for line in prompt.splitlines() if line.startswith("["))
            return json.dumps(
                [
                    {
                        "box_index": i,
                        "bbox_assessment": "ambiguous",
                        "object_identity": "character",
                        "matches_semantic_label": True,
                        "animatable": False,
                        "movable_parts": [],
                        "static_parts": [],
                        "motion_kind": None,
                        "direction": None,
                        "amplitude_band": None,
                        "speed_band": None,
                        "pivot_hint": None,
                        "constraints": [],
                        "neighbor_conflicts": ["two characters in the box"],
                        "confidence": 0.7,
                        "reason": "fake rejection",
                    }
                    for i in range(n_boxes)
                ]
            )

    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=RejectingVLM(),
        grounding_client=CountingGroundingClient(),
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )

    assert [panel.status for panel in result.panels] == ["REJECTED", "REJECTED"]
    assert all(panel.failure_stage == "object_description" for panel in result.panels)
    manifest = json.loads(result.manifest_path.read_text())
    for item in manifest["panels"]:
        assert item["status"] == "REJECTED"
        assert item["output_video"] is None


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_page_panels_unexpected_stage_exception_isolates_to_that_panel(
    two_panel_page_path: Path, config: PipelineConfig, tmp_path: Path
):
    """An unexpected (non-PipelineStageError) exception inside a model stage -- the same class
    of failure as a CUDA OOM or a raw model RuntimeError -- must isolate to its panel (ERROR)
    and leave the other panel eligible, not abort the whole page.
    """
    grounding = CountingGroundingClient(raise_on_call=1)
    result = run_page_panels(
        two_panel_page_path,
        config,
        vlm_client=StageLevelVLMClient(),
        grounding_client=grounding,
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )

    assert [panel.status for panel in result.panels] == ["ERROR", "PASS"]
    assert result.panels[0].failure_stage == "RuntimeError"
    # Panel 1 raised on its first label's detect call and stopped (ERROR); panel 2's labels
    # were still grounded.
    assert grounding.detect_calls == 1 + len(DEFAULT_ANIMATION_LABELS)
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1


@pytest.fixture
def two_panel_page_path_2(tmp_path: Path) -> Path:
    page = np.full((900, 300, 3), 255, dtype=np.uint8)
    page[0:300, 0:300] = _noise_block(300, 300, seed=31)
    page[500:900, 0:300] = _noise_block(400, 300, seed=32)
    path = tmp_path / "two_panel_page_2.png"
    Image.fromarray(page).save(path)
    return path


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_batch_loads_each_model_once_for_all_pages(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """Phase 18.4 batch residency: `run_pages` processes MANY pages with one model
    residency across ALL of them -- grounding, then object description, then segmentation,
    then reconstruction each load exactly ONCE, never once per page."""
    grounding = CountingGroundingClient()
    segmentation = CountingSegmentationClient()
    reconstruction = CountingReconstructionClient()
    vlm = StageLevelVLMClient()

    results = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=segmentation,
        reconstruction_client=reconstruction,
        out_dir=tmp_path / "videos",
    )

    assert len(results) == 2
    for result in results:
        assert all(panel.status == "PASS" for panel in result.panels)
        assert result.manifest_path.exists()
    # One load/unload per model for the WHOLE batch (stage-level residency across pages),
    # not per page.
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1
    assert segmentation.load_calls == 1
    assert segmentation.unload_calls == 1
    assert reconstruction.load_calls == 1
    assert reconstruction.unload_calls == 1
    # The VLM's one stage stays one residency too.
    assert vlm.unload_calls == 1
    # Every panel of every page was grounded (4 panels, 2 pages x 2).
    assert grounding.detect_calls == 4 * len(DEFAULT_ANIMATION_LABELS)


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_co_residency_loads_all_models_up_front_and_unloads_at_the_end(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """Phase 20/22 residency split (ADR 0021 + 0023): the VLM instance is run-level
    co-resident -- loaded before the first stage call, released once at the end. DINO/SAM/
    LaMa are stage-owned instead: each loads when its stage's worker starts and unloads
    when the stage finishes, so a full int8 Qwen keeps the card's headroom for KV cache
    and prefill (the real OOM this split fixes)."""
    events: list[str] = []

    class LoggedGrounding(CountingGroundingClient):
        def load(self) -> None:
            events.append("load:dino")
            super().load()

        def detect(self, image, text_prompt: str) -> list[Detection]:
            events.append("detect")
            return super().detect(image, text_prompt)

        def unload(self) -> None:
            events.append("unload:dino")
            super().unload()

    class LoggedSegmentation(CountingSegmentationClient):
        def load(self) -> None:
            events.append("load:sam")
            super().load()

        def segment(self, image, box) -> list[MaskCandidate]:
            events.append("segment")
            return super().segment(image, box)

        def unload(self) -> None:
            events.append("unload:sam")
            super().unload()

    class LoggedReconstruction(CountingReconstructionClient):
        def load(self) -> None:
            events.append("load:lama")
            super().load()

        def inpaint(self, image, hole_mask):
            events.append("inpaint")
            return super().inpaint(image, hole_mask)

        def unload(self) -> None:
            events.append("unload:lama")
            super().unload()

    class LoggedVLM(StageLevelVLMClient):
        def __init__(self):
            super().__init__()
            self._n = 0

        def generate(self, image, prompt: str) -> str:
            self._n += 1
            events.append(f"generate:{self._n}")
            return super().generate(image, prompt)

    grounding = LoggedGrounding()
    segmentation = LoggedSegmentation()
    reconstruction = LoggedReconstruction()
    vlm = LoggedVLM()

    results = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=segmentation,
        reconstruction_client=reconstruction,
        out_dir=tmp_path / "videos",
    )
    assert all(p.status == "PASS" for result in results for p in result.panels)

    # Every model loaded once for the whole run, released exactly once.
    assert grounding.load_calls == 1 and grounding.unload_calls == 1
    assert segmentation.load_calls == 1 and segmentation.unload_calls == 1
    assert reconstruction.load_calls == 1 and reconstruction.unload_calls == 1

    # VLM run-level residency: the real client's load() happens in the run-level
    # ModelStage BEFORE the pipeline starts (the fake lazy-loads in generate()).
    assert events.index("generate:1") > events.index("detect")

    # DINO: stage-owned -- loads before its first detect, unloads after its last detect.
    assert events.index("load:dino") < events.index("detect")
    last_detect = max(i for i, e in enumerate(events) if e == "detect")
    assert events.index("unload:dino") > last_detect
    # SAM and LaMa are stage-owned too: each loads before its first call and unloads
    # after its last call, not before it ever started.
    assert events.index("load:sam") < events.index("segment")
    assert events.index("load:lama") < events.index("inpaint")
    last_segment = max(i for i, e in enumerate(events) if e == "segment")
    last_inpaint = max(i for i, e in enumerate(events) if e == "inpaint")
    assert events.index("unload:sam") > last_segment
    assert events.index("unload:lama") > last_inpaint


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_pipelines_panels_without_stage_barriers(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """Phase 21 panel pipeline: a panel moves to the next model as soon as the previous
    stage produced ITS result -- there is no stage barrier. DINO must still be working on
    panel 2 while Qwen is already describing panel 1: the pipeline gates DINO's second
    panel behind an event that only Qwen's first generate() sets."""
    import threading

    qwen_started = threading.Event()
    n_labels = len(DEFAULT_ANIMATION_LABELS)

    class GatedGrounding(CountingGroundingClient):
        def detect(self, image, text_prompt: str) -> list[Detection]:
            if self.detect_calls >= n_labels and not qwen_started.is_set():
                # Panel 2 is being grounded while Qwen has not yet started panel 1 -- a
                # stage barrier would let DINO finish everything first. Wait for Qwen.
                if not qwen_started.wait(timeout=30):
                    raise AssertionError(
                        "Qwen never started before DINO finished panel 1: stage barrier"
                    )
            return super().detect(image, text_prompt)

    class SignalingVLM(StageLevelVLMClient):
        def generate(self, image, prompt: str) -> str:
            qwen_started.set()  # Qwen began describing panel 1
            return super().generate(image, prompt)

    results = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=SignalingVLM(),
        grounding_client=GatedGrounding(),
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )
    assert all(p.status == "PASS" for result in results for p in result.panels)
    assert qwen_started.is_set()
    # The pipeline completed despite DINO's gate, proving overlap happened (and the gated
    # client would have deadlocked under the old sequential scheme).
    assert sum(r.manifest_path.exists() for r in results) == 2


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_vlm_worker_pool_splits_panels_across_instances(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """Phase 22 (ADR 0023): the object-description stage accepts a POOL of VLM clients (one
    int8 Qwen per GPU). Every panel must still be described exactly once -- the 4 panels are
    split between the 2 instances, each instance answers only the panels it actually got."""
    class PooledVLM(StageLevelVLMClient):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            self.generate_calls = 0

        def generate(self, image, prompt: str) -> str:
            self.generate_calls += 1
            return super().generate(image, prompt)

    vlm_a, vlm_b = PooledVLM("a"), PooledVLM("b")
    results = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=[vlm_a, vlm_b],
        grounding_client=CountingGroundingClient(),
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )
    assert all(p.status == "PASS" for result in results for p in result.panels)
    # Every panel was described exactly once, split between the two instances.
    assert vlm_a.generate_calls + vlm_b.generate_calls == 4
    assert vlm_a.generate_calls >= 1 and vlm_b.generate_calls >= 1
    # Both instances were torn down by the run-level ModelStages.
    assert vlm_a.unload_calls == 1 and vlm_b.unload_calls == 1


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_batch_isolates_a_failed_page_from_the_other_page(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """A page that detects nothing at grounding must fail only its own panels (REJECTED);
    the other page still renders, and each model still loaded exactly once."""
    n_labels = len(DEFAULT_ANIMATION_LABELS)
    grounding = CountingGroundingClient(fail_first_n=2 * n_labels)
    results = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=StageLevelVLMClient(),
        grounding_client=grounding,
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=tmp_path / "videos",
    )

    assert [panel.status for panel in results[0].panels] == ["REJECTED", "REJECTED"]
    assert [panel.status for panel in results[1].panels] == ["PASS", "PASS"]
    assert results[0].panels[0].failure_stage == "object_description"
    assert grounding.load_calls == 1
    assert grounding.unload_calls == 1


# --- Phase 18.4 per-stage disk persistence -----------------------------------------------


class NeverLoadedSegmentationClient(CountingSegmentationClient):
    """Fails the test if SAM is ever loaded -- used to prove stage resume skips the model."""

    def load(self) -> None:
        raise AssertionError("SAM must not load when the segmentation stage is resumed")


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_resumes_from_disk_checkpoints_without_loading_completed_models(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """Phase 18.4 disk persistence: after a full `run_pages` pass, a second invocation on the
    same pages loads grounding.json / descriptions.json / segmentation.json and does NOT load
    DINO, Qwen or SAM again -- only the CV/LaMa/render work is re-done."""
    out_dir = tmp_path / "videos"
    grounding = CountingGroundingClient()
    segmentation = CountingSegmentationClient()
    reconstruction = CountingReconstructionClient()
    vlm = StageLevelVLMClient()

    first = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=vlm,
        grounding_client=grounding,
        segmentation_client=segmentation,
        reconstruction_client=reconstruction,
        out_dir=out_dir,
    )
    assert all(p.status == "PASS" for result in first for p in result.panels)

    # Checkpoints exist on disk for every page.
    for result in first:
        page_dir = result.manifest_path.parent
        assert (page_dir / "grounding.json").exists()
        assert (page_dir / "descriptions.json").exists()
        assert (page_dir / "segmentation.json").exists()
        assert any((page_dir / "segmentation").glob("*.npz"))

    # Second run on the same pages + same out_dir: every model-backed stage is restored
    # from disk; the clients must never load.
    class AssertiveGrounding(CountingGroundingClient):
        def load(self) -> None:
            raise AssertionError("DINO must not load when grounding is resumed")

    class AssertiveVLM(StageLevelVLMClient):
        def unload(self) -> None:
            pass  # allow teardown

        def generate(self, image, prompt: str) -> str:
            raise AssertionError("Qwen must not generate when descriptions are resumed")

    second = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=AssertiveVLM(),
        grounding_client=AssertiveGrounding(),
        segmentation_client=NeverLoadedSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=out_dir,
    )
    assert all(p.status == "PASS" for result in second for p in result.panels)


@pytest.mark.skipif(not _requires_ffmpeg(), reason="no ffmpeg binary resolvable")
def test_run_pages_partial_checkpoint_resumes_after_grounding_only(
    two_panel_page_path: Path,
    two_panel_page_path_2: Path,
    config: PipelineConfig,
    tmp_path: Path,
):
    """A killed run leaves only the grounding checkpoint: the next invocation must skip DINO
    (grounding restored from disk) but still run Qwen/SAM -- proving per-stage resume, not
    just full-run checkpointing. The manifest is removed together with the later
    checkpoints: a PASS manifest entry means the panel is fully done (Phase 21 reuse), so
    the test must not rely on it to fake a partial state."""
    out_dir = tmp_path / "videos"
    page_dir = out_dir / two_panel_page_path.stem

    # Simulate a run that completed ONLY the grounding stage: copy the checkpoint files a
    # first (fake) full run produced, then delete the later checkpoints.
    grounding = CountingGroundingClient()
    first = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=StageLevelVLMClient(),
        grounding_client=grounding,
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=out_dir,
    )
    for result in first:
        page_dir = result.manifest_path.parent
        (page_dir / "descriptions.json").unlink()
        for f in (page_dir / "segmentation").glob("*.npz"):
            f.unlink()
        (page_dir / "segmentation.json").unlink()
        (page_dir / "page_manifest.json").unlink()

    class AssertiveGrounding(CountingGroundingClient):
        def load(self) -> None:
            raise AssertionError("DINO must not load: grounding is already checkpointed")

    class CountedVLM(StageLevelVLMClient):
        def __init__(self):
            super().__init__()
            self.generate_calls = 0

        def generate(self, image, prompt: str) -> str:
            self.generate_calls += 1
            return super().generate(image, prompt)

    vlm = CountedVLM()
    second = run_pages(
        [two_panel_page_path, two_panel_page_path_2],
        config,
        vlm_client=vlm,
        grounding_client=AssertiveGrounding(),
        segmentation_client=CountingSegmentationClient(),
        reconstruction_client=CountingReconstructionClient(),
        out_dir=out_dir,
    )
    assert all(p.status == "PASS" for result in second for p in result.panels)
    # Qwen ran once per panel (4 panels) -- grounding was skipped but description was not.
    assert vlm.generate_calls == 4
