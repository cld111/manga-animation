"""Tests for the AnimateAnything generative animation engine (ADR 0024).

Covers the deterministic, locally-testable pieces of the engine: the spec contract, the
prompt builder, the SAM-mask merger, and the client's subprocess hand-off (with a FAKE worker
script -- real diffusion inference is remote-GPU work and never runs here). The pipeline
integration is covered in tests/test_animation_anything_pipeline.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from manga_animation.animation_anything.client import AnimateAnythingClient
from manga_animation.animation_anything.mask import merge_motion_masks
from manga_animation.animation_anything.prompt import (
    build_animation_prompt,
    motion_phrase,
)
from manga_animation.animation_anything.spec import AnimateAnythingSpec
from manga_animation.pipeline.types import ObjectDescriptionResult
from manga_animation.schemas.animation_plan import (
    MotionSpec,
    MotionType,
    ObjectPlan,
    TransformKind,
)


def _description(
    *,
    object_id: str,
    accepted: bool = True,
    object_identity: str | None = None,
    transform_kind: TransformKind = TransformKind.MESH_WARP,
    confidence: float = 0.9,
) -> ObjectDescriptionResult:
    motion_spec = None
    if accepted:
        kwargs: dict = {"transform_kind": transform_kind, "amplitude": 0.1, "speed": 1.0}
        if transform_kind in (TransformKind.TRANSLATE, TransformKind.SHEAR):
            from manga_animation.schemas.animation_plan import Vector2

            kwargs["direction"] = Vector2(x=0.0, y=1.0)
        motion_spec = MotionSpec(**kwargs)
    return ObjectDescriptionResult(
        object_id=object_id,
        accepted=accepted,
        assessment="pass" if accepted else None,
        matches_semantic_label=accepted or None,
        animatable=accepted or None,
        object_identity=object_identity,
        motion_spec=motion_spec,
        confidence=confidence,
        reason="the panel shows this object in motion",
    )


def _object_plan(object_id: str, motion_type: MotionType = MotionType.SECONDARY) -> ObjectPlan:
    return ObjectPlan(
        object_id=object_id,
        panel_id="panel_001",
        semantic_label=object_id,
        confidence=0.9,
        motion_type=motion_type,
        motion=(
            MotionSpec(transform_kind=TransformKind.MESH_WARP, amplitude=0.1, speed=1.0)
            if motion_type != MotionType.STATIC
            else None
        ),
    )


# -------------------------------------------------------------------------------------
# Spec contract
# -------------------------------------------------------------------------------------


def test_spec_round_trips_through_json(tmp_path: Path):
    spec = AnimateAnythingSpec(
        image_path="/tmp/img.png",
        mask_path="/tmp/mask.png",
        prompt="the speed lines flowing",
        output_dir="/tmp/out",
        checkpoint_path="/tmp/ckpt",
        num_frames=16,
        fps=8,
        seed=7,
    )
    path = tmp_path / "spec.json"
    spec.to_json_file(path)
    loaded = AnimateAnythingSpec.from_json_file(path)
    assert loaded == spec
    assert loaded.fps == 8
    assert loaded.motion_strength == 1.0  # default is deliberately slow/gentle motion


def test_spec_defaults_match_the_models_native_output():
    spec = AnimateAnythingSpec(
        image_path="i", mask_path="m", prompt="p", output_dir="o", checkpoint_path="c"
    )
    assert (spec.num_frames, spec.fps) == (16, 8)  # 2 s native clip


def test_spec_as_manifest_is_json_safe_and_basename_only():
    spec = AnimateAnythingSpec(
        image_path="/a/b/c.png",
        mask_path="/a/b/m.png",
        prompt="x",
        output_dir="/a/b/o",
        checkpoint_path="/kaggle/working/models/animate_anything_512_v1.02",
    )
    manifest = spec.as_manifest()
    assert manifest["image_path"] == "c.png"
    assert manifest["checkpoint_path"] == "animate_anything_512_v1.02"
    assert "/" not in manifest["checkpoint_path"]


# -------------------------------------------------------------------------------------
# Prompt builder (Qwen descriptions -> AnimateAnything prompt)
# -------------------------------------------------------------------------------------


def test_prompt_puts_primary_first_and_comma_joins():
    objects = [
        (_object_plan("obj_speed_lines_0", MotionType.SECONDARY),
         _description(object_id="obj_speed_lines_0", object_identity="speed_lines")),
        (_object_plan("obj_character_1", MotionType.PRIMARY),
         _description(object_id="obj_character_1", object_identity="a character",
                      transform_kind=TransformKind.MESH_WARP)),
    ]
    prompt = build_animation_prompt(objects)
    assert prompt.startswith("a character flowing")
    assert prompt.endswith("speed lines flowing")
    assert ", " in prompt


def test_prompt_uses_transform_kind_phrase():
    objects = [
        (_object_plan("obj_1", MotionType.PRIMARY),
         _description(object_id="obj_1", object_identity="the burst",
                      transform_kind=TransformKind.RADIAL_EXPAND)),
    ]
    assert build_animation_prompt(objects) == "the burst pulsing outward"


def test_prompt_falls_back_to_semantic_label_when_identity_missing():
    objects = [
        (_object_plan("obj_1", MotionType.PRIMARY),
         _description(object_id="obj_1", object_identity=None)),
    ]
    assert build_animation_prompt(objects) == "obj 1 flowing"


def test_prompt_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        build_animation_prompt([])


def test_motion_phrase_maps_every_transform_kind():
    for kind in TransformKind:
        result = motion_phrase(
            _description(object_id="x", transform_kind=kind)
        )
        assert isinstance(result, str) and result


# -------------------------------------------------------------------------------------
# SAM mask merger
# -------------------------------------------------------------------------------------


def test_merge_motion_masks_unions_masks():
    a = np.zeros((10, 10), dtype=np.uint8)
    a[:5, :5] = 255
    b = np.zeros((10, 10), dtype=np.uint8)
    b[5:, 5:] = 255
    merged = merge_motion_masks([a, b])
    assert merged.shape == (10, 10)
    assert set(np.unique(merged)) <= {0, 255}
    assert (merged[:5, :5] == 255).all()
    assert (merged[5:, 5:] == 255).all()
    assert merged[2, 7] == 0  # untouched corner


def test_merge_motion_masks_is_binary_not_summed():
    a = np.full((4, 4), 255, dtype=np.uint8)
    b = np.full((4, 4), 255, dtype=np.uint8)
    assert set(np.unique(merge_motion_masks([a, b]))) == {255}


def test_merge_motion_masks_rejects_empty_and_shape_mismatch():
    with pytest.raises(ValueError, match="empty"):
        merge_motion_masks([])
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        merge_motion_masks([a, b])


# -------------------------------------------------------------------------------------
# Client subprocess hand-off (fake worker -- no torch/diffusers needed)
# -------------------------------------------------------------------------------------

_FAKE_WORKER = textwrap.dedent(
    """\
    import json, sys
    from pathlib import Path
    from PIL import Image
    import numpy as np

    spec = json.loads(Path(sys.argv[sys.argv.index("--spec") + 1]).read_text())
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for i in range(spec["num_frames"]):
        Image.fromarray(np.full((16, 16, 3), i * 10, dtype=np.uint8)).save(
            out / f"frame_{i:04d}.png"
        )
    print(json.dumps({"ok": True, "frames": spec["num_frames"]}))
    """
)


@pytest.fixture
def fake_worker(tmp_path: Path) -> Path:
    path = tmp_path / "fake_worker.py"
    path.write_text(_FAKE_WORKER)
    return path


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    ckpt = tmp_path / "animate_anything_512_v1.02"
    for sub in ("scheduler", "tokenizer", "text_encoder", "vae", "unet"):
        (ckpt / sub).mkdir(parents=True, exist_ok=True)
    return ckpt


def test_client_animate_writes_spec_inputs_and_reads_frames(
    fake_worker: Path, fake_checkpoint: Path, tmp_path: Path
):
    client = AnimateAnythingClient(
        source=str(fake_checkpoint),
        python_bin=sys.executable,
        worker_script=fake_worker,
        device="cpu",
        num_frames=4,
        fps=8,
    )
    client.load()
    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:16, :16] = 255
    frames = client.animate(image, mask, "the character flowing", tmp_path / "out")

    assert frames.fps == 8
    assert len(frames.frames) == 4
    assert all(f.shape == (16, 16, 3) for f in frames.frames)

    workdir = tmp_path / "out"
    assert (workdir / "spec.json").exists()
    assert (workdir / "input_image.png").exists()
    assert (workdir / "motion_mask.png").exists()


def test_client_load_raises_on_missing_worker_env(fake_checkpoint: Path):
    client = AnimateAnythingClient(
        source=str(fake_checkpoint),
        python_bin="/nonexistent/python",
        worker_script="/nonexistent/worker.py",
        device="cpu",
    )
    with pytest.raises(FileNotFoundError):
        client.load()


def test_client_animate_fails_closed_on_worker_error(
    fake_checkpoint: Path, tmp_path: Path
):
    bad_worker = tmp_path / "bad_worker.py"
    bad_worker.write_text("import sys; sys.exit(1)\n")
    client = AnimateAnythingClient(
        source=str(fake_checkpoint),
        python_bin=sys.executable,
        worker_script=bad_worker,
        device="cpu",
        num_frames=2,
    )
    with pytest.raises(subprocess.CalledProcessError):
        client.animate(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8), dtype=np.uint8),
            "x",
            tmp_path / "out",
        )
