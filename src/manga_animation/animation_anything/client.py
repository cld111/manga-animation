"""`AnimateAnythingClient`: the pipeline's generative animation engine.

The client follows the project's model-client shape (`model_id`, `load()`, `animate()`,
`unload()`) so the panel orchestrator can hold it behind one interface and run it inside a
`ModelStage`. The actual diffusion inference happens in `worker.py`, which runs in an
ISOLATED environment (AnimateAnything's pinned diffusers==0.24.0/transformers==4.36.2/
torch==2.0.0 cannot coexist with this project's transformers>=5.0 stack -- see the package
docstring and docs/decisions/0024). The client therefore shells out: it serializes an
`AnimateAnythingSpec`, invokes the worker interpreter, and reads the frame PNGs back.

`animate()` takes the ORIGINAL panel image, the merged SAM motion mask and the prompt built
from the accepted Qwen descriptions -- exactly the integration contract
docs/decisions/0024-animate-anything-animation-engine.md defines.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.animation_anything.spec import DEFAULT_FPS, AnimateAnythingSpec
from manga_animation.core.logging import get_logger
from manga_animation.pipeline.types import FrameSequence, ImageArray, MaskArray

logger = get_logger(__name__)

# The isolated worker loads the model into the AA GPU on every call. The panel pipeline runs
# several animation stages concurrently (Phase 21), so without serialization N panels hitting
# the AA stage at once would spawn N concurrent worker processes on the same GPU and OOM it
# (real run). One global lock serializes the generative calls across all clients.
_AA_WORKER_LOCK = threading.Lock()

_DEFAULT_TIMEOUT_S = 3600


class AnimateAnythingClient:
    """Subprocess-backed AnimateAnything inference client (image + mask + prompt -> frames).

    Heavy work stays out of this process: `load()` only verifies the worker environment exists
    on the worker, and `animate()` launches `worker.py` under the isolated interpreter and
    blocks until the frames are written. `unload()` is a no-op (nothing model-related is held
    here). Construction is cheap and safe without the `ml` extra installed.
    """

    model_id = "animate-anything-512-v1.02"

    def __init__(
        self,
        source: str,
        python_bin: str,
        worker_script: str | Path,
        device: str = "cuda",
        *,
        num_frames: int = 16,
        fps: int = DEFAULT_FPS,
        num_inference_steps: int = 25,
        guidance_scale: float = 9.0,
        motion_strength: float = 5.0,
        seed: int = 42,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self.source = source  # extracted AnimateAnything checkpoint dir on the worker
        self.python_bin = python_bin  # isolated interpreter (venv with the pinned stack)
        self.worker_script = str(worker_script)
        self.device = device
        self.num_frames = num_frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.motion_strength = motion_strength
        self.seed = seed
        self.timeout_s = timeout_s

    def load(self) -> None:
        """Verify the isolated worker env and checkpoint are present (no model load here --
        the model lives in the worker's process, loaded per `animate()` call)."""
        missing = [
            path
            for path in (Path(self.python_bin), Path(self.worker_script), Path(self.source))
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "AnimateAnything worker environment incomplete on this worker; missing "
                + ", ".join(str(p) for p in missing)
            )

    def animate(
        self,
        image: ImageArray,
        mask: MaskArray,
        prompt: str,
        out_dir: Path,
    ) -> FrameSequence:
        """Generate one panel's animation from (image, mask, prompt) and return the frames.

        `image` is the original RGB panel crop; `mask` is the uint8 0/255 motion mask (merged
        SAM masks); `prompt` comes from `build_animation_prompt`. Writes the spec + inputs into
        `out_dir`, runs the worker, reads the frame PNGs back. Raises
        `subprocess.CalledProcessError`/`FileNotFoundError` on worker failure (fail closed).
        """
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "input_image.png"
        mask_path = out_dir / "motion_mask.png"
        spec_path = out_dir / "spec.json"
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        spec = AnimateAnythingSpec(
            image_path=str(image_path),
            mask_path=str(mask_path),
            prompt=prompt,
            output_dir=str(frames_dir),
            checkpoint_path=self.source,
            device=self.device,
            num_frames=self.num_frames,
            fps=self.fps,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            motion_strength=self.motion_strength,
            seed=self.seed,
        )
        spec.to_json_file(spec_path)

        started = time.perf_counter()
        logger.info(
            "animation_anything: launching worker (prompt=%r, frames=%d, fps=%d) under %s",
            prompt,
            spec.num_frames,
            spec.fps,
            self.python_bin,
        )
        # Serialize across concurrent pipeline workers: only ONE worker process may load the
        # AA model into its GPU at a time (multiple concurrent loads OOM the card).
        with _AA_WORKER_LOCK:
            result = subprocess.run(
                [self.python_bin, self.worker_script, "--spec", str(spec_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        if result.returncode != 0:
            logger.error(
                "animation_anything: worker failed rc=%d stderr=%s",
                result.returncode,
                (result.stderr or "")[-2000:],
            )
            raise subprocess.CalledProcessError(
                result.returncode,
                [self.python_bin, self.worker_script],
                output=result.stdout,
                stderr=result.stderr,
            )
        logger.info(
            "animation_anything: worker finished in %.1fs (%d frames)",
            time.perf_counter() - started,
            spec.num_frames,
        )

        frames = self._read_frames(frames_dir, expected=spec.num_frames)
        return FrameSequence(frames=frames, fps=spec.fps)

    @staticmethod
    def _read_frames(frames_dir: Path, expected: int) -> list[np.ndarray]:
        """Read `frame_%04d.png` files back into RGB uint8 arrays, in order."""
        files = sorted(frames_dir.glob("frame_*.png"))
        if len(files) < expected:
            raise FileNotFoundError(
                f"worker produced {len(files)} frames, expected {expected} in {frames_dir}"
            )
        frames = [np.asarray(Image.open(f).convert("RGB")) for f in files[:expected]]
        if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
            raise ValueError("worker produced a non-RGB frame")
        return frames

    def unload(self) -> None:
        """No in-process model to release -- the worker subprocess already exited."""
