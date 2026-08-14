# manga-animation

Turn a single manga page into a short (~3-5s), seamlessly looping animation using the
minimum visually justified motion needed to express the action already present in the
artwork.

This is not an image-to-video generation system. The original artwork is the source of
truth: raw composited frames preserve unrelated pixels, while selected objects receive
deterministic kinematic motion and are composited back onto the original image.

## Status

The deterministic pipeline and evaluation infrastructure are implemented through Phase 12.
Panel-aware analysis is the default. Real model inference runs only on a remote GPU worker.
The project remains an engineering prototype with documented real-world visual limitations.

See [`docs/current-status.md`](docs/current-status.md) for the active runtime baseline,
invariants and known gaps. Historical results remain in [`docs/`](docs) as evidence records.

## Pipeline

```text
Manga page
    -> panel/scene analysis
    -> VLM semantic understanding
    -> Animation Plan
    -> grounding
    -> target validation
    -> precise segmentation
    -> post-segmentation safety gates
    -> semantic mask validation
    -> deterministic animation
    -> hidden-region reconstruction when needed
    -> original-image compositing
    -> decoded-output validation
    -> H.264 video
```

The architecture is intentionally hybrid: learned models decide what an object is and where
it is, while deterministic CV controls how its pixels move. A completely static semantic
read is valid evidence that no justified motion was found; the current animation pipeline
reports it as a rejected run because it has no target to render.

## Hardware

- Local development: Apple Silicon/macOS, CPU or MPS, without model inference.
- Remote execution: Kaggle/Jupyter T4, L4 or another CUDA worker.

Model weights and actual inference are never required for local unit tests. Remote workers
are disposable; the local git checkout is canonical.

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
source .venv/bin/activate  # optional; uv run also works
```

Optional groups:

```bash
uv sync --extra video  # ffmpeg-python; the ffmpeg binary is system-provided
uv sync --extra ml     # torch/transformers/model clients for a remote worker
```

OpenCV is a base dependency because implemented analysis, animation and rendering modules
import `cv2` directly. `ffmpeg` itself is a system dependency.

## Development

```bash
uv run pytest
uv run pytest -m slow
uv run ruff check .
uv run mypy src
uv run ruff format .
```

## Remote Workflow

```text
local: edit -> commit -> push
remote: pull -> run GPU work -> commit/push only if source changed
local: pull
```

Never guess a Jupyter/Kaggle URL or hand-copy source files. Generated videos, frames,
experiment JSON and fetched sample pages remain git-ignored.

## Repository Layout

```text
src/manga_animation/   application code
tests/                  behavioral and regression tests
configs/                YAML configuration and datasets
docs/                   current contracts, ADRs and historical evidence
scripts/                reproducible developer/remote-worker entry points
examples/               local, non-canonical sample inputs
outputs/                git-ignored generated artifacts
```

## License

MIT (see `pyproject.toml`).
