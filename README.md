# manga-animation

Turn a single manga page into a short (~3-5s), seamlessly looping animation —
with the minimum amount of visually justified motion needed to express the
action already present in the artwork.

This is **not** an image-to-video generation system. The original manga
artwork is the source of truth: unrelated regions stay pixel-identical, and
only semantically justified objects (a flag, hair, a falling object, an
outstretched hand, cloth, an eye blink...) receive deterministic, kinematic
motion, composited back onto the original image.

## Project status

**Phase 1 — Engineering foundation — complete.** No ML models are integrated yet. This
phase established the project skeleton, configuration system, logging
foundation, the Animation Plan schema, `.claude/` agents and skills, and the
test suite Phase 2+ builds on. See [`docs/pipeline.md`](docs/pipeline.md)
for the planned pipeline and [`docs/decisions/`](docs/decisions) for why it's
structured this way.

**Phase 2 — Model benchmarking & selection — in progress.** The candidate
shortlist and benchmark methodology are written up in
[`docs/decisions/0004-phase2-model-candidates.md`](docs/decisions/0004-phase2-model-candidates.md),
with the machine-readable shortlist in
[`configs/benchmark_candidates.yaml`](configs/benchmark_candidates.yaml) and the
(model-agnostic, no-GPU-required) timing/reporting harness in
[`src/manga_animation/benchmarking`](src/manga_animation/benchmarking). Actual benchmark
runs — loading each candidate and measuring it — happen on the remote Kaggle/Jupyter GPU
worker per [ADR 0003](docs/decisions/0003-remote-compute-workers.md); no model weights are
downloaded or run locally. Reproducible adapter code for every shortlisted candidate lives
in [`scripts/phase2_kaggle_benchmark.py`](scripts/phase2_kaggle_benchmark.py); local,
non-GPU feasibility checks for the `deterministic-animation` and `video-rendering` stages
live in [`scripts/phase2_cv_feasibility.py`](scripts/phase2_cv_feasibility.py) and
[`scripts/phase2_video_feasibility.py`](scripts/phase2_video_feasibility.py). Current
per-stage status (PRIMARY/FALLBACK/PENDING) is tracked in
[ADR 0005](docs/decisions/0005-phase2-model-selection.md).

Planned phases:

| Phase | Scope |
|---|---|
| 1 | Engineering foundation: repo, config, schema, tests, docs, agents/skills |
| 2 | Model benchmarking & selection (VLM, grounding, segmentation, inpainting) |
| 3 | Animation Plan generation from real VLM output |
| 4 | Segmentation, layer decomposition, hidden-region reconstruction |
| 5 | Deterministic/kinematic animation + secondary motion |
| 6 | Compositing, seamless looping, H.264 rendering |
| 7 | End-to-end QA, evaluation, regression testing |

## Architecture overview

```text
Manga page
    -> Panel / scene analysis
    -> VLM semantic understanding
    -> Structured Animation Plan
    -> Object grounding
    -> Precise segmentation
    -> Layer decomposition
    -> Optional hidden-region reconstruction
    -> Deterministic / kinematic animation
    -> Secondary motion
    -> Original-image compositing
    -> Seamless loop
    -> H.264 video
```

Full principles are documented in [`docs/architecture.md`](docs/architecture.md).
The two load-bearing rules:

1. **The local project is the canonical source of truth.** Kaggle/Jupyter GPU
   servers are ephemeral remote compute workers, never the only copy of the
   code.
2. **Static is a valid result.** If there's no visually justified reason for
   an object to move, the system should prefer leaving it static over
   inventing motion.

## Hardware

Developed against:

- **Local:** Apple Silicon (M1 Max, 32GB unified memory), macOS, arm64 — no
  NVIDIA GPU. PyTorch here means CPU or MPS backend, not CUDA.
- **Remote (as needed):** Kaggle T4 / L4 or other Jupyter GPU workers, CUDA.

Because local and remote hardware differ (MPS vs. CUDA, memory budgets,
available dtypes), every hardware-sensitive parameter (`device`, `dtype`,
`batch_size`, `resolution`, `model_variant`, `num_workers`, ...) is
configuration-driven — see [`src/manga_animation/core/config.py`](src/manga_animation/core/config.py)
and [`configs/`](configs) — never hardcoded in pipeline code.

## Local setup

Requires Python 3.11+. This project uses [`uv`](https://docs.astral.sh/uv/)
for environment and dependency management.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the environment and install base + dev dependencies
uv sync --extra dev

# activate it (optional — `uv run` works without activating)
source .venv/bin/activate
```

Optional dependency groups (added as later phases need them):

```bash
uv sync --extra cv      # OpenCV, for segmentation/compositing/animation stages
uv sync --extra video   # ffmpeg-python wrapper (the `ffmpeg` binary itself is a system dependency)
uv sync --extra ml      # torch, for VLM/grounding/segmentation model stages
```

> `ffmpeg` (the binary) and, later, CUDA/NVIDIA drivers on GPU workers are
> **system** dependencies, not installed by `uv`/`pip`. Phase 1 does not
> require `ffmpeg` to be installed locally; it will be needed starting with
> the video-rendering stage.

## Development workflow

```bash
# run tests
uv run pytest

# lint
uv run ruff check .

# type-check
uv run mypy src

# format
uv run ruff format .
```

Configuration lives in [`configs/`](configs) as YAML, loaded and validated
through pydantic models in `src/manga_animation/core/config.py`. Don't scatter
`device="cuda"` / magic numbers through pipeline code — add a field to the
config schema instead.

## Testing

```bash
uv run pytest -v
```

Phase 1 tests cover configuration validation, Animation Plan schema
validation/serialization, deterministic seed behavior, loop-parameter
validation, and package imports. They deliberately avoid tests that only
assert "the class exists" — see `tests/`.

## Remote GPU workflow (Kaggle / Jupyter)

Kaggle/Jupyter GPU servers are **ephemeral remote compute workers**, never
the canonical copy of the project. Workflow:

```text
local: edit code -> git commit -> git push
remote: git pull -> run experiments (GPU-bound work only)
remote: git commit/push  (only if source files changed on the remote)
local: git pull
```

Do not hand-copy files to/from the remote as a substitute for git. Avoid
editing the same files on both sides at once.

If a task requires connecting to a Kaggle/Jupyter server, the assistant will
ask you for the server URL explicitly rather than guessing or reusing a
possibly-stale one.

## Repository layout

```text
src/manga_animation/   # application code (empty stage packages until Phase 2+)
tests/                 # pytest suite
configs/                # YAML configuration files
docs/                   # architecture, pipeline, schema docs, ADRs
.claude/agents/         # specialist Claude Code agents for this project
.claude/skills/         # domain-specific Claude Code skills
scripts/                # one-off developer scripts
examples/               # example inputs/usage (populated in later phases)
outputs/                # git-ignored generated artifacts (videos, frames, debug)
```

## License

MIT (see `pyproject.toml`).
