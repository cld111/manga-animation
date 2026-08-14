# Current Status

This is the current project status. Historical implementation details and experiment
evidence live in the `phase*-results.md` files and should not be used as the current
pipeline contract.

## Maturity

The deterministic pipeline and its local test/evaluation infrastructure are implemented
through Phase 12. Real model execution remains a remote-GPU operation. The project is an
engineering prototype with real end-to-end evidence, not a production-quality animation
service.

| Area | Current state | Source of evidence |
|---|---|---|
| Analysis | Panel mode is the default; page mode remains explicit | `docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md` |
| Grounding | Grounding DINO, panel-crop aware | `docs/decisions/0011-panel-aware-grounding.md` |
| Target validation | Pre-segmentation semantic and transform-geometry gates | `docs/decisions/0006-grounding-target-validation.md`, `0008-transform-aware-target-validation.md` |
| Segmentation | SAM 2.1 with mask-shape validation | `docs/decisions/0015-duplicate-silhouette-and-seam-fixes.md` |
| Semantic mask validation | Post-segmentation ACCEPT/REJECT/ABSTAIN gate, enabled by default | `docs/decisions/0018-semantic-mask-validation.md` |
| Animation | Deterministic OpenCV/NumPy transforms | `src/manga_animation/animation` |
| Reconstruction | LaMa, loaded only when motion reveals a hole | `src/manga_animation/reconstruction` |
| Compositing | Multi-layer, deterministic z-order and overlap protection | `docs/decisions/0010-multi-object-layer-decomposition.md`, `0015-duplicate-silhouette-and-seam-fixes.md` |
| Rendering | FFmpeg/H.264 with decoded-output validation | `src/manga_animation/rendering` |
| Evaluation | Golden and real-world datasets, structured outcome metrics | `src/manga_animation/evaluation` |

## Runtime Baseline

The currently implemented production clients are:

| Stage | Candidate |
|---|---|
| VLM | `qwen2.5-vl-7b-instruct` |
| Grounding | `grounding-dino-swin-l` |
| Segmentation | `sam2.1-hiera-base` |
| Inpainting | `lama-large` |

These are preliminary operational selections, not an exhaustive benchmark conclusion.
Other entries in `configs/benchmark_candidates.yaml` are benchmark candidates and are
not silently accepted by the production client factory until an adapter exists.

## Current Invariants

- Before encoding, pixels outside transformed masks are copied from the original image.
- After H.264 decoding, bounded codec noise is expected; bit-exact static preservation is
  an invariant of raw composited frames, not of decoded compressed video.
- A plan contains at most one `PRIMARY` object.
- A `SECONDARY`/`MICRO` failure is isolated from the run; a `PRIMARY` failure rejects it.
- A segmentation mask must be a full-source-image, 2D `uint8` array.
- `analysis_mode="panel"` is the default. Use `analysis_mode="page"` only intentionally.
- All model-backed stages, including semantic mask validation, release their model after the
  stage completes or fails.
- `PipelineConfig.resolution` controls VLM analysis resizing only; downstream CV stages use
  the original source geometry. `dtype` describes the VLM; verified grounding/segmentation
  clients use `float32`.
- Only `h264` is a supported output codec. Crossfade frames are reserved and must remain zero.

## Known Product Gaps

- `wind_breaker_finish` remains a real, unfixed mid-cycle visual defect.
- `marika_love_meter` has an unknown root cause and is currently mitigated by safe rejection
  in panel mode, not fixed.
- There is no evidence-calibrated general detector for masks that are too large for their
  semantic target.
- `MESH_WARP` has no calibrated upper bound relative to panel/page geometry.
- The automated seam detector has approximately 50% real-world precision and is not a
  substitute for visual QA.
- Semantic-mask validation is development-data calibrated only: the current benchmark has
  13 samples, reported recall is 0.60, and `cloth_5` is a known false negative.
- ABSTAIN has not yet been observed on a real benchmark call, and same-category instance
  identity is not checked.
- Kinematic parent/child fields are structurally validated, but automatic transform
  inheritance is not implemented yet.

## Verification

Run locally:

```bash
uv run pytest
uv run pytest -m slow
uv run ruff check .
uv run mypy src
```

Actual model loading and inference must run on a remote GPU worker, never as a local
pipeline smoke test. Generated media and experiment JSON remain under git-ignored
`outputs/`.

## Documentation Map

- `README.md`: short project entry point and developer commands.
- `docs/architecture.md`: current engineering principles and invariants.
- `docs/pipeline.md`: current stage order, ownership and lifecycle contract.
- `docs/animation-plan-schema.md`: machine-readable plan contract.
- `docs/decisions/`: architectural decisions and their revisions.
- `docs/phase*-results.md`: historical experiment/evidence records.
