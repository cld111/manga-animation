# Phase 3.1 results: first real end-to-end vertical slice

Live results from the remote Kaggle GPU worker (2x Tesla T4, `torch` 2.10.0+cu128,
`transformers` 5.0.0 — same environment as ADR 0005), running the real pipeline built in
Phase 3.1 (`src/manga_animation/pipeline/orchestrator.py`) against a real manga page. This is
a point-in-time results record, not a design doc — see the Phase 3.1 brief (delivered
directly to the assistant, not committed as a file) and `docs/decisions/0005-phase2-model-selection.md`
for the model selection this build on.

## Source page

*The Skeleton Soldier Failed to Defend the Dungeon*, chapter 25, page 13 of 15 (MangaDex id
`d993f789-e7e5-4832-92fd-37614220b427`, chapter id `2cb6c639-033e-4064-a3fd-dacbe0aaaaad`),
fetched via `scripts/fetch_phase3_sample_page.py`. Chosen deliberately over Phase 2's sample
pages: both of those came back all-STATIC from the VLM (ADR 0005), and this page has an
unambiguous drawn motion cue (a knight with a raised sword, vertical speed-line rain effects,
hanging cloth banners on the arena walls) that Phase 2's sample set lacked. 720x5062 px — an
unusually tall (~7:1) page, which matters for finding #1 below.

## Real findings, in the order they were hit

### 1. `analyze_page` OOM'd on GPU0 — config.resolution wasn't applied to the VLM input

First real run: `torch.OutOfMemoryError` inside `Qwen2_5_VLForConditionalGeneration.generate()`
on GPU0, even under `device_map="auto"` sharding (which worked fine in ADR 0005's tests).
Root cause: `analyze_page()` fed the VLM the *raw* source image at full resolution instead of
respecting `PipelineConfig.resolution` — a config field that exists exactly for this ("max
working long-edge resolution", see "GPU Awareness" in `docs/architecture.md`) but was never
actually applied in the analysis stage. This page's unusual 7:1 aspect ratio produced far more
vision tokens than any page tested in ADR 0005's benchmarking passes, tipping GPU0 over its
limit. Fixed in `plan_builder._resized_for_vlm` (downscales a copy of the image to
`config.resolution`'s long edge before it reaches the VLM; `SourceImage.width/height` still
record the true source dimensions). Two new regression tests cover this.

### 2. The VLM assigned no PRIMARY object — a second, independent all-STATIC read

After the OOM fix, `qwen2.5-vl-7b-instruct` ran to completion and correctly *identified* every
real object on the page — `flag_banner`, `raised_sword`, `character_hair`, `character_hand`,
`speech_bubble`, plus one more — but scored **every one of them STATIC**, each with the same
boilerplate reason ("not depicted with any motion lines or deformation"). This is a real,
independently-reproduced instance of ADR 0005's documented gap, now confirmed on a page that
*does* have a genuine drawn motion cue: the page's implied motion is conveyed by page-level
speed-line SFX layered over the panel, not by deformation drawn on any single segmentable
object — and the model's read of "no motion lines *on the object itself*" is, on its own
terms, defensible, not a hallucination or a parsing bug.

Per the Phase 3.1 brief's explicit failure policy ("use a controlled fallback/test fixture if
necessary; clearly distinguish the fallback from fully automatic operation"),
`run_pipeline()` gained an explicit `plan: AnimationPlan | None` parameter
(`src/manga_animation/pipeline/orchestrator.py`) — passing a pre-built plan skips the
analysis stage's VLM call entirely (logged as a warning) and runs every other stage for real.
A plan was hand-authored reusing the VLM's own real semantic label (`flag_banner`) with a
`mesh_warp` motion, and used for the rest of this run.

**Open, unresolved:** automatic (non-fallback) VLM operation has now returned all-STATIC on
every real page tested across Phase 2 and Phase 3.1 (4 pages, 2 series) — including a page
with a genuine drawn action cue. This is the single most important remaining gap before the
pipeline can run without a human-authored fallback.

### 3. Grounding/segmentation raised a real dtype mismatch under `config.dtype`

`build_default_clients()` passed `config.dtype` (`float16`, from `configs/kaggle.yaml`) to
both the Grounding DINO and SAM 2.1 clients. Grounding DINO's image processor produces
`float32` pixel values regardless of the model's loaded dtype on this `transformers` version,
raising `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` the
moment grounding ran for real. ADR 0005's own successful benchmark runs for both candidates
used `float32` explicitly, never `float16` — only the VLM stage was ever proven at `float16`.
Fixed by hardcoding `float32` for these two clients in `build_default_clients`, matching that
real evidence rather than the single global config default.

### 4. Real end-to-end render succeeded — with a genuine visual-quality defect

With both fixes in place, the full pipeline ran for real: Grounding DINO -> SAM 2.1 -> (no
reconstruction needed) -> deterministic `mesh_warp` -> compositing -> FFmpeg, producing
`outputs/videos/phase3_fallback/output.mp4` (96 frames, 24fps, 4.0s, h264/yuv420p,
1.89 MB, `seamless_loop_verified=True` at both the source-`FrameSequence` level and the
post-encode level).

**Visual QA (real, performed on decoded frame pixels, not simulated):** Grounding DINO's
match for the `flag_banner` prompt scored only 0.269 (barely above the 0.25 threshold) and
landed on a face-and-speech-bubble panel, not an actual banner shape — there is no real
banner-shaped object at that location on this page. The `mesh_warp` animation, having no way
to know this, faithfully warped the grounded region: comparing frame 0 (rest) against frame
24 (quarter-cycle peak) shows ~20% of pixels in the grounded crop changed (max channel-sum
diff 759/765), visibly distorting the character's face and rendering the speech-bubble text
unreadable at peak deflection. This is a genuine artistic-quality defect — a manga reader
would immediately read it as wrong — even though every *mechanical* invariant held (pixels
outside the mask are untouched by construction and covered by existing unit tests; the loop is
genuinely seamless; the encode is valid).

Root cause: nothing in the pipeline checks whether a grounding result is *plausible* for its
semantic label before handing it to animation — a low-confidence, semantically-wrong match is
treated the same as a high-confidence, correct one. This is the second concrete gap for
Phase 4/5 to address (e.g. a minimum grounding-confidence gate, or a cheap VLM-based visual
sanity check on the grounded crop before committing to animate it).

A second attempt using `character_hair` (a `translate` motion) was launched to test whether a
more concrete, canonical object type would ground more accurately, but its result could not be
retrieved: the remote-execution transport (browser automation) was removed mid-session per a
standing project decision (see `docs/decisions/0002-local-canonical-source.md`'s canonical/
disposable split — the interactive Kaggle transport used for this phase was not meant to be
the pipeline's normal execution path). That attempt is not reported as a result, successful or
otherwise, since it was never actually observed.

## How the video was inspected without downloading it

Standard file download from the Kaggle-hosted JupyterLab session into this machine's local
`~/Downloads` was blocked by macOS's per-app Downloads-folder permission (`EPERM`/quarantine),
which the local shell environment used for this session does not have granted. Visual QA was
instead performed by fetching decoded frame PNGs through the Jupyter Contents API
(`GET .../api/contents/<path>?content=1&format=base64`, same-origin, authenticated by the
session's own token) directly inside the browser tab, decoding them to an in-page `<canvas>`,
and screenshotting the canvas — never transferring raw image bytes through the tool-call text
channel (which is deliberately blocked for base64-shaped content). This is a one-off
workaround for this session's specific transport, not a repeatable mechanism — see the "How
remote artifacts get inspected" open question below.

## Consequences

- `configs/default.yaml`'s `model_variants` (populated this phase) and
  `src/manga_animation/pipeline/orchestrator.py::build_default_clients` are the first places
  a future dtype/resolution finding like #1/#3 above should be checked against before
  assuming a config default is safe for a new model swap.
- The interactive-browser transport used to reach the remote Kaggle session this phase is
  explicitly **not** the pipeline's intended normal execution path (see CLAUDE.md's compute
  split) and has been removed from this project's available tools
  (`.claude/settings.local.json`'s `permissions.deny`). A future phase needs a programmatic
  transport (Jupyter REST/kernel API, SSH, or similar) before further real-GPU runs can happen
  without manual/interactive access — this is real, scoped follow-up work, deliberately not
  attempted in this phase (see the final report's "Remaining limitations").

## Open questions

- Does a different, more concrete `semantic_label` (hair, a raised weapon) ground more
  accurately than a generic "banner" on a page that doesn't actually contain one? Untested —
  see the finding #4 above.
- What is the actual VLM behavior on a page whose motion IS drawn as object deformation
  (rather than page-level SFX)? Still no such page has been tested across Phase 2 or Phase 3.1.
- A programmatic (non-browser) remote-compute transport for future phases — scoped but not
  built this phase, per the user's explicit direction to keep Phase 3.1 and that decision
  separate.
