# Phase 2 qualitative evaluation checklist

The Phase 2 brief splits benchmarking into objective measurements (latency, VRAM, load
time — `BenchmarkResult`'s numeric fields, see `src/manga_animation/benchmarking/schemas.py`)
and human/semantic quality assessment, which is deliberately *not* reduced to a single
score (see [ADR 0004](decisions/0004-phase2-model-candidates.md)'s benchmark methodology).
This checklist standardizes what that qualitative pass actually looks at, so
`BenchmarkResult.quality_notes` entries across different candidates and stages answer the
same questions instead of being freeform impressions. Every answer here is a **subjective,
labeled judgment call**, not a metric — record it as prose in `quality_notes`, not as a
number in `quality_score` unless the candidate/stage genuinely has an automatable check
(e.g. mask-area sanity, per the `segmentation` skill).

Run this against a handful of real, representative manga pages per candidate (see
`scripts/fetch_sample_pages.py` — colored interior pages only, per the project's sourcing
rule), not one page — see `docs/phase2-benchmark-results.md`'s "Next steps" for why n=2 was
already flagged as too small to conclude from.

## `vlm` (semantic analysis)

- Does the model correctly identify what's actually drawn (objects, action), or hallucinate
  content the page doesn't show?
- Does it recognize manga-specific motion cues (speed lines, impact marks, flowing
  linework) per the `manga-analysis` skill, or only generic "person standing" descriptions?
- Is its STATIC-vs-motion judgment appropriately conservative (see "Static Is a Valid
  Result" in `docs/architecture.md`), or does it over-eagerly assign motion everywhere?
- OCR/dialogue handling: does dialogue-heavy text interfere with or get confused for scene
  content?
- Would its output, as-is, survive `AnimationPlan` schema validation, or does it need
  significant reshaping?

## `grounding` (object localization)

- Is the correct object identified for the given semantic label, not a visually similar but
  wrong region (e.g. a speech bubble mistaken for a face)?
- Are boxes tight, or noticeably loose/oversized on manga line art vs. what the same
  candidate would produce on photographic data?
- How many expected classes return *no* detection at any reasonable threshold (see
  `docs/phase2-benchmark-results.md`'s finding: `face`/`hair`/`speech bubble` returned
  nothing on the first pass) — is this a threshold problem or a genuine domain-fit gap?
- Overlap/occlusion: when two objects' true regions overlap, does the candidate return
  separate sensible boxes, or one merged box?

## `segmentation` (pixel-accurate masks)

- Do mask edges follow the actual inked linework, or cut across it (see the
  `segmentation` skill's "Mask validation checklist")?
- Any stray disconnected components or unintended holes, especially over heavy linework or
  screentone shading?
- Is coverage plausible for the object type (e.g. hair masks rarely near-100% of their
  bbox)?
- Does quality visibly degrade specifically on manga-style shading/screentone vs. how the
  same candidate performs on flat-color or photographic regions?

## `inpainting` (hidden-region reconstruction)

- Does the reconstructed region look plausible *in context* (continues linework, shading,
  screentone pattern from the surrounding area), or is it visibly a generic
  photographic-style fill?
- Does it introduce hallucinated detail (new lines, new shading patterns, new objects) that
  wasn't implied by the surrounding art — see `cv-agent.md`'s ownership note: reconstruction
  should fill the hole, not redraw it?
- Is line art continuity preserved across the reconstructed boundary, or is there a visible
  seam?
- Does output resolution/style match the source page's, or does it look like a
  different (smoother, more "generated") texture?

## Cross-cutting: suitability for downstream deterministic animation

For every stage, the question that actually matters for this project (not just "is this
output good in isolation"):

- Is the output precise/stable enough for `cv-agent` to apply a deterministic transform to
  it without visible artifacts (see the `cv-animation` skill's "Common artifacts to check
  for")? A grounding box that's "close enough" for a human reader may still be too loose for
  `mesh_warp` to look clean at its edges.
- Would this candidate's failure mode (missed detection, hallucinated inpainting content,
  ragged mask edge) show up as a *silent* quality problem downstream, or would it fail
  loudly enough for `qa-agent`'s checks (`docs/decisions` + the `evaluation` skill) to catch
  it?

## Recording results

Attach findings to the relevant `BenchmarkResult.quality_notes` (free text) when produced by
`scripts/phase2_kaggle_benchmark.py`, and summarize cross-candidate qualitative comparisons
in prose in `docs/phase2-benchmark-results.md` (or a stage-specific results doc following
that file's format) — not folded into the objective-measurement table, so a reader can tell
at a glance which numbers are measured and which judgments are subjective.
