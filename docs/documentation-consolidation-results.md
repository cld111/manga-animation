# Documentation Consolidation Results

## STATUS

Completed as a documentation-only change. Production source, tests, configuration,
dependencies, and runtime behavior were not changed.

## BEFORE

- `CLAUDE.md` contained useful rules but duplicated navigation and lacked the explicit
  documentation-maintenance policy.
- `docs/current-status.md` was the right canonical file but did not capture several current
  Phase 8-12 limitations, metrics, performance facts, and priorities.
- `docs/README.md` did not define a minimal fresh-session reading path or source ownership.
- One relative link in `.claude/skills/video-rendering/SKILL.md` was broken.

## AFTER

- `CLAUDE.md` is a compact operational entry point.
- `docs/current-status.md` is the single current-state source: pipeline, defaults, validated
  evidence, invariants, limitations, metrics, technical debt, and future priorities.
- `docs/README.md` defines canonical sources, ADR status conventions, and documentation policy.
- Historical phase reports and ADRs remain in place and were not rewritten or duplicated.
- The broken skill link was corrected.

## CANONICAL SOURCES

- Rules and routine workflow: [`CLAUDE.md`](../CLAUDE.md)
- Current truth: [`current-status.md`](current-status.md)
- Stable principles: [`architecture.md`](architecture.md)
- Pipeline contract: [`pipeline.md`](pipeline.md)
- Plan schema: [`animation-plan-schema.md`](animation-plan-schema.md)
- Decisions: [`decisions/`](decisions/)
- Historical evidence: `phase*-results.md`

## HISTORY

All phase reports and ADRs were preserved. No documents were archived, moved, or deleted.
ADR 0004 remains historical and is explicitly identified as superseded by ADR 0005.

## CONTEXT OPTIMIZATION

A fresh session can now use `CLAUDE.md` + `docs/current-status.md` + the current task for
routine orientation. Historical reports and ADRs are reserved for evidence or decision
rationale. Improvement is qualitative; no token reduction is claimed.

## VALIDATION

- Relative Markdown links checked across 50 Markdown files; all resolve.
- Required Phase 8-12 knowledge checked in current state: panel default, transform-aware
  validation, overlap and asymmetric-mask protection, semantic mask overreach and false
  negatives, instance identity, CPU performance, visual QA limits, provenance, and safe
  rejection.
- Focused independent review completed; four wording ambiguities were corrected.
- `uv run pytest`: 576 passed, 2 deselected.
- `uv run ruff check .`: clean.
- `uv run mypy src`: clean, 46 files.

## GIT

Branch, commits, PR URL, and final working-tree state are recorded here after push and PR
creation. Changed files: `CLAUDE.md`, `docs/README.md`, `docs/current-status.md`,
`docs/documentation-consolidation-results.md`, `.claude/skills/video-rendering/SKILL.md`.
