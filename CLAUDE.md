# CLAUDE.md

Operational rules for any Claude Code session (main or subagent) working in this
repository. This file is the authoritative summary — it points to detailed docs rather
than duplicating them. When in doubt, the linked doc wins over this file's paraphrase of
it.

## Project purpose

Turn a single manga page into a short (~3-5s), seamlessly looping animation, using the
minimum amount of visually justified motion needed to express the action already present
in the artwork. This is **not** an image-to-video generation system: the original artwork
is the source of truth, and only semantically justified objects get deterministic motion.
See [`README.md`](README.md) for the full pitch and current phase status, and
[`docs/architecture.md`](docs/architecture.md) for the engineering principles every
change should be checked against.

## Canonical source / compute split

1. **The local checkout is always the complete, canonical copy** of code, tests, config,
   docs, and `.claude/agents` / `.claude/skills`. See
   [ADR 0002](docs/decisions/0002-local-canonical-source.md).
2. **Remote compute (Kaggle/Jupyter/etc.) is a disposable worker, never canonical.** It
   can disappear at any time without warning; nothing about the project's state may depend
   on a remote session surviving. See
   [ADR 0003](docs/decisions/0003-remote-compute-workers.md).
3. Moving source changes between local and remote happens **only via git**
   (`local: edit → commit → push`, `remote: pull → run GPU work → commit/push only if
   source changed`, `local: pull`) — never manual file copying.
4. **Generated outputs (rendered videos, frames, debug images, fetched sample pages,
   benchmark artifacts) must never become canonical project files.** They belong under
   `outputs/` or `examples/`, both git-ignored (see `.gitignore`), and must stay
   regenerable from source rather than committed.
5. **Never guess a Jupyter/Kaggle server URL**, and never reuse a possibly-stale one from
   an earlier session. If a task needs to reach an actual remote server and no URL has been
   given, ask the user for it explicitly before proceeding.
6. Standing project policy (see [ADR 0004](docs/decisions/0004-phase2-model-candidates.md))
   is that the pipeline is not run locally even for a smoke test — actual model
   loading/inference happens on the remote GPU worker. Local work is code, tests, config,
   and docs.

## Architecture and decisions

- Respect the existing architecture ([`docs/architecture.md`](docs/architecture.md)) and
  the accepted ADRs in [`docs/decisions/`](docs/decisions/). Read the relevant ones before
  touching a stage's design.
- Do not silently change an architectural decision (schema shape, stage boundaries, model
  abstraction, pipeline order, agent ownership). If something documented needs to change,
  say so explicitly and update the doc/ADR — don't let code and docs drift apart.
- The pipeline stage sequence and each stage's `src/manga_animation/<package>` home are in
  [`docs/pipeline.md`](docs/pipeline.md); the Animation Plan contract is in
  [`docs/animation-plan-schema.md`](docs/animation-plan-schema.md).

## Agents and skills

- Use the specialist agents in [`.claude/agents/`](.claude/agents/) and the domain skills
  in [`.claude/skills/`](.claude/skills/) where a task matches their scope, rather than
  reimplementing their judgment ad hoc in the orchestrating session.
- Every agent is a specialist, not the project owner — the orchestrating session makes
  final decisions and is responsible for wiring agents' work together.
- **Agents must report completion, blockers, or critical findings back to the
  orchestrating session via `SendMessage`** — see the "Reporting completion to the
  orchestrator" section in each agent file for the exact format. This is not a
  continuous-progress channel; routine intermediate steps don't get a message.
- Stage ownership, including the hidden-region reconstruction stage, is spelled out per
  agent in `.claude/agents/*.md` and summarized in the "Stage ownership" section of
  [`docs/pipeline.md`](docs/pipeline.md) — check there before assuming who owns a stage.

## Before declaring implementation complete

- Tests, lint, and type checks must pass: `uv run pytest`, `uv run ruff check .`,
  `uv run mypy src`. Don't report a task done with any of these red.
- Keep tests behavioral (assert on actual outputs/invariants), not existence checks — see
  `tests/` for the existing style.
- Artifacts and source changes produced anywhere (including on a remote worker) must be
  synchronized back into this local canonical repository via git before the work is
  considered finished — an uncommitted remote-only change is not done.
