# CLAUDE.md

Permanent operational context for Claude Code sessions in this repository. The short
reading path is: this file -> [`docs/current-status.md`](docs/current-status.md) -> the
current task. Detailed contracts, decisions, and historical evidence are linked there;
do not use phase reports as the current contract.

## Project

Turn one manga page into a short (~3-5s), seamlessly looping animation with the minimum
visually justified motion. This is not image-to-video generation: the source artwork is
the truth, unaffected pixels are preserved, and selected objects receive deterministic
CV motion. Read [`docs/architecture.md`](docs/architecture.md) for stable principles and
[`docs/pipeline.md`](docs/pipeline.md) for the stage contract.

## Non-negotiable workflow

- The local checkout is the canonical copy of source, tests, config, docs, and `.claude/`;
  remote Kaggle/Jupyter sessions are disposable workers ([ADR 0002](docs/decisions/0002-local-canonical-source.md),
  [ADR 0003](docs/decisions/0003-remote-compute-workers.md)).
- Move source only through git: `local: edit -> commit -> push`, `remote: pull -> run GPU
  work -> commit/push only if source changed`, `local: pull`. Never hand-copy source files.
- Never guess or reuse a Jupyter/Kaggle URL. Ask the user for a current URL when remote
  execution is required.
- Real model loading/inference is remote-GPU work only. Local work is code, tests, config,
  documentation, and deterministic CPU/CV checks.
- Generated media, fetched pages, debug files, and experiment JSON are ignored artifacts
  under `outputs/` or `examples/`, not canonical project files.

## Invariants

- Original-image compositing is the default: raw frames preserve pixels outside transformed
  masks exactly, except for deliberately filled motion-revealed holes; decoded H.264 output
  is checked separately for bounded codec noise.
- Prefer deterministic CV and local modification over generative video; reconstruction only
  fills holes revealed by motion.
- Fail closed: PRIMARY failures reject a run; SECONDARY/MICRO failures are isolated and
  dropped. A valid semantic STATIC result is not evidence of model failure.
- `analysis_mode="panel"` is the current default; use page mode explicitly when required.
- Do not assume parent/child plan links apply transforms automatically; inheritance is not
  implemented.

## Engineering rules

- Preserve [`docs/pipeline.md`](docs/pipeline.md), [`docs/animation-plan-schema.md`](docs/animation-plan-schema.md),
  and relevant accepted ADRs. If a boundary, invariant, schema, or default changes, update
  the appropriate current doc and ADR rather than letting them drift.
- Use matching specialist agents and skills in `.claude/agents/` and `.claude/skills/`.
  Agents report completion, blockers, and critical findings to the orchestrator via
  `SendMessage`; the orchestrating session owns final decisions and integration.
- Keep tests behavioral and focused on meaningful behavior or failure modes.

## Completion checks

Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` before declaring an
implementation complete. Synchronize any remote source changes back to this checkout via
git before considering the work finished.

## Documentation policy

Use concise investigation notes, document only meaningful engineering findings during
implementation, and write one concise final report per phase. Create an ADR only for a
meaningful architectural decision; preserve important negative experimental results. Do
not add documents or tests merely to increase counts. Update `docs/current-status.md` first
when a phase changes the current truth; keep phase reports historical.
