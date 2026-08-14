# Documentation Guide

The documentation is intentionally split by reading frequency. A fresh session should
normally read `CLAUDE.md`, then `current-status.md`, then the current task. Historical
reports and ADRs are opened only when the task needs their evidence or decision rationale.
Current contracts must not be inferred from a historical phase report.

## Canonical sources

| Information | Canonical source | Use |
|---|---|---|
| Permanent operating rules | [`../CLAUDE.md`](../CLAUDE.md) | Every session |
| Current pipeline, defaults, capabilities, limitations, metrics, and priorities | [`current-status.md`](current-status.md) | Current project truth |
| Stable engineering principles and invariants | [`architecture.md`](architecture.md) | Design and implementation review |
| Stage order, ownership, lifecycle, and safety contracts | [`pipeline.md`](pipeline.md) | Stage changes and orchestration |
| Animation Plan contract | [`animation-plan-schema.md`](animation-plan-schema.md) | Schema and motion-plan work |
| Architectural decisions | [`decisions/`](decisions/) | Decision rationale and supersession history |
| Remote Kaggle Jupyter workflow | [`kaggle-jupyter.md`](kaggle-jupyter.md) | Verified connection/execution/watchdog procedure for the remote GPU worker |
| Experiment evidence | `phase*-results.md` | Historical provenance, metrics, and negative results |

ADR status is authoritative within each ADR. ADR 0004 is superseded by ADR 0005; ADR 0005
is the preliminary operational model baseline, not an exhaustive selection conclusion.
Later accepted ADRs refine the active architecture, while their phase-specific evidence
remains historical.

## Maintenance policy

- During investigation, use concise research notes.
- During implementation, document only meaningful engineering findings.
- At phase completion, create one concise final phase report.
- Use an ADR only for a meaningful architectural decision.
- Add regression tests for meaningful behavior and failure modes, not to increase the count.
- Preserve important negative results when they prevent repeated investigation.

When a phase changes current truth, update `current-status.md` first, then add its detailed
evidence record. Do not copy full evidence narratives into README, architecture, pipeline,
or the current-status document.
