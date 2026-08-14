# Documentation Guide

Documentation is divided by purpose. Current contracts must not be inferred from a
historical phase report.

- **Current:** `current-status.md`, `architecture.md`, `pipeline.md`,
  `animation-plan-schema.md`.
- **Decisions:** `decisions/` contains ADRs. A later revision or ADR supersedes an older
  choice; historical context remains valuable but is not the active contract.
- **Evidence:** `phase*-results.md` contains experiment logs, metrics, visual QA and
  limitations for a particular phase.
- **Entry point:** `README.md` links to the current status and operational commands.

When a new phase finishes, update `current-status.md` first, then add the detailed evidence
record. Do not copy the full evidence narrative into README or architecture documents.
