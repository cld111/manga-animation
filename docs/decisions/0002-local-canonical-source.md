# 2. The local project is the canonical source of truth

Status: Accepted

## Context

Model development/inference will at times require GPU compute this project's primary local
machine doesn't have (see [0003](0003-remote-compute-workers.md) — no NVIDIA/CUDA locally).
That compute will come from remote, ephemeral Kaggle/Jupyter sessions. Those sessions are
not durable: they can time out, hit quota limits, or simply be closed, and nothing
guarantees the working directory on such a server survives between sessions.

If the project's code, docs, schema, tests, or `.claude/` agents/skills only existed on a
remote server at some point in time, that server disappearing would be a real loss of work,
not just an inconvenience.

## Decision

The local checkout is always the complete, canonical copy of the project — code, tests,
configuration, documentation, and `.claude/agents` / `.claude/skills`. Git is the only
sanctioned channel for moving source changes between local and remote:

```text
local:  edit → commit → push
remote: pull → run GPU-bound experiments → commit/push only if source changed
local:  pull
```

Remote sessions are treated as workers that check out the repo, run something, and
optionally push results back — never as a place where uncommitted, local-only changes are
allowed to accumulate. Manual file copying between local and remote is avoided in favor of
git so history stays legible and nothing is silently overwritten in either direction.

## Consequences

- No task in this project should ever require "the remote session" to still exist for the
  project to be understandable or continuable — losing a remote session should cost at most
  the uncommitted work done in that session.
- Large binary artifacts produced remotely (model weights, rendered videos, frame dumps)
  are *not* pulled back into git (see `.gitignore`) — they're regenerable from source, so
  their loss on a remote session's expiry is not a canonicity problem.
- If local and remote both changed the same files without syncing, that's a merge conflict
  to resolve deliberately, not something to paper over by picking whichever copy is more
  convenient at the time.
