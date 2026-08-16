# Working With the Remote Kaggle Jupyter Server

Operational guide: how this project actually reaches a remote Kaggle GPU (Tesla T4) from a
local Claude/OpenCode session. This documents the **verified** transport only — every
endpoint, command, and expected output below was exercised against a live Kaggle session
during Phase 15. Do not substitute an unverified mechanism for it.

## 1. How the server is provided

Kaggle gives an ephemeral **Jupyter Server URL** for an interactive GPU session. Two facts
are non-negotiable:

- The URL **changes between sessions**. The user supplies the current URL when GPU work is
  required. Never hardcode it into source, git, config, or docs; never assume the previous
  session's URL still works.
- The provided URL is a **Jupyter/VSCode-compatible connection URL**, not a web page.
  A browser `GET` returning `404` does **not** mean the server is down. The definitive test
  is Jupyter-protocol communication and code execution on the kernel.

The URL has the shape:

```text
https://kkb-production.jupyter-proxy.kaggle.net/k/<session_id>/<long_jwt_token>/proxy
```

## 2. Remote execution model

```text
LOCAL MACHINE (Claude/OpenCode)
    │  HTTPS (Jupyter REST API)   +   WSS (kernel websocket)
    ▼
REMOTE KAGGLE JUPYTER SERVER (ephemeral)
    │
    ▼
REMOTE KERNEL (Python 3.12, torch, 2× Tesla T4)
```

There is **no SSH** and **no Kaggle CLI** in the verified workflow. Confusing these with the
Jupyter transport is the most common error:

| Transport | What it actually is | Used here? |
|---|---|---|
| Browser `GET` of the URL | HTTP page fetch | No — not a valid connectivity test |
| Jupyter REST API | `https://…/proxy/api/…` | Yes — status/kernels listing |
| Jupyter kernel websocket | `wss://…/proxy/api/kernels/<id>/channels` | Yes — actual code execution |
| MCP Jupyter server | a configured MCP tool | Not configured; do not invent one |
| SSH / Kaggle API | different access paths | Not used |

## 3. Verifying the connection (do this first)

Ask the user for the current URL, then confirm the server answers the Jupyter REST API:

```bash
curl -s -k "<URL>/api/status"
# -> {"connections": 1, "kernels": 1, "last_activity": "...", ...}
curl -s -k "<URL>/api/kernels"
# -> [{"id": "...", "name": "python3", "execution_state": "idle", ...}]
```

Then the definitive check — **execute Python on the kernel** via its websocket channels.
A minimal Jupyter client (see §4) that sends a `kernel_info_request` and then an
`execute_request` for:

```python
import platform, torch, subprocess
print(subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout.strip())
print(platform.python_version())
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

Verified output on the Phase 15 worker:

```text
<container_hostname>
3.12.13
2.10.0+cu128
True
2
['Tesla T4', 'Tesla T4']
```

Only this counts as "connected". A `404` from a browser `GET` means nothing.

## 4. Executing code on the remote kernel (the verified client)

There is no Jupyter MCP server in this project. The verified mechanism is a small local
Python client using `websocket-client` + `urllib`:

1. **List kernels** `GET <URL>/api/kernels`; pick the first `"python3"` kernel (`id`).
   (Creating a new one via `POST /api/kernels` also works — returns `201`.)
2. **Connect a websocket** to `wss://<host>/k/<sid>/<token>/proxy/api/kernels/<id>/channels`
   (convert `https://` → `wss://`, keep the `/proxy` path).
3. **Send `kernel_info_request`** on the `shell` channel; wait for `kernel_info_reply`
   (proves the kernel is ready).
4. **Send `execute_request`** on the `shell` channel with `{"code": "...", "silent": False,
   "store_history": True, "allow_stdin": False, "stop_on_error": True}`.
5. **Read messages** until a `status` message with `execution_state == "idle"`:
   - `msg_type == "stream"` → kernel stdout/stderr (`content.text`)
   - `msg_type == "execute_result"` → rich result (`content.data["text/plain"]`)
   - `msg_type == "error"` → `content.ename` / `content.evalue`
   - filter replies by `parent_header.msg_id` to ignore unrelated kernel chatter.

A reusable client that implements exactly this lives in this repo's working tooling as
`scripts/kaggle_exec.py`-style helper (base URL passed as argv; code to run passed as the
second argument or via base64). Everything in §5–§7 below was run through it.

## 5. Running repository commands on the remote worker

The remote checkout lives at `/kaggle/working/manga-animation`. Run git/Python commands by
sending a Python snippet that shells out with `subprocess.run`, e.g.:

```python
import subprocess
r = subprocess.run("cd /kaggle/working/manga-animation && git branch --show-current && git rev-parse --short HEAD && git status --short",
                   shell=True, capture_output=True, text=True, timeout=600)
print(r.stdout); print(r.stderr)
```

Verified commands that work remotely:

```bash
git branch --show-current            # phase-15-gpu-regression-stability
git rev-parse --short HEAD           # d605d84 (must match local HEAD)
git status --short                   # clean
python scripts/fetch_phase9_realworld_pages.py   # downloads pages to examples/realworld/
nvidia-smi --query-gpu=name,memory.total --format=csv
```

You know it ran on Kaggle, not locally, because the output carries the remote host/GPU/commit.

## 6. Synchronizing the remote checkout with local git

- The remote repo is a **git clone** of the local canonical repo
  (`https://github.com/cld111/manga-animation`), checked out in `/kaggle/working/manga-animation`.
  Files are **not** hand-copied (ADR 0002/0003).
- Select the working branch and pin the commit:
  ```bash
  cd /kaggle/working/manga-animation
  git fetch origin
  git checkout -B <branch> origin/<branch>   # or git checkout <branch>
  git pull --ff-only origin <branch>
  git rev-parse --short HEAD                 # must equal the local commit
  ```
- **Pre-flight:** always compare `git rev-parse --short HEAD` between local and remote before
  a GPU run. A GPU experiment must never run against a stale commit.

## 7. Model weights and example pages on the worker

Neither models nor example pages are committed (git-ignored). On a fresh session they must be
re-created on the worker:

```bash
# pages (into /kaggle/working/manga-animation/examples/realworld/)
python scripts/fetch_phase9_realworld_pages.py

# models (Phase 14 layout: local dirs, no HF cache needed at runtime)
mkdir -p /kaggle/working/models
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-8B-Instruct", local_dir="/kaggle/working/models/qwen")
snapshot_download("IDEA-Research/grounding-dino-base", local_dir="/kaggle/working/models/dino")
snapshot_download("facebook/sam2.1-hiera-base-plus", local_dir="/kaggle/working/models/sam")
PY
```

Verify: `ls /kaggle/working/models/*/config.json` and
`ls /kaggle/working/manga-animation/examples/realworld/*.png`.

Python deps come from `pip install -e "/kaggle/working/manga-animation[ml]"` (torch
2.10.0+cu128, transformers 5.0.0, simple-lama-inpainting, accelerate, editable install of the
repo). This was already present on the Phase 15 worker and only needs re-installing on a
fresh session.

## 8. Launching GPU pipeline scripts

The production entry points are under `scripts/` and are run on the remote worker (never
locally — real model inference is remote-GPU work). Verified pattern:

```bash
cd /kaggle/working/manga-animation
python scripts/run_phase15_gpu_regression.py \
    --pages examples/realworld/villainess_ending_scuffle.png ... \
    --qwen /kaggle/working/models/qwen \
    --dino /kaggle/working/models/dino \
    --sam /kaggle/working/models/sam \
    --out outputs/experiments/phase15_<ts>.json
```

Outputs (videos, frames, manifests, experiment JSON) land under the git-ignored
`outputs/` tree on the worker; source is never modified by a run.

## 9. Monitoring long-running GPU jobs

Jupyter kernels are not a job queue; a foreground `execute_request` occupies the one kernel.
For a multi-page run, use the **no-screen → background process + log** pattern so the kernel
stays responsive and the job survives local disconnects:

```bash
nohup python scripts/run_phase15_gpu_regression.py --pages ... --out outputs/experiments/run.json \
    > outputs/experiments/run.log 2>&1 &
echo $! > outputs/experiments/run.pid
```

Monitor from the local client by polling the log and process table:

```bash
ps aux | grep run_phase15      # still running?
tail -50 outputs/experiments/run.log
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

**Avoid duplicate launches:** check `run.pid` / `ps` for the same script before starting, and
name each experiment output distinctly. Because the kernel accepts one execution at a time,
do not send two long `execute_request`s concurrently.

To run two independent heavy jobs, do **not** rely on two kernel connections on one T4 pool;
serialize them (Phase 15 rule: don't trade VRAM stability for artificial parallelism).

## 10. GPU diagnostics

`nvidia-smi` (system-level) and `torch.cuda` (allocator-level) answer different questions:

- `nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv` — **process
  memory**: what the GPU processes actually hold (includes CUDA context and cached blocks).
- `torch.cuda.memory_allocated(i)` — **allocated**: live tensors in the caching allocator.
- `torch.cuda.memory_reserved(i)` — **reserved**: the allocator's pool, possibly larger than
  allocated (freed blocks are reused, not returned to the driver).
- `torch.cuda.max_memory_allocated(i)` — **peak** allocated since process start (use
  `reset_peak_memory_stats()` between phases).
- `torch.cuda.memory_free()` / `torch.cuda.mem_get_info()` — actual free / total device
  memory (the true headroom the driver reports).

In this project the run-level lifecycle (ADR 0021) keeps every model resident for the whole
run and releases them all together at the end, so `allocated` staying high between stages
while `nvidia-smi` shows the CUDA context is expected and correct; it drops back to ~9 MB
after the run finishes.

**CUDA OOM detection:** a real OOM surfaces as `torch.OutOfMemoryError: CUDA out of memory`
in the job log. Check the log, not just exit codes.

## 11. Watchdog: keeping the ephemeral server alive

Kaggle shuts down an idle Jupyter session. Project rule (Phase 15):

> Ping/access the Jupyter Server approximately every **20 minutes** while any work is
> in progress — local analysis, code writing, waiting on GPU jobs, analyzing results.

A minimal watchdog loop (plain `curl` to `<URL>` every 1200 s, appending to a log) is the
verified mechanism. The URL is ephemeral — when the user provides a new one, update the
watchdog's URL. If the server becomes unreachable: continue local work, do **not** claim GPU
validation succeeded, retry the connection, and ask for a fresh URL when GPU work must resume.

## 12. Recovery after kernel restart / session loss

After any kernel restart or new session, **all in-process state is gone**. Re-establish in
this order:

1. New URL from the user; update the watchdog.
2. Re-verify the connection (§3).
3. Re-check the remote checkout commit (§6) — a new session is a fresh clone, so re-run
   `git fetch`/`git checkout`/`git pull` and compare `HEAD`.
4. Re-create worker-local artifacts if missing: pages (§7), model dirs (§7), `pip install -e
   ".[ml]"`.
5. Re-launch any interrupted GPU job (§9); resumable pipelines pick up from their manifest.
6. A lost websocket mid-execution means the `execute_request` result may be lost; re-run the
   command (idempotent resumability makes this safe for pipeline runs).

Distinguish failure classes when reporting: HTTP endpoint failure vs. authentication vs.
websocket/kernel connection vs. expired session vs. unavailable kernel. The REST 404 from a
browser `GET` is **not** a kernel failure (§1).

## 13. Pre-flight checklist (before any real GPU run)

```
[ ] User supplied the current Jupyter URL
[ ] Jupyter REST API answers (<URL>/api/status -> 200)
[ ] Remote Python execution confirmed (§3)
[ ] Correct repository: /kaggle/working/manga-animation
[ ] Correct branch selected
[ ] Correct commit: remote HEAD == local HEAD
[ ] Working tree state verified (git status --short)
[ ] Python environment: manga-animation editable + torch + transformers present
[ ] CUDA available: torch.cuda.is_available() == True
[ ] Expected GPU count verified (Phase 15: 2× Tesla T4)
[ ] Models present: /kaggle/working/models/{qwen,dino,sam}/config.json
[ ] Pages present: examples/realworld/*.png
[ ] Output directory writable (outputs/experiments, outputs/videos)
[ ] No duplicate GPU job already running (check run.pid / ps)
[ ] Watchdog active (pings every ~20 min)
```
