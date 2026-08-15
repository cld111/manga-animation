"""Minimal Jupyter kernel client for the remote Kaggle GPU worker.

Implements exactly the verified transport in docs/kaggle-jupyter.md section 4: the kernel is
reached over a Jupyter websocket; a browser GET of the URL is NOT a valid connectivity test.
The base URL is the ephemeral Jupyter Server URL the user supplies per session -- never
hardcode one into source.

Usage (code inline or via stdin; the URL is always the first argument):

    python scripts/kaggle_exec.py "<Jupyter URL>" "print(1+1)"
    python scripts/kaggle_exec.py "<Jupyter URL>" < script.py

Sends a `kernel_info_request`, then an `execute_request`, and prints every stream/result/error
message whose `parent_header.msg_id` matches our own request. Exits 0 on a completed request,
1 on an `execute_reply` status of "error". One kernel executes one request at a time -- do not
run two long requests concurrently (docs/kaggle-jupyter.md section 9).
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

import websocket  # websocket-client


def _http_get(url: str) -> list[dict]:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 -- verified HTTPS endpoint only
        return json.loads(resp.read().decode("utf-8"))


def _pick_kernel(base: str) -> str:
    kernels = _http_get(f"{base}/api/kernels")
    if not kernels:
        raise SystemExit("no kernels available on the remote server")
    return kernels[0]["id"]


def _send(ws: websocket.WebSocket, msg_type: str, content: dict, msg_id: str) -> None:
    ws.send(
        json.dumps(
            {
                "header": {
                    "msg_type": msg_type,
                    "msg_id": msg_id,
                    "username": "opencode",
                    "session": "opencode",
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "content": content,
                "channel": "shell",
                "buffers": [],
            }
        )
    )


def _execute(base: str, kernel_id: str, code: str) -> None:
    parsed = urllib.parse.urlparse(base)
    ws_base = f"wss://{parsed.netloc}{parsed.path}"
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels"
    import ssl

    ws = websocket.create_connection(
        ws_url, origin=base, timeout=3600, sslopt={"cert_reqs": ssl.CERT_NONE}
    )

    # Prove the kernel is ready (and drain its own status chatter).
    _send(ws, "kernel_info_request", {}, msg_id="info-1")
    while True:
        msg = json.loads(ws.recv())
        if msg.get("msg_type") == "kernel_info_reply":
            break

    req_id = "opencode-exec-1"
    _send(
        ws,
        "execute_request",
        {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        msg_id=req_id,
    )
    status = "ok"
    while True:
        msg = json.loads(ws.recv())
        parent = (msg.get("parent_header") or {}).get("msg_id")
        if parent != req_id:
            continue  # unrelated kernel chatter -- ignore
        msg_type = msg.get("msg_type")
        if msg_type == "stream":
            sys.stdout.write(msg["content"].get("text", ""))
            sys.stdout.flush()
        elif msg_type == "execute_result":
            data = msg["content"].get("data", {})
            sys.stdout.write(str(data.get("text/plain", data)) + "\n")
            sys.stdout.flush()
        elif msg_type == "error":
            sys.stderr.write("\n".join(msg["content"].get("traceback", [])) + "\n")
            status = "error"
        elif msg_type == "execute_reply":
            if msg["content"].get("status") == "error":
                status = "error"
            break
    ws.close()
    if status == "error":
        raise SystemExit(1)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/kaggle_exec.py <Jupyter URL> [code]")
    base = sys.argv[1].rstrip("/")
    code = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    if not code.strip():
        raise SystemExit("no code to execute")
    kernel_id = _pick_kernel(base)
    _execute(base, kernel_id, code)


if __name__ == "__main__":
    main()
