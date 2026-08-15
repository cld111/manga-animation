"""python-3.10 compat: provides `datetime.UTC` for the official OMG-LLaVA worker env.

The phase-19 benchmark runs inside the official `omg_llava` python-3.10 venv (the stack is
pinned to 2024-era deps, transformers==4.36.0, python 3.10), but the manga-animation harness
modules use `datetime.UTC`, which only exists in python 3.11+. `datetime.timezone.utc` is
identical. Import this module BEFORE any `manga_animation` import (e.g. first line of the
phase-19 CLI) so `from datetime import UTC` resolves everywhere. Benchmark-harness-only, not
a production change.
"""

from __future__ import annotations

import datetime as _datetime

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]  # noqa: UP017
