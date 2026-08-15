"""python-3.10 compat: provides `datetime.UTC` and `enum.StrEnum` for the OMG-LLaVA worker env.

The phase-19 benchmark runs inside the official `omg_llava` python-3.10 venv (the stack is
pinned to 2024-era deps: xtuner requires python<3.11, transformers==4.36.0), but the
manga-animation harness modules use `datetime.UTC` and `enum.StrEnum`, which only exist in
python 3.11+. Import this module BEFORE any `manga_animation` import (e.g. first line of the
phase-19 CLI) so `from datetime import UTC` / `from enum import StrEnum` resolve everywhere.
The StrEnum shim covers the harness's usage (explicit string values, `__str__` == value).
Benchmark-harness-only, not a production change.
"""

from __future__ import annotations

import datetime as _datetime
import enum as _enum

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]  # noqa: UP017

if not hasattr(_enum, "StrEnum"):

    class _StrEnum(str, _enum.Enum):  # noqa: UP042 -- intentional py3.10 StrEnum stand-in
        """Minimal python-3.10 stand-in for `enum.StrEnum` (explicit-value members)."""

        def __str__(self) -> str:  # noqa: D105 -- mirrors StrEnum's value-as-str
            return str(self.value)

        def __repr__(self) -> str:  # noqa: D105 -- mirrors StrEnum's value-as-str
            return str(self.value)

    _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]
