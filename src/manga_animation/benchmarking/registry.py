"""Loads the Phase 2 candidate shortlist from `configs/benchmark_candidates.yaml`.

Kept separate from `core/config.py`'s `PipelineConfig` deliberately: the candidate manifest
is benchmarking input data, not runtime pipeline configuration, and changes on a different
cadence (every time a new model is worth trying) than hardware profiles do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from manga_animation.benchmarking.schemas import ModelCandidate

DEFAULT_CANDIDATES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "benchmark_candidates.yaml"
)


def load_candidates(path: Path = DEFAULT_CANDIDATES_PATH) -> dict[str, list[ModelCandidate]]:
    """Load and validate the manifest, grouped by stage.

    Raises `FileNotFoundError` if `path` doesn't exist, and `pydantic.ValidationError` if an
    entry is malformed — both are meant to fail loudly rather than silently skip a bad entry.
    """
    if not path.exists():
        raise FileNotFoundError(f"no benchmark candidate manifest at {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping of stage -> candidate list")

    by_stage: dict[str, list[ModelCandidate]] = {}
    for stage, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"stage '{stage}' in {path} must map to a list of candidates")
        by_stage[stage] = [_parse_candidate(stage, entry) for entry in entries]
    return by_stage


def _parse_candidate(stage: str, entry: Any) -> ModelCandidate:
    if not isinstance(entry, dict):
        raise ValueError(f"candidate entry under stage '{stage}' must be a mapping, got {entry!r}")
    return ModelCandidate(stage=stage, **entry)


def flat_candidates(path: Path = DEFAULT_CANDIDATES_PATH) -> list[ModelCandidate]:
    """All candidates across all stages, as a single flat list."""
    return [c for candidates in load_candidates(path).values() for c in candidates]
