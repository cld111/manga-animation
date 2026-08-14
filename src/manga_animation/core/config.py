"""Pipeline configuration.

Every hardware- or environment-sensitive parameter (device, dtype, resolution, model
variant...) belongs here, loaded from YAML, and never hardcoded in pipeline stage code. This
is what lets the same codebase run unchanged on a local Apple
Silicon machine and on a remote Kaggle T4/L4 worker — see docs/architecture.md
("Remote Compute Is Disposable" / "Model Abstraction").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

Device = Literal["auto", "cpu", "cuda", "mps"]
DType = Literal["float32", "float16", "bfloat16"]
Codec = Literal["h264"]


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    `model_variants` deliberately stays an open string->string map: which key names
    (e.g. "vlm", "grounding", "segmentation") exist is decided in Phase 2 once models are
    benchmarked, and this config must not need a schema change to record that choice.
    """

    device: Device = "auto"
    dtype: DType = Field(
        default="float32",
        description="VLM model dtype; grounding and segmentation use verified float32 clients.",
    )
    model_variants: dict[str, str] = Field(default_factory=dict)

    resolution: int = Field(
        gt=0, default=1536, description="Max VLM analysis long-edge resolution, in pixels."
    )
    fps: int = Field(gt=0, le=60, default=24)
    duration_s: float = Field(gt=0.0, le=30.0, default=4.0)

    output_codec: Codec = "h264"
    seed: int = 42

    enable_semantic_mask_validation: bool = Field(
        default=True,
        description=(
            "Phase 12: run the post-segmentation semantic mask validation gate "
            "(validation.mask_semantics.verify_mask_semantics) between segmentation and "
            "animation. Defaults on, per this project's 'a clean honest REJECTED is preferable "
            "to a visually corrupted PASS' precedent (docs/phase11-results.md section 7) -- the "
            "gate exists specifically to catch the real, confirmed Phase 11 failure mode "
            "(semantically over-inclusive real SAM masks that pass every existing geometric "
            "check). Exposed as a config toggle rather than hardcoded so a caller that has "
            "already characterized this gate's real false-rejection rate for its own dataset "
            "can disable it deliberately -- see docs/decisions/0018-semantic-mask-validation.md."
        ),
    )
    def resolve_device(self) -> Literal["cpu", "cuda", "mps"]:
        """Resolve "auto" to a concrete device, without requiring torch to be installed."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping at the top level")
    return data


def load_config(
    env: str | None = None,
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load `default.yaml`, optionally layered with `{env}.yaml`, then explicit overrides.

    `env` selects a hardware profile, e.g. "local" or "kaggle" (see configs/). Layering
    keeps environment-specific files small — they only need to state what differs from the
    default.
    """
    merged: dict[str, Any] = {}
    default_path = config_dir / "default.yaml"
    if default_path.exists():
        merged = _load_yaml(default_path)

    if env and env != "default":
        env_path = config_dir / f"{env}.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"no config profile named '{env}' at {env_path}")
        merged = _deep_merge(merged, _load_yaml(env_path))

    if overrides:
        merged = _deep_merge(merged, overrides)

    return PipelineConfig(**merged)
