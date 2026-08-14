from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_animation.core.config import PipelineConfig, load_config


def test_defaults_are_valid():
    cfg = PipelineConfig()
    assert cfg.device == "auto"
    assert cfg.fps == 24
    assert cfg.duration_s == pytest.approx(4.0)


@pytest.mark.parametrize(
    "field,value",
    [("fps", 0), ("fps", -5), ("resolution", 0), ("duration_s", 0.0)],
)
def test_non_positive_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        PipelineConfig(**{field: value})


def test_unknown_device_literal_is_rejected():
    with pytest.raises(ValidationError):
        PipelineConfig(device="tpu")


def test_resolve_device_returns_explicit_choice_without_importing_torch():
    cfg = PipelineConfig(device="cuda")
    assert cfg.resolve_device() == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_torch_unavailable(monkeypatch):
    # Phase 1 does not depend on torch; "auto" must degrade gracefully rather than crash.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch in this environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = PipelineConfig(device="auto")
    assert cfg.resolve_device() == "cpu"


def test_load_config_default_matches_repo_configs_default_yaml():
    cfg = load_config()
    assert cfg.fps == 24
    assert cfg.output_codec == "h264"


def test_load_config_layers_environment_profile_over_default():
    base = load_config()
    local = load_config("local")
    # local.yaml only overrides a subset; unspecified fields must still come from default.yaml
    assert local.output_codec == base.output_codec == "h264"
    assert local.seed == base.seed == 42
    # fields local.yaml does override must win
    assert local.resolution == 1024
    assert local.debug is True
    assert base.debug is False


def test_load_config_unknown_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist")


def test_load_config_explicit_overrides_win_over_profile():
    cfg = load_config("local", overrides={"resolution": 42})
    assert cfg.resolution == 42
