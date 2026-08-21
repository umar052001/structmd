"""Tests for structmd.config loading and merge precedence."""

from __future__ import annotations

import pytest
import yaml

from structmd.config import StructMDConfig, load_config


@pytest.fixture()
def isolated_cwd(tmp_path, monkeypatch):
    """Run the loader from an empty temp dir with no STRUCTMD_* env vars."""
    monkeypatch.chdir(tmp_path)
    for key in list(_env_keys()):
        monkeypatch.delenv(key, raising=False)
    # Neutralize any real global config on this machine.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return tmp_path


def _env_keys():
    import os

    return [k for k in os.environ if k.startswith("STRUCTMD_")]


class TestDefaults:
    def test_defaults(self) -> None:
        cfg = StructMDConfig()
        assert cfg.ollama_url == "http://localhost:11434"
        assert cfg.ollama_model == "qwen2-vl:2b"
        assert cfg.ollama_timeout == 120
        assert cfg.dpi == 150
        assert cfg.table_caption_position == "before"
        assert cfg.cache_dir == "~/.cache/structmd"


class TestYAMLLayers:
    def test_local_yaml_overrides_global(self, isolated_cwd) -> None:
        global_cfg = {"ollama": {"model": "smolvlm:2b", "url": "http://global:11434"}}
        local_cfg = {"ollama": {"model": "qwen2-vl:7b"}}
        (isolated_cwd / "home" / ".config" / "structmd").mkdir(parents=True)
        (isolated_cwd / "home" / ".config" / "structmd" / "config.yaml").write_text(
            yaml.safe_dump(global_cfg), encoding="utf-8"
        )
        (isolated_cwd / ".structmd.yaml").write_text(yaml.safe_dump(local_cfg), encoding="utf-8")

        cfg = load_config()
        assert cfg.ollama_model == "qwen2-vl:7b"  # local wins
        assert cfg.ollama_url == "http://global:11434"  # global fills gaps

    def test_nested_and_flat_keys_both_work(self, isolated_cwd) -> None:
        (isolated_cwd / ".structmd.yaml").write_text(
            yaml.safe_dump({"processing": {"dpi": 300}, "verbose": True}), encoding="utf-8"
        )
        cfg = load_config()
        assert cfg.dpi == 300
        assert cfg.verbose is True

    def test_explicit_config_path(self, isolated_cwd) -> None:
        explicit = isolated_cwd / "custom.yaml"
        explicit.write_text(yaml.safe_dump({"ollama": {"timeout": 42}}), encoding="utf-8")
        cfg = load_config(str(explicit))
        assert cfg.ollama_timeout == 42


class TestEnvVars:
    def test_env_overrides_yaml(self, isolated_cwd, monkeypatch) -> None:
        (isolated_cwd / ".structmd.yaml").write_text(
            yaml.safe_dump({"ollama": {"model": "smolvlm:2b", "dpi": 100}}), encoding="utf-8"
        )
        monkeypatch.setenv("STRUCTMD_OLLAMA_MODEL", "llama3.2-vision:11b")
        monkeypatch.setenv("STRUCTMD_DPI", "200")
        monkeypatch.setenv("STRUCTMD_VERBOSE", "true")

        cfg = load_config()
        assert cfg.ollama_model == "llama3.2-vision:11b"  # env beats yaml
        assert cfg.dpi == 200
        assert cfg.verbose is True
        assert cfg.ollama_timeout == 120  # untouched default

    def test_env_bool_coercion(self, isolated_cwd, monkeypatch) -> None:
        monkeypatch.setenv("STRUCTMD_DETECT_COLUMNS", "false")
        cfg = load_config()
        assert cfg.detect_columns is False


class TestRobustness:
    def test_missing_files_fall_back_to_defaults(self, isolated_cwd) -> None:
        cfg = load_config()
        assert cfg == StructMDConfig()

    def test_invalid_yaml_is_ignored(self, isolated_cwd) -> None:
        (isolated_cwd / ".structmd.yaml").write_text("not: [valid: yaml", encoding="utf-8")
        cfg = load_config()
        assert cfg == StructMDConfig()

    def test_unknown_keys_warn_but_do_not_crash(self, isolated_cwd) -> None:
        (isolated_cwd / ".structmd.yaml").write_text(
            yaml.safe_dump({"totally_unknown": 1, "dpi": 250}), encoding="utf-8"
        )
        cfg = load_config()
        assert cfg.dpi == 250
