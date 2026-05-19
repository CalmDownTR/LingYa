from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lingya.config import Config, load_config


class TestConfigDefaults:
    def test_default_llm_provider(self):
        cfg = Config()
        assert cfg.llm.provider == "deepseek"

    def test_default_memory_settings(self):
        cfg = Config()
        assert cfg.memory.short_term_max_messages == 20
        assert cfg.memory.long_term_top_k == 5
        assert cfg.memory.compression_enabled is True

    def test_default_personality_settings(self):
        cfg = Config()
        assert cfg.personality.reflection_interval_turns == 10


class TestLoadConfig:
    def test_defaults_when_no_file(self, monkeypatch):
        # Ensure no config file or env vars interfere
        monkeypatch.delenv("LINGYA_CONFIG", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("LINGYA_API_KEY", raising=False)
        monkeypatch.delenv("LINGYA_LLM_PROVIDER", raising=False)
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.llm.provider == "deepseek"

    def test_loads_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"llm": {"provider": "openai", "model": "gpt-4"}}, f)
            tmp_path = f.name

        try:
            cfg = load_config(tmp_path)
            assert cfg.llm.provider == "openai"
            assert cfg.llm.model == "gpt-4"
        finally:
            Path(tmp_path).unlink()

    def test_env_overlay_provider(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  provider: deepseek\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv("LINGYA_LLM_PROVIDER", "openai")

        cfg = load_config()
        assert cfg.llm.provider == "openai"

    def test_env_overlay_model(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  model: deepseek-chat\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv("LINGYA_LLM_MODEL", "gpt-4o")

        cfg = load_config()
        assert cfg.llm.model == "gpt-4o"

    def test_env_overlay_max_tokens_as_int(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  max_tokens: 100\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv("LINGYA_LLM_MAX_TOKENS", "4096")

        cfg = load_config()
        assert cfg.llm.max_tokens == 4096
        assert isinstance(cfg.llm.max_tokens, int)

    def test_env_overlay_db_path(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("db_path: ./data/lingya.db\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv("LINGYA_DB_PATH", "/custom/path/db.sqlite")

        cfg = load_config()
        assert cfg.db_path == "/custom/path/db.sqlite"

    def test_lingya_api_key_sets_api_key_env(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  provider: deepseek\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv("LINGYA_API_KEY", "sk-test-key")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        cfg = load_config()
        assert cfg.llm.api_key_env == "LINGYA_API_KEY"

    def test_deepseek_api_key_fallback(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  provider: deepseek\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.delenv("LINGYA_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-key")

        cfg = load_config()
        assert cfg.llm.api_key_env == "DEEPSEEK_API_KEY"

    def test_lingya_config_env_var(self, monkeypatch, tmp_path):
        config_path = tmp_path / "custom.yaml"
        config_path.write_text("llm:\n  provider: ollama\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))

        cfg = load_config()
        assert cfg.llm.provider == "ollama"
