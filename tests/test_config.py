from __future__ import annotations

import pytest

from lingya.config import Config, load_config


class TestConfigDefaults:
    def test_default_llm_provider(self):
        cfg = Config()
        assert cfg.llm.provider == "deepseek"

    def test_default_personality_settings(self):
        cfg = Config()
        assert cfg.personality.reflection_interval_turns == 10


class TestLoadConfig:
    def test_defaults_when_no_file(self, monkeypatch):
        for var in ("LINGYA_CONFIG", "DEEPSEEK_API_KEY", "LINGYA_API_KEY", "LINGYA_LLM_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.llm.provider == "deepseek"

    def test_loads_from_yaml_file(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  provider: openai\n  model: gpt-4\n")

        cfg = load_config(str(config_path))
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4"

    @pytest.mark.parametrize(
        "env_var,env_value,attr,expected",
        [
            ("LINGYA_LLM_PROVIDER", "openai", "provider", "openai"),
            ("LINGYA_LLM_MODEL", "gpt-4o", "model", "gpt-4o"),
        ],
    )
    def test_env_overlay_string_fields(self, monkeypatch, tmp_path, env_var, env_value, attr, expected):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"llm:\n  {attr}: default-value\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.setenv(env_var, env_value)

        cfg = load_config()
        assert getattr(cfg.llm, attr) == expected

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

    @pytest.mark.parametrize(
        "lingya_key,deepseek_key,expected_env",
        [
            ("sk-lingya", None, "LINGYA_API_KEY"),
            (None, "sk-deepseek", "DEEPSEEK_API_KEY"),
        ],
    )
    def test_api_key_env_selection(
        self, monkeypatch, tmp_path, lingya_key, deepseek_key, expected_env
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  provider: deepseek\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))
        monkeypatch.delenv("LINGYA_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        if lingya_key:
            monkeypatch.setenv("LINGYA_API_KEY", lingya_key)
        if deepseek_key:
            monkeypatch.setenv("DEEPSEEK_API_KEY", deepseek_key)

        cfg = load_config()
        assert cfg.llm.api_key_env == expected_env

    def test_lingya_config_env_var(self, monkeypatch, tmp_path):
        config_path = tmp_path / "custom.yaml"
        config_path.write_text("llm:\n  provider: ollama\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))

        cfg = load_config()
        assert cfg.llm.provider == "ollama"
