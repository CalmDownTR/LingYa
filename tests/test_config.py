from __future__ import annotations

import pytest

from lingya.config import Config, load_config


class TestConfigDefaults:
    def test_default_llm_model(self):
        cfg = Config()
        assert cfg.llm.model == "deepseek/deepseek-v4-flash"

class TestLoadConfig:
    def test_defaults_when_no_file(self, monkeypatch):
        for var in ("LINGYA_CONFIG", "DEEPSEEK_API_KEY", "LINGYA_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.llm.model == "deepseek/deepseek-v4-flash"

    def test_loads_from_yaml_file(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm:\n  model: openai/gpt-4\n")

        cfg = load_config(str(config_path))
        assert cfg.llm.model == "openai/gpt-4"

    @pytest.mark.parametrize(
        "env_var,env_value,attr,expected",
        [
            ("LINGYA_LLM_MODEL", "openai/gpt-4o", "model", "openai/gpt-4o"),
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

    def test_lingya_config_env_var(self, monkeypatch, tmp_path):
        config_path = tmp_path / "custom.yaml"
        config_path.write_text("llm:\n  model: ollama/llama3\n")
        monkeypatch.setenv("LINGYA_CONFIG", str(config_path))

        cfg = load_config()
        assert cfg.llm.model == "ollama/llama3"
