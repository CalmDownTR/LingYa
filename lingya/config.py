from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 32768
    max_input_tokens: int = 1_048_576  # DeepSeek V4 Flash 1M context window, for summarization thresholds


class OtelConfig(BaseModel):
    enabled: bool = False


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    otel: OtelConfig = OtelConfig()
    db_path: str = "./data/lingya.db"
    memory_path: str = "./data/memory"
    data_dir: str = "./data"
    persona_config_path: str = "agent_config.yaml"
    diary_period_days: int = 1
    auth_enabled: bool = False  # Set to True + LINGYA_API_KEY env var for production


def load_config(path: str | None = None) -> Config:
    if path is None:
        path = os.environ.get("LINGYA_CONFIG", "config.yaml")

    data: dict = {}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Overlay env vars
    env_map = {
        "llm_provider": ("llm", "provider"),
        "llm_model": ("llm", "model"),
        "llm_api_base_url": ("llm", "api_base_url"),
        "llm_max_tokens": ("llm", "max_tokens"),
        "db_path": ("db_path",),
        "data_dir": ("data_dir",),
    }

    for env_key, cfg_path in env_map.items():
        val = os.environ.get(f"LINGYA_{env_key.upper()}")
        if val is not None:
            d = data
            for key in cfg_path[:-1]:
                d = d.setdefault(key, {})
            if env_key == "llm_max_tokens":
                d[cfg_path[-1]] = int(val)
            else:
                d[cfg_path[-1]] = val

    # Handle api_key_env
    if os.environ.get("LINGYA_API_KEY"):
        data.setdefault("llm", {})["api_key_env"] = "LINGYA_API_KEY"
    elif os.environ.get("DEEPSEEK_API_KEY"):
        data.setdefault("llm", {})["api_key_env"] = "DEEPSEEK_API_KEY"

    return Config.model_validate(data)
