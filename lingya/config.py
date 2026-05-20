from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 32768
    model_context_window: int = 128000


class MemoryConfig(BaseModel):
    short_term_max_messages: int = 100  # Hard cap safety guard — NOT the primary context management trigger
    long_term_top_k: int = 5
    chroma_persist_dir: str = "./data/chroma"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    compression_enabled: bool = True
    max_context_tokens: int = 80000
    compression_trigger_ratio: float = 0.75
    max_agent_iterations: int = 5


class PersonalityConfig(BaseModel):
    reflection_interval_turns: int = 10
    seed_personality: str = ""


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    personality: PersonalityConfig = PersonalityConfig()
    db_path: str = "./data/lingya.db"
    data_dir: str = "./data"


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
            # Keep value type-coerced
            if env_key == "llm_max_tokens":
                d[cfg_path[-1]] = int(val)
            else:
                d[cfg_path[-1]] = val

    # Handle api_key_env specifically for LLM
    if os.environ.get("LINGYA_API_KEY"):
        data.setdefault("llm", {})["api_key_env"] = "LINGYA_API_KEY"
    elif os.environ.get("DEEPSEEK_API_KEY"):
        data.setdefault("llm", {})["api_key_env"] = "DEEPSEEK_API_KEY"

    return Config.model_validate(data)
