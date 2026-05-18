from __future__ import annotations

from lingya.config import LLMConfig

from .base import BaseLLMBackend
from .openai_compat import OpenAICompatBackend


def create_backend(config: LLMConfig) -> BaseLLMBackend:
    provider = config.provider.lower()
    if provider in ("deepseek", "openai", "ollama"):
        return OpenAICompatBackend(config)
    raise ValueError(f"Unknown LLM provider: {provider}")
