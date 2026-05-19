from __future__ import annotations

import pytest

from lingya.config import LLMConfig
from lingya.llm.base import BaseLLMBackend
from lingya.llm.openai_compat import OpenAICompatBackend
from lingya.llm.factory import create_backend


class TestCreateBackend:
    def test_deepseek_returns_openai_compat(self):
        backend = create_backend(LLMConfig(provider="deepseek"))
        assert isinstance(backend, OpenAICompatBackend)
        assert isinstance(backend, BaseLLMBackend)

    def test_openai_returns_openai_compat(self):
        backend = create_backend(LLMConfig(provider="openai"))
        assert isinstance(backend, OpenAICompatBackend)

    def test_ollama_returns_openai_compat(self):
        backend = create_backend(LLMConfig(provider="ollama"))
        assert isinstance(backend, OpenAICompatBackend)

    def test_case_insensitive(self):
        backend = create_backend(LLMConfig(provider="DeepSeek"))
        assert isinstance(backend, OpenAICompatBackend)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_backend(LLMConfig(provider="unknown_provider"))
