"""Tests for lingya.protocols — IMemoryStore and ILLMBackend."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest


class TestIMemoryStore:
    """IMemoryStore Protocol — 12 methods covering all callers."""

    def test_enhanced_memory_store_isinstance(self):
        """EnhancedMemoryStore should pass isinstance check."""
        from lingya.protocols import IMemoryStore
        from lingya.memory import EnhancedMemoryStore

        store = EnhancedMemoryStore(persist_path=":memory:")
        assert isinstance(store, IMemoryStore)

    def test_mock_passes_when_all_methods_present(self):
        """A Mock with all 12 methods should pass isinstance."""
        from lingya.protocols import IMemoryStore

        class MockMemory:
            def warmup(self) -> None:
                pass

            def store(self, text: str) -> str:
                return "mem_1"

            def search(self, query: str, top_k: int = 3) -> list[dict]:
                return []

            def list_all(self) -> list[dict]:
                return []

            def delete(self, mem_id: str) -> bool:
                return True

            def recover(self, mem_id: str) -> bool:
                return True

            def store_with_importance(self, text: str, importance: float) -> str:
                return "mem_1"

            async def score_importance(
                self, text: str, llm_call: Callable[[str], Awaitable[str]]
            ) -> float:
                return 5.0

            def update_importance(self, entry_id: str, importance: float) -> None:
                pass

            def search_weighted(
                self, query: str, top_k: int = 5, recency_lambda: float = 0.01
            ) -> list[dict]:
                return []

            def apply_decay(self) -> int:
                return 0

            def get_cumulative_importance(self) -> float:
                return 0.0

        assert isinstance(MockMemory(), IMemoryStore)

    def test_mock_fails_when_method_missing(self):
        """A class missing a required method should fail isinstance."""
        from lingya.protocols import IMemoryStore

        class IncompleteMemory:
            def store(self, text: str) -> str:
                return "mem_1"

        assert not isinstance(IncompleteMemory(), IMemoryStore)

    def test_mock_used_in_mind_engine_init(self, mind_config):
        """MindEngine.__init__ accepts an IMemoryStore Mock."""
        from unittest.mock import AsyncMock, MagicMock

        from lingya.mind import MindEngine

        mock_memory = MagicMock()
        mock_memory.store_with_importance.return_value = "mem_test"
        mock_memory.score_importance = AsyncMock(return_value=7.0)
        mock_memory.update_importance = MagicMock()

        async def llm_call(prompt: str) -> str:
            return "test"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=llm_call,
        )
        assert engine.memory is mock_memory


class TestILLMBackend:
    def test_litellm_passes_isinstance(self):
        """LiteLLMModel with ainvoke should pass isinstance check."""
        import os

        from lingya.llm import LiteLLMModel
        from lingya.protocols import ILLMBackend

        if "DEEPSEEK_API_KEY" not in os.environ:
            pytest.skip("DEEPSEEK_API_KEY not set")

        model = LiteLLMModel(
            model="deepseek/deepseek-v4-flash",
            temperature=0.7,
        )
        assert isinstance(model, ILLMBackend)

    def test_mock_passes(self):
        """A mock with ainvoke should pass isinstance."""
        from unittest.mock import MagicMock

        from lingya.protocols import ILLMBackend

        mock_llm = MagicMock()
        mock_llm.ainvoke = MagicMock()
        assert isinstance(mock_llm, ILLMBackend)
