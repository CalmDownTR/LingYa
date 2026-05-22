from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lingya.memory.long_term import MemoryEntry
from lingya.memory.tools import create_memory_tools


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.search = AsyncMock()
    mem.store = AsyncMock()
    return mem


@pytest.fixture
def tools(mock_memory):
    return create_memory_tools(mock_memory)


class TestSearchMemory:
    pytestmark = pytest.mark.asyncio

    async def test_returns_formatted_results(self, tools, mock_memory):
        mock_memory.search.return_value = [
            MemoryEntry(id="1", text="Fact A", metadata={"source": "chat1"}),
            MemoryEntry(id="2", text="Fact B", metadata={"source": "chat2"}),
        ]
        result = await tools[0].ainvoke({"query": "Fact"})
        assert "[chat1] Fact A" in result
        assert "[chat2] Fact B" in result

    async def test_returns_no_results_message(self, tools, mock_memory):
        mock_memory.search.return_value = []
        result = await tools[0].ainvoke({"query": "nothing"})
        assert result == "No relevant memories found."

    async def test_handles_missing_source_metadata(self, tools, mock_memory):
        mock_memory.search.return_value = [
            MemoryEntry(id="1", text="Bare fact", metadata={}),
        ]
        result = await tools[0].ainvoke({"query": "fact"})
        assert "[unknown] Bare fact" in result

    async def test_passes_query_to_memory(self, tools, mock_memory):
        mock_memory.search.return_value = []
        await tools[0].ainvoke({"query": "specific query"})
        mock_memory.search.assert_awaited_once_with("specific query")


class TestSaveMemory:
    pytestmark = pytest.mark.asyncio

    async def test_saves_chunks_and_returns_count(self, tools, mock_memory):
        result = await tools[1].ainvoke(
            {"content": "Important fact about user", "source": "chat"}
        )
        assert "Saved" in result
        assert "chat" in result
        mock_memory.store.assert_awaited_once()

    @pytest.mark.parametrize("content", ["", "   \n  \t  "])
    async def test_empty_or_whitespace_returns_nothing(self, tools, mock_memory, content):
        result = await tools[1].ainvoke({"content": content, "source": "chat"})
        assert result == "Nothing to save."
        mock_memory.store.assert_not_awaited()

    async def test_stores_entries_with_metadata(self, tools, mock_memory):
        await tools[1].ainvoke({"content": "A fact", "source": "direct-input"})

        entries: list[MemoryEntry] = mock_memory.store.call_args[0][0]
        assert len(entries) >= 1
        for entry in entries:
            assert isinstance(entry, MemoryEntry)
            assert entry.metadata["type"] == "agent_saved"
            assert entry.metadata["source"] == "direct-input"

    async def test_long_content_is_chunked(self, tools, mock_memory):
        long_text = "This is a test. " * 500
        result = await tools[1].ainvoke({"content": long_text, "source": "long-chat"})

        entries: list[MemoryEntry] = mock_memory.store.call_args[0][0]
        assert len(entries) > 1
        assert result == f"Saved {len(entries)} chunks from long-chat."

class TestCreateMemoryTools:
    def test_returns_two_tools(self, tools):
        assert len(tools) == 2
