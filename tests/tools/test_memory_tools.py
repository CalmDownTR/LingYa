"""Tests for lingya.tools.memory_tools — tool registration decoupled."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestCreateMemoryTools:
    def test_returns_two_tools(self):
        """create_memory_tools should return a list of 2 callable tools."""
        from lingya.tools.memory_tools import create_memory_tools

        mock_store = MagicMock()
        tools = create_memory_tools(mock_store)
        assert len(tools) == 2

    def test_remember_calls_memory_store(self):
        """The remember tool should call memory_store.store()."""
        from lingya.tools.memory_tools import create_memory_tools

        mock_store = MagicMock()
        mock_store.store.return_value = "mem_test123"

        tools = create_memory_tools(mock_store)
        remember = tools[0]
        result = remember.func("I like rainy days")

        mock_store.store.assert_called_once_with("I like rainy days")
        assert result == "mem_test123"

    def test_recall_calls_memory_search(self):
        """The recall tool should call memory_store.search()."""
        from lingya.tools.memory_tools import create_memory_tools

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"id": "mem_1", "text": "likes coffee"}
        ]

        tools = create_memory_tools(mock_store)
        recall = tools[1]
        result = recall.func("coffee")

        mock_store.search.assert_called_once_with("coffee")
        assert "[mem_1] likes coffee" in result

    def test_recall_no_results(self):
        """recall tool should return a friendly message when no results."""
        from lingya.tools.memory_tools import create_memory_tools

        mock_store = MagicMock()
        mock_store.search.return_value = []

        tools = create_memory_tools(mock_store)
        recall = tools[1]
        result = recall.func("nonexistent")

        assert "No matching memories" in result

    def test_both_tools_are_langchain_tools(self):
        """Both returned tools should be @tool decorated (LangChain tool)."""
        from langchain_core.tools import BaseTool

        from lingya.tools.memory_tools import create_memory_tools

        mock_store = MagicMock()
        tools = create_memory_tools(mock_store)
        for t in tools:
            assert isinstance(t, BaseTool)
