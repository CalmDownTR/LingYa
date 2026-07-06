"""Tests for LiteLLMModel bind_tools passthrough to litellm.completion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


class TestBindToolsPassthrough:
    """Verify that bind_tools stores are forwarded to litellm.completion."""

    @pytest.fixture
    def mock_litellm(self):
        """Mock litellm.completion to return a canned response."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as m:
            yield m

    def test_bind_tools_passed_to_generate(self, mock_litellm):
        """bind_tools + _generate should pass tools in OpenAI function calling format."""
        from lingya.llm import LiteLLMModel

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        model = LiteLLMModel(model="test/model")
        bound = model.bind_tools([search])
        bound._generate([HumanMessage(content="hi")])

        call_kwargs = mock_litellm.call_args.kwargs
        assert "tools" in call_kwargs, "Expected tools= in litellm.completion kwargs"
        tools = call_kwargs["tools"]
        assert isinstance(tools, list)
        assert len(tools) == 1
        # Verify OpenAI function calling format
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert "parameters" in tools[0]["function"]

    def test_no_bind_tools_no_tools_passed(self, mock_litellm):
        """Without bind_tools, litellm.completion should NOT receive tools."""
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel(model="test/model")
        model._generate([HumanMessage(content="hi")])

        call_kwargs = mock_litellm.call_args.kwargs
        assert "tools" not in call_kwargs, "Expected no tools= in litellm.completion kwargs"

    def test_bind_tools_passed_to_stream(self):
        """bind_tools + _stream should pass tools in OpenAI function calling format."""
        from lingya.llm import LiteLLMModel

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"

        with patch("litellm.completion", return_value=[mock_chunk]) as m:
            model = LiteLLMModel(model="test/model")
            bound = model.bind_tools([search])
            list(bound._stream([HumanMessage(content="hi")]))

            call_kwargs = m.call_args.kwargs
            assert "tools" in call_kwargs, "Expected tools= in litellm.completion kwargs"
            tools = call_kwargs["tools"]
            assert tools[0]["type"] == "function"
            assert tools[0]["function"]["name"] == "search"

    def test_tool_choice_passed(self, mock_litellm):
        """tool_choice should be forwarded."""
        from lingya.llm import LiteLLMModel

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        model = LiteLLMModel(model="test/model")
        bound = model.bind_tools([search], tool_choice="auto")
        bound._generate([HumanMessage(content="hi")])

        call_kwargs = mock_litellm.call_args.kwargs
        assert call_kwargs.get("tool_choice") == "auto"

    def test_setdefault_does_not_override_existing_tools(self, mock_litellm):
        """If kwargs already has tools= (from DeepAgents), bind_tools should not override."""
        from lingya.llm import LiteLLMModel

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        pre_existing_tools = [{"type": "function", "function": {"name": "existing", "parameters": {}}}]

        model = LiteLLMModel(model="test/model")
        bound = model.bind_tools([search])
        bound._generate([HumanMessage(content="hi")], tools=pre_existing_tools)

        call_kwargs = mock_litellm.call_args.kwargs
        # setdefault means kwargs-supplied tools win, not bind_tools
        assert call_kwargs["tools"] == pre_existing_tools


class TestToLitellmMessages:
    """Verify message conversion with tool-related types."""

    def test_tool_message_conversion(self):
        """ToolMessage should become {"role": "tool", "content": "...", "tool_call_id": "..."}."""
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        msg = ToolMessage(content="result text", tool_call_id="call_123")
        result = model._to_litellm_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "result text"
        assert result[0]["tool_call_id"] == "call_123"

    def test_aimessage_with_tool_calls(self):
        """AIMessage with tool_calls should preserve them."""
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        tool_calls = [{"name": "search", "args": {"query": "hi"}, "id": "call_1"}]
        msg = AIMessage(content="Let me search", tool_calls=tool_calls)
        result = model._to_litellm_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me search"
        # AIMessage auto-adds type='tool_call' to each dict
        assert result[0]["tool_calls"][0]["name"] == "search"
        assert result[0]["tool_calls"][0]["id"] == "call_1"

    def test_aimessage_without_tool_calls(self):
        """AIMessage without tool_calls should not have tool_calls key."""
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        msg = AIMessage(content="Hello")
        result = model._to_litellm_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello"
        assert "tool_calls" not in result[0]

    def test_mixed_messages(self):
        """A conversation with Human, AI tool_call, and Tool response."""
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        tool_calls = [{"name": "search", "args": {"q": "hi"}, "id": "c1"}]
        messages = [
            HumanMessage(content="search for hi"),
            AIMessage(content="", tool_calls=tool_calls),
            ToolMessage(content="found it", tool_call_id="c1"),
        ]
        result = model._to_litellm_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "search for hi"}
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == ""
        assert result[1]["tool_calls"][0]["name"] == "search"
        assert result[2] == {"role": "tool", "content": "found it", "tool_call_id": "c1"}
