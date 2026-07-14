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
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.reasoning_content = None

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


class TestFallbacks:
    """Verify fallbacks parameter forwarding to litellm.completion."""

    def test_fallbacks_passed_to_generate(self):
        """When model has fallbacks, they should be passed to litellm.completion."""
        from unittest.mock import patch, MagicMock

        from lingya.llm import LiteLLMModel

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as m:
            model = LiteLLMModel(
                model="test/model",
                fallbacks=["openai/gpt-4o", "anthropic/claude-sonnet-4"],
            )
            model._generate([HumanMessage(content="hi")])

            call_kwargs = m.call_args.kwargs
            assert "fallbacks" in call_kwargs
            assert call_kwargs["fallbacks"] == ["openai/gpt-4o", "anthropic/claude-sonnet-4"]

    def test_no_fallbacks_when_empty(self):
        """Without fallbacks configured, none should be passed."""
        from unittest.mock import patch, MagicMock

        from lingya.llm import LiteLLMModel

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as m:
            model = LiteLLMModel(model="test/model", fallbacks=[])
            model._generate([HumanMessage(content="hi")])

            call_kwargs = m.call_args.kwargs
            assert "fallbacks" not in call_kwargs

    def test_setdefault_does_not_override_kwargs_fallbacks(self):
        """If kwargs already has fallbacks, model fallbacks should not override."""
        from unittest.mock import patch, MagicMock

        from lingya.llm import LiteLLMModel

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response) as m:
            model = LiteLLMModel(model="test/model", fallbacks=["backup/model"])
            model._generate(
                [HumanMessage(content="hi")],
                fallbacks=["explicit/fallback"],
            )

            call_kwargs = m.call_args.kwargs
            # kwargs-supplied fallbacks take precedence via setdefault
            assert call_kwargs["fallbacks"] == ["explicit/fallback"]


class TestStreamChunks:
    """Verify _stream yields text and tool-call chunks correctly."""

    def test_stream_yields_text_chunks(self):
        """Text deltas should be yielded as ChatGenerationChunk."""
        from langchain_core.outputs import ChatGenerationChunk
        from lingya.llm import LiteLLMModel

        chunks = []
        for text in ("Hello", " world"):
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = text
            mock_chunk.choices[0].delta.tool_calls = None
            mock_chunk.choices[0].delta.reasoning_content = None
            chunks.append(mock_chunk)

        with patch("litellm.completion", return_value=chunks):
            model = LiteLLMModel(model="test/model")
            result = list(model._stream([HumanMessage(content="hi")]))

        assert len(result) == 2
        assert all(isinstance(c, ChatGenerationChunk) for c in result)
        assert "".join(c.message.content for c in result) == "Hello world"

    def test_stream_yields_tool_call_chunks_with_empty_content(self):
        """Tool-call-only deltas must be yielded so agents can execute tools."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from lingya.llm import LiteLLMModel

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = ""
        mock_chunk.choices[0].delta.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "hi"}'},
                "index": 0,
            }
        ]
        mock_chunk.choices[0].delta.reasoning_content = None

        with patch("litellm.completion", return_value=[mock_chunk]):
            model = LiteLLMModel(model="test/model")
            result = list(model._stream([HumanMessage(content="hi")]))

        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, ChatGenerationChunk)
        msg = chunk.message
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == ""
        assert len(msg.tool_call_chunks) == 1
        assert msg.tool_call_chunks[0]["name"] == "search"
        assert msg.tool_call_chunks[0]["args"] == '{"q": "hi"}'
        assert msg.tool_call_chunks[0]["id"] == "call_1"

    def test_stream_skips_completely_empty_chunks(self):
        """Chunks with no content and no tool calls should not be yielded."""
        from lingya.llm import LiteLLMModel

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = ""
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.reasoning_content = None

        with patch("litellm.completion", return_value=[mock_chunk]):
            model = LiteLLMModel(model="test/model")
            result = list(model._stream([HumanMessage(content="hi")]))

        assert result == []

    def test_stream_preserves_reasoning_content(self):
        """DeepSeek-style reasoning_content should be attached to chunks."""
        from lingya.llm import LiteLLMModel

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "final"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.reasoning_content = "think step by step"

        with patch("litellm.completion", return_value=[mock_chunk]):
            model = LiteLLMModel(model="test/model")
            result = list(model._stream([HumanMessage(content="hi")]))

        assert len(result) == 1
        assert result[0].message.additional_kwargs.get("reasoning_content") == "think step by step"


class TestAStreamChunks:
    """Verify async _astream yields text and tool-call chunks."""

    @pytest.mark.asyncio
    async def test_astream_yields_text_chunks(self):
        from langchain_core.outputs import ChatGenerationChunk
        from lingya.llm import LiteLLMModel

        async def _async_chunks():
            for text in ("Hello", " world"):
                mock_chunk = MagicMock()
                mock_chunk.choices = [MagicMock()]
                mock_chunk.choices[0].delta.content = text
                mock_chunk.choices[0].delta.tool_calls = None
                mock_chunk.choices[0].delta.reasoning_content = None
                yield mock_chunk

        with patch("litellm.acompletion", return_value=_async_chunks()):
            model = LiteLLMModel(model="test/model")
            result = []
            async for chunk in model._astream([HumanMessage(content="hi")]):
                result.append(chunk)

        assert len(result) == 2
        assert all(isinstance(c, ChatGenerationChunk) for c in result)
        assert "".join(c.message.content for c in result) == "Hello world"

    @pytest.mark.asyncio
    async def test_astream_yields_tool_call_chunks(self):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from lingya.llm import LiteLLMModel

        async def _async_chunks():
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = ""
            mock_chunk.choices[0].delta.tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "hi"}'},
                    "index": 0,
                }
            ]
            mock_chunk.choices[0].delta.reasoning_content = None
            yield mock_chunk

        with patch("litellm.acompletion", return_value=_async_chunks()):
            model = LiteLLMModel(model="test/model")
            result = []
            async for chunk in model._astream([HumanMessage(content="hi")]):
                result.append(chunk)

        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, ChatGenerationChunk)
        assert isinstance(chunk.message, AIMessageChunk)
        assert len(chunk.message.tool_call_chunks) == 1
        assert chunk.message.tool_call_chunks[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_astream_passes_tools(self):
        from langchain_core.tools import tool
        from lingya.llm import LiteLLMModel

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for {query}"

        async def _async_chunks():
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "ok"
            mock_chunk.choices[0].delta.tool_calls = None
            mock_chunk.choices[0].delta.reasoning_content = None
            yield mock_chunk

        with patch("litellm.acompletion", return_value=_async_chunks()) as m:
            model = LiteLLMModel(model="test/model")
            bound = model.bind_tools([search])
            async for _ in bound._astream([HumanMessage(content="hi")]):
                pass

        call_kwargs = m.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "search"


class TestOutputVersion:
    """Verify content is emitted as plain text, not content_blocks."""

    def test_output_version_defaults_to_v0(self):
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        assert model.output_version == "v0"

    def test_output_version_can_override_env(self, monkeypatch):
        from lingya.llm import LiteLLMModel

        monkeypatch.setenv("LC_OUTPUT_VERSION", "v1")
        model = LiteLLMModel()
        assert model.output_version == "v0"


class TestExtractTextContent:
    """Verify _extract_text_content normalizes various content shapes."""

    def test_plain_string(self):
        from lingya.llm import LiteLLMModel

        assert LiteLLMModel._extract_text_content("hello") == "hello"

    def test_content_blocks(self):
        from lingya.llm import LiteLLMModel

        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert LiteLLMModel._extract_text_content(blocks) == "hello\nworld"

    def test_content_blocks_skip_non_text(self):
        from lingya.llm import LiteLLMModel

        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "http://example.com/x.png"},
        ]
        assert LiteLLMModel._extract_text_content(blocks) == "hello"

    def test_none(self):
        from lingya.llm import LiteLLMModel

        assert LiteLLMModel._extract_text_content(None) == ""


class TestToLitellmMessagesContentBlocks:
    """Verify _to_litellm_messages flattens content_blocks to strings."""

    def test_aimessage_with_content_blocks(self):
        from langchain_core.messages import AIMessage
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        msg = AIMessage(content=[{"type": "text", "text": "hello"}])
        result = model._to_litellm_messages([msg])
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "hello"

    def test_human_message_with_content_blocks(self):
        from langchain_core.messages import HumanMessage
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        msg = HumanMessage(content=[{"type": "text", "text": "hi there"}])
        result = model._to_litellm_messages([msg])
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi there"

    def test_tool_message_with_content_blocks(self):
        from langchain_core.messages import ToolMessage
        from lingya.llm import LiteLLMModel

        model = LiteLLMModel()
        msg = ToolMessage(
            content=[{"type": "text", "text": "tool result"}],
            tool_call_id="call_1",
        )
        result = model._to_litellm_messages([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "tool result"
        assert result[0]["tool_call_id"] == "call_1"


class TestGenerateContentBlocks:
    """Verify _generate flattens content_blocks in the litellm response."""

    def test_generate_with_content_block_response(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import HumanMessage
        from lingya.llm import LiteLLMModel

        mock_choice = MagicMock()
        mock_choice.message.content = [{"type": "text", "text": "Hello"}]
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("litellm.completion", return_value=mock_response):
            model = LiteLLMModel(model="test/model")
            result: ChatResult = model._generate([HumanMessage(content="hi")])

        generation = result.generations[0]
        assert isinstance(generation, ChatGeneration)
        assert isinstance(generation.message, AIMessage)
        assert generation.message.content == "Hello"


class TestStreamContentBlocks:
    """Verify _stream flattens content_blocks in deltas."""

    def test_stream_with_content_block_delta(self):
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import HumanMessage
        from lingya.llm import LiteLLMModel

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = [{"type": "text", "text": "Hello"}]
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].delta.reasoning_content = None

        with patch("litellm.completion", return_value=[mock_chunk]):
            model = LiteLLMModel(model="test/model")
            result = list(model._stream([HumanMessage(content="hi")]))

        assert len(result) == 1
        assert isinstance(result[0], ChatGenerationChunk)
        assert result[0].message.content == "Hello"
