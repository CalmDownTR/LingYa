"""Thin BaseChatModel adapter over LiteLLM — bridges two open-source interfaces.

LiteLLM handles all provider complexity (100+ providers, key discovery,
streaming, tool calling, fallback). This adapter translates between
LangChain BaseChatModel (expected by DeepAgents, stream_events, etc.)
and litellm's completion API.  ~50 lines of glue code, not a provider
abstraction layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field


class LiteLLMModel(BaseChatModel):
    """LangChain-compatible chat model that delegates to litellm.completion.

    Supports all providers that litellm supports — provider is inferred
    from the model name (e.g. ``deepseek/deepseek-v4-flash``).

    API keys are auto-discovered by litellm from standard env vars
    (``DEEPSEEK_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
    """

    model: str = Field(default="deepseek/deepseek-v4-flash")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=32768, gt=0)
    fallbacks: list[str] = Field(default_factory=list)
    # Pin to plain-string content. This prevents LangChain's v1 content_blocks
    # format from leaking into the response and keeps message history as simple
    # strings that the frontend can render directly.
    output_version: str | None = Field(default="v0")

    # Allow extra fields set by ApplicationBuilder (e.g. "profile")
    model_config = {"extra": "allow"}

    @staticmethod
    def _extract_text_content(content: str | list | Any) -> str:
        """Normalize message content to a plain string.

        Handles the common cases:
        - plain string -> returned as-is
        - list of content blocks -> concatenate text blocks
        - anything else -> str()
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts) if parts else str(content)
        return str(content) if content is not None else ""

    def _to_litellm_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to litellm dict format.

        Supports Human, System, AI (with optional tool_calls), and Tool messages.
        """
        from langchain_core.messages import ToolMessage

        result: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                entry: dict[str, Any] = {
                    "role": "tool",
                    "content": self._extract_text_content(msg.content),
                }
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                result.append(entry)
                continue
            else:
                role = getattr(msg, "type", "user")

            entry = {"role": role, "content": self._extract_text_content(msg.content)}

            # Preserve tool_calls on AIMessage for function calling.
            # LangChain stores tool_calls in a flat dict format:
            #   {"name": ..., "args": {...}, "id": "...", "type": "tool_call"}
            # Every OpenAI-compatible API expects the nested format:
            #   {"type": "function", "id": "...", "function": {"name": ..., "arguments": "..."}}
            # Normalize here so LiteLLM passes them through without rejection.
            if isinstance(msg, AIMessage):
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    import json as _json

                    normalized = []
                    for t in tc:
                        t = dict(t)
                        if "function" not in t:
                            # Flat LangChain format → nested OpenAI format
                            fn_name = t.pop("name", "")
                            fn_args = t.pop("args", {})
                            t["type"] = "function"
                            t["function"] = {
                                "name": fn_name,
                                "arguments": fn_args if isinstance(fn_args, str) else _json.dumps(fn_args, ensure_ascii=False),
                            }
                        elif t.get("type") == "tool_call":
                            t["type"] = "function"
                        normalized.append(t)
                    entry["tool_calls"] = normalized

            result.append(entry)
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call litellm.completion and return a LangChain ChatResult."""
        import litellm

        from langchain_core.utils.function_calling import convert_to_openai_tool

        litellm_messages = self._to_litellm_messages(messages)

        # Forward bound tools to litellm in OpenAI function calling format.
        # setdefault ensures DeepAgents-supplied tools in kwargs take precedence.
        tools = getattr(self, "_bound_tools", None)
        if tools:
            openai_tools = [convert_to_openai_tool(t) for t in tools]
            kwargs.setdefault("tools", openai_tools)
            tool_choice = getattr(self, "_bound_tools_kwargs", {}).get("tool_choice", "auto")
            kwargs.setdefault("tool_choice", tool_choice)

        # Forward fallbacks — litellm.completion natively supports
        # fallbacks=[...] and will try them in order on failure.
        if self.fallbacks:
            kwargs.setdefault("fallbacks", self.fallbacks)

        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            num_retries=2,
            **kwargs,
        )
        choice = response.choices[0]
        content = self._extract_text_content(choice.message.content)
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chunks via litellm.completion with stream=True."""
        import litellm

        from langchain_core.utils.function_calling import convert_to_openai_tool

        litellm_messages = self._to_litellm_messages(messages)

        # Forward bound tools to litellm in OpenAI function calling format.
        tools = getattr(self, "_bound_tools", None)
        if tools:
            openai_tools = [convert_to_openai_tool(t) for t in tools]
            kwargs.setdefault("tools", openai_tools)
            tool_choice = getattr(self, "_bound_tools_kwargs", {}).get("tool_choice", "auto")
            kwargs.setdefault("tool_choice", tool_choice)

        # Forward fallbacks — litellm.completion natively supports
        # fallbacks=[...] and will try them in order on failure.
        if self.fallbacks:
            kwargs.setdefault("fallbacks", self.fallbacks)

        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            num_retries=2,
            **kwargs,
        )
        for chunk in response:
            gen_chunk = self._litellm_chunk_to_generation_chunk(chunk)
            msg = gen_chunk.message
            if msg.content or getattr(msg, "tool_call_chunks", None):
                yield gen_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chunks asynchronously via litellm.acompletion with stream=True."""
        import litellm

        from langchain_core.utils.function_calling import convert_to_openai_tool

        litellm_messages = self._to_litellm_messages(messages)

        tools = getattr(self, "_bound_tools", None)
        if tools:
            openai_tools = [convert_to_openai_tool(t) for t in tools]
            kwargs.setdefault("tools", openai_tools)
            tool_choice = getattr(self, "_bound_tools_kwargs", {}).get("tool_choice", "auto")
            kwargs.setdefault("tool_choice", tool_choice)

        if self.fallbacks:
            kwargs.setdefault("fallbacks", self.fallbacks)

        response = await litellm.acompletion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            num_retries=2,
            **kwargs,
        )
        async for chunk in response:
            gen_chunk = self._litellm_chunk_to_generation_chunk(chunk)
            msg = gen_chunk.message
            if msg.content or getattr(msg, "tool_call_chunks", None):
                yield gen_chunk

    def _litellm_chunk_to_generation_chunk(self, chunk: Any) -> ChatGenerationChunk:
        """Convert a single litellm/OpenAI streaming chunk to a LangChain chunk.

        Preserves text deltas, tool-call deltas, and provider-specific fields
        such as DeepSeek's ``reasoning_content``.
        """
        delta = chunk.choices[0].delta
        content = self._extract_text_content(getattr(delta, "content", None))

        tool_call_chunks = []
        raw_tool_calls = getattr(delta, "tool_calls", None)
        if raw_tool_calls and isinstance(raw_tool_calls, list):
            for raw_tc in raw_tool_calls:
                try:
                    tc = raw_tc if isinstance(raw_tc, dict) else raw_tc.model_dump()
                    fn = tc.get("function", {}) or {}
                    tool_call_chunks.append(
                        tool_call_chunk(
                            name=fn.get("name"),
                            args=fn.get("arguments"),
                            id=tc.get("id"),
                            index=tc.get("index"),
                        )
                    )
                except Exception:
                    # Malformed tool-call deltas should not break the stream.
                    pass

        additional_kwargs: dict[str, Any] = {}
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            additional_kwargs["reasoning_content"] = reasoning

        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_call_chunks,
            )
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> LiteLLMModel:
        """Bind tools to the model, returning a new instance with tools stored.

        LiteLLM handles tool calling transparently — we just need to pass
        tools through to litellm.completion. This override stores the tools
        on the model instance so `_generate` / `_stream` can forward them.
        """
        import copy

        bound = copy.copy(self)
        object.__setattr__(bound, "_bound_tools", list(tools))
        object.__setattr__(bound, "_bound_tools_kwargs", kwargs)
        return bound

    @property
    def _llm_type(self) -> str:
        return "litellm"
