"""Thin BaseChatModel adapter over LiteLLM — bridges two open-source interfaces.

LiteLLM handles all provider complexity (100+ providers, key discovery,
streaming, tool calling, fallback). This adapter translates between
LangChain BaseChatModel (expected by DeepAgents, stream_events, etc.)
and litellm's completion API.  ~50 lines of glue code, not a provider
abstraction layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
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

    # Allow extra fields set by ApplicationBuilder (e.g. "profile")
    model_config = {"extra": "allow"}

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
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                }
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                result.append(entry)
                continue
            else:
                role = getattr(msg, "type", "user")

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            entry = {"role": role, "content": content}

            # Preserve tool_calls on AIMessage for function calling
            if isinstance(msg, AIMessage):
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    entry["tool_calls"] = tc

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
        content = choice.message.content or ""
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
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

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
