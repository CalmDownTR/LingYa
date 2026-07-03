"""Thin BaseChatModel adapter over LiteLLM — bridges two open-source interfaces.

LiteLLM handles all provider complexity (100+ providers, key discovery,
streaming, tool calling, fallback). This adapter translates between
LangChain BaseChatModel (expected by DeepAgents, stream_events, etc.)
and litellm's completion API.  ~50 lines of glue code, not a provider
abstraction layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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

    def _to_litellm_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        """Convert LangChain messages to litellm dict format."""
        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }
        result: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = getattr(msg, "type", "user")
                role = role_map.get(role, "user")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append({"role": role, "content": content})
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

        litellm_messages = self._to_litellm_messages(messages)
        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
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
    ) -> Iterator[ChatGeneration]:
        """Stream chunks via litellm.completion with stream=True."""
        import litellm

        litellm_messages = self._to_litellm_messages(messages)
        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                yield ChatGeneration(message=AIMessage(content=content))

    @property
    def _llm_type(self) -> str:
        return "litellm"
