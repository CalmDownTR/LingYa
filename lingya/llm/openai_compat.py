from __future__ import annotations

import os

from openai import AsyncOpenAI

from lingya.config import LLMConfig

from .base import BaseLLMBackend, LLMResponse, ToolDefinition


class OpenAICompatBackend(BaseLLMBackend):
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = os.environ.get(self._config.api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"API key not found. Set {self._config.api_key_env} in .env file."
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._config.api_base_url,
            )
        return self._client

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        api_messages: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        api_messages.extend(messages)

        kwargs: dict = {
            "model": self._config.model,
            "messages": api_messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        if tools:
            kwargs["tools"] = [t.to_openai_format() for t in tools]

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            },
        )

    async def generate_simple(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int | None = None,
    ) -> str:
        api_messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        kwargs: dict = {
            "model": self._config.model,
            "messages": api_messages,
            "temperature": self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }

        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
