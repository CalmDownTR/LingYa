from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from lingya.middleware import PersonalityMiddleware, _extract_last_user_text


class TestExtractLastUserText:
    @pytest.mark.parametrize(
        "blocks,expected",
        [
            ([{"type": "text", "text": "Hello world"}], "Hello world"),
            ([{"type": "image", "url": "cat.jpg"}, {"type": "text", "text": "Describe"}], "Describe"),
            ([{"type": "text"}], ""),
        ],
    )
    def test_extracts_text_from_content_blocks(self, blocks, expected):
        msgs = [HumanMessage(blocks)]
        assert _extract_last_user_text(msgs) == expected

    def test_skips_ai_messages(self):
        msgs = [
            HumanMessage([{"type": "text", "text": "First"}]),
            MagicMock(),
            HumanMessage([{"type": "text", "text": "Last"}]),
        ]
        assert _extract_last_user_text(msgs) == "Last"

    @pytest.mark.parametrize("msgs", [[], [MagicMock()]])
    def test_returns_empty_when_no_human_message(self, msgs):
        assert _extract_last_user_text(msgs) == ""


class TestPersonalityMiddleware:
    pytestmark = pytest.mark.asyncio

    @pytest.fixture
    def engine(self):
        eng = MagicMock()
        eng.get_system_prompt.return_value = "PERSONALITY_PROMPT"
        return eng

    @pytest.fixture
    def middleware(self, engine):
        return PersonalityMiddleware(engine)

    @pytest.fixture
    def handler(self):
        return AsyncMock()

    async def test_prepends_personality_to_system_message(self, middleware, engine, handler):
        request = MagicMock()
        request.messages = [HumanMessage([{"type": "text", "text": "hi"}])]
        request.system_message = SystemMessage("base system prompt")
        request.override.return_value = request

        await middleware.awrap_model_call(request, handler)

        engine.get_system_prompt.assert_called_once_with("hi")
        new_sys = request.override.call_args[1]["system_message"]
        assert new_sys.content == "PERSONALITY_PROMPT\n\nbase system prompt"
        handler.assert_awaited_once_with(request)

    @pytest.mark.parametrize(
        "system_msg,expected",
        [
            (None, "PERSONALITY_PROMPT"),
            (SystemMessage([{"type": "image", "url": "img.jpg"}]), "PERSONALITY_PROMPT"),
            (SystemMessage([{"type": "text", "text": ""}]), "PERSONALITY_PROMPT"),
        ],
    )
    async def test_personality_alone_when_no_meaningful_system_msg(
        self, middleware, engine, handler, system_msg, expected
    ):
        request = MagicMock()
        request.messages = [HumanMessage([{"type": "text", "text": "hi"}])]
        request.system_message = system_msg
        request.override.return_value = request

        await middleware.awrap_model_call(request, handler)

        new_sys = request.override.call_args[1]["system_message"]
        assert new_sys.content == expected

    async def test_extracts_text_from_content_blocks_in_system_message(
        self, middleware, engine, handler
    ):
        request = MagicMock()
        request.messages = [HumanMessage([{"type": "text", "text": "hi"}])]
        request.system_message = SystemMessage([
            {"type": "text", "text": "Part A"},
            {"type": "image", "url": "img.jpg"},
            {"type": "text", "text": "Part B"},
        ])
        request.override.return_value = request

        await middleware.awrap_model_call(request, handler)

        new_sys = request.override.call_args[1]["system_message"]
        assert new_sys.content == "PERSONALITY_PROMPT\n\nPart APart B"

    async def test_passes_last_user_text_to_engine(self, middleware, engine, handler):
        request = MagicMock()
        request.messages = [
            HumanMessage([{"type": "text", "text": "old"}]),
            MagicMock(),
            HumanMessage([{"type": "text", "text": "current"}]),
        ]
        request.system_message = None
        request.override.return_value = request

        await middleware.awrap_model_call(request, handler)

        engine.get_system_prompt.assert_called_once_with("current")
