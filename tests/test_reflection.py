from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.messages import AIMessage

pytestmark = pytest.mark.asyncio


class TestGenerateOpeningLine:
    @pytest.fixture
    def model(self):
        """A mock model that returns a canned response."""
        m = MagicMock()
        m.ainvoke = AsyncMock(return_value=AIMessage(content="你好，我是 LingYa。"))
        return m

    @pytest.fixture
    def failing_model(self):
        """A mock model that raises on invoke."""
        m = MagicMock()
        m.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        return m

    async def test_first_time_returns_introduction(self, model, mind_config):
        from lingya.reflection import generate_opening_line

        result = await generate_opening_line(model, mind_config, transcript=None)
        assert result == "你好，我是 LingYa。"

    async def test_returning_user_has_transcript_in_prompt(self, model, mind_config):
        from lingya.reflection import generate_opening_line

        transcript = "User: 你好\nLingYa: 你好，有什么可以帮你的？"
        result = await generate_opening_line(model, mind_config, transcript=transcript)
        assert result == "你好，我是 LingYa。"
        # Verify the prompt contained the transcript
        call_arg = model.ainvoke.call_args[0][0]
        prompt_text = call_arg[0].content if hasattr(call_arg[0], "content") else str(call_arg[0])
        assert "你好" in prompt_text

    async def test_model_error_returns_none(self, failing_model, mind_config):
        from lingya.reflection import generate_opening_line

        result = await generate_opening_line(
            failing_model, mind_config, transcript=None
        )
        assert result is None

    async def test_empty_model_response_returns_none(self, mind_config):
        from lingya.reflection import generate_opening_line

        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        result = await generate_opening_line(model, mind_config, transcript=None)
        assert result is None


class TestDatabaseTurns:
    async def test_add_and_get_turns(self, db):
        cid = await db.create_conversation("Test turns")

        await db.add_turn(cid, "user", "你好")
        await db.add_turn(cid, "ai", "你好，有什么可以帮你？")
        await db.add_turn(cid, "user", "今天天气不错")
        await db.add_turn(cid, "ai", "是的，适合出去走走")

        turns = await db.get_turns(cid, limit=6)
        assert len(turns) == 4
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "你好"
        assert turns[-1]["role"] == "ai"
        assert turns[-1]["content"] == "是的，适合出去走走"

    async def test_get_turns_respects_limit(self, db):
        cid = await db.create_conversation("Test limit")

        for i in range(10):
            await db.add_turn(cid, "user", f"msg {i}")

        turns = await db.get_turns(cid, limit=3)
        assert len(turns) == 3
        # Should be the last 3
        assert turns[0]["content"] == "msg 7"
        assert turns[-1]["content"] == "msg 9"

    async def test_get_turns_empty_conversation(self, db):
        cid = await db.create_conversation("Empty")
        turns = await db.get_turns(cid)
        assert turns == []

    async def test_get_turns_since_filters_by_date(self, db):
        cid = await db.create_conversation("Test date filter")
        await db.add_turn(cid, "user", "old message")
        await db.add_turn(cid, "ai", "old reply")

        # Get turns since far future — should be empty
        turns = await db.get_turns_since("2099-01-01", limit=100)
        assert len(turns) == 0

        # Get turns since epoch — should include all
        turns = await db.get_turns_since("1970-01-01", limit=100)
        assert len(turns) >= 2

    async def test_get_turns_since_includes_conv_metadata(self, db):
        cid = await db.create_conversation("Metadata test")
        await db.add_turn(cid, "user", "hello")

        turns = await db.get_turns_since("1970-01-01", limit=100)
        assert len(turns) >= 1
        assert turns[0]["conv_id"] == cid
        assert turns[0]["conv_title"] == "Metadata test"
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "hello"
        assert "created_at" in turns[0]

    async def test_get_turns_since_respects_limit(self, db):
        cid = await db.create_conversation("Limit test")
        for i in range(5):
            await db.add_turn(cid, "user", f"msg {i}")

        turns = await db.get_turns_since("1970-01-01", limit=2)
        assert len(turns) == 2
