from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.messages import AIMessage

pytestmark = pytest.mark.asyncio


class TestDiaryStorage:
    @pytest.fixture
    def diary_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_get_last_diary_date_empty_dir(self, diary_dir):
        from lingya.diary import get_last_diary_date

        assert get_last_diary_date(diary_dir) is None

    def test_get_last_diary_date_returns_latest(self, diary_dir):
        from lingya.diary import get_last_diary_date, save_diary

        save_diary(diary_dir, date(2026, 5, 25), "Older entry.")
        save_diary(diary_dir, date(2026, 5, 27), "Newer entry.")

        assert get_last_diary_date(diary_dir) == date(2026, 5, 27)

    def test_save_and_read_diary(self, diary_dir):
        from lingya.diary import save_diary, read_diary

        path = save_diary(diary_dir, date(2026, 5, 27), "今天在想一些事情。")
        assert path.exists()
        assert path.suffix == ".md"

        result = read_diary(diary_dir, 0)
        assert result is not None
        diary_date, content = result
        assert diary_date == date(2026, 5, 27)
        assert "今天在想一些事情" in content

    def test_read_diary_out_of_range(self, diary_dir):
        from lingya.diary import read_diary

        assert read_diary(diary_dir, 0) is None
        assert read_diary(diary_dir, 99) is None

    def test_list_diaries_newest_first(self, diary_dir):
        from lingya.diary import save_diary, list_diaries

        save_diary(diary_dir, date(2026, 5, 25), "Old.")
        save_diary(diary_dir, date(2026, 5, 27), "New.")

        items = list_diaries(diary_dir)
        assert len(items) == 2
        assert items[0]["date"] == date(2026, 5, 27)
        assert items[1]["date"] == date(2026, 5, 25)

    def test_list_diaries_includes_preview(self, diary_dir):
        from lingya.diary import save_diary, list_diaries

        save_diary(diary_dir, date(2026, 5, 27), "First line.\nMore content here.")

        items = list_diaries(diary_dir)
        assert len(items) == 1
        assert "First line." in items[0]["preview"]

    def test_save_diary_creates_dir(self, diary_dir):
        from lingya.diary import save_diary

        nested = diary_dir / "nested"
        path = save_diary(nested, date(2026, 5, 27), "Content.")
        assert path.exists()


class TestDiaryJudgement:
    @pytest.fixture
    def diary_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_should_generate_when_no_diary_exists(self, diary_dir):
        from lingya.diary import should_generate_diary

        assert should_generate_diary(diary_dir, period_days=1) is True

    def test_should_generate_when_period_elapsed(self, diary_dir):
        from lingya.diary import should_generate_diary, save_diary

        old = date.today() - timedelta(days=3)
        save_diary(diary_dir, old, "Old diary.")
        assert should_generate_diary(diary_dir, period_days=1) is True

    def test_should_not_generate_when_within_period(self, diary_dir):
        from lingya.diary import should_generate_diary, save_diary

        save_diary(diary_dir, date.today(), "Today's diary.")
        assert should_generate_diary(diary_dir, period_days=1) is False

    def test_has_deep_conversation_enough_turns(self):
        from lingya.diary import has_deep_conversation

        turns = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "I'm feeling down"},
            {"role": "user", "content": "just work stuff"},
            {"role": "user", "content": "also this"},
            {"role": "user", "content": "one more"},
        ]
        assert has_deep_conversation(turns, min_turns=5) is True

    def test_has_deep_conversation_too_few_turns(self):
        from lingya.diary import has_deep_conversation

        turns = [
            {"role": "user", "content": "hello"},
            {"role": "ai", "content": "hi"},
        ]
        assert has_deep_conversation(turns, min_turns=5) is False

    def test_has_deep_conversation_filters_commands(self):
        from lingya.diary import has_deep_conversation

        turns = [
            {"role": "user", "content": "/memories"},
            {"role": "ai", "content": "listing..."},
            {"role": "user", "content": "/new"},
            {"role": "ai", "content": "created"},
            {"role": "user", "content": "/sessions"},
        ]
        assert has_deep_conversation(turns, min_turns=3) is False

    def test_has_deep_conversation_mixed_commands_and_talk(self):
        from lingya.diary import has_deep_conversation

        turns = [
            {"role": "user", "content": "/memories"},
            {"role": "user", "content": "最近心情不太好"},
            {"role": "ai", "content": "怎么了？"},
            {"role": "user", "content": "工作压力大"},
            {"role": "ai", "content": "跟我说说"},
            {"role": "user", "content": "就是..."},
        ]
        # 4 non-command user messages (excluding /memories)
        assert has_deep_conversation(turns, min_turns=3) is True


class TestDiaryGeneration:
    @pytest.fixture
    def model(self):
        m = MagicMock()
        m.ainvoke = AsyncMock(return_value=AIMessage(content="一篇日记内容。"))
        return m

    @pytest.fixture
    def failing_model(self):
        m = MagicMock()
        m.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        return m

    def test_format_transcript_groups_by_conversation(self):
        from lingya.diary import format_transcript

        turns = [
            {"role": "user", "content": "hello", "created_at": "2026-05-26", "conv_id": 1, "conv_title": "Session 1"},
            {"role": "ai", "content": "hi", "created_at": "2026-05-26", "conv_id": 1, "conv_title": "Session 1"},
            {"role": "user", "content": "bye", "created_at": "2026-05-27", "conv_id": 2, "conv_title": "Session 2"},
            {"role": "ai", "content": "see you", "created_at": "2026-05-27", "conv_id": 2, "conv_title": "Session 2"},
        ]
        result = format_transcript(turns)
        assert "2026-05-26" in result
        assert "2026-05-27" in result
        assert "User: hello" in result
        assert "LingYa: hi" in result

    async def test_generate_diary_returns_content(self, model, persona_config):
        from lingya.diary import generate_diary

        transcript = "User: 你好\nLingYa: 你好"
        result = await generate_diary(model, persona_config, transcript)
        assert result == "一篇日记内容。"

    async def test_generate_diary_includes_persona_in_prompt(self, model, persona_config):
        from lingya.diary import generate_diary

        transcript = "User: 你好\nLingYa: 你好"
        await generate_diary(model, persona_config, transcript)

        call_arg = model.ainvoke.call_args[0][0]
        prompt_text = call_arg[0].content if hasattr(call_arg[0], "content") else str(call_arg[0])
        assert persona_config.mind_core.identity in prompt_text
        assert persona_config.mind_core.core_belief in prompt_text
        assert transcript in prompt_text

    async def test_generate_diary_model_error_raises(self, failing_model, persona_config):
        from lingya.diary import generate_diary

        with pytest.raises(RuntimeError, match="API error"):
            await generate_diary(failing_model, persona_config, "transcript")
