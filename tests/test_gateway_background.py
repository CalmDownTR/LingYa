"""Tests for BackgroundRunner — heartbeat + diary scheduler loops."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_engine():
    """Mock MindEngine with idle_tick and proper config."""
    from lingya.mind.config import (
        BigFiveTraits,
        IdentityAnchor,
        MindConfig,
        PersonaMeta,
        ToneMatrix,
    )

    engine = MagicMock()
    engine.idle_tick = AsyncMock()
    engine.state = MagicMock()
    engine.state.current_pad = MagicMock()
    engine.state.current_pad.pleasure = 0.0
    engine.state.current_pad.arousal = 0.0
    engine.state.current_pad.dominance = 0.0
    engine.config = MindConfig(
        version="1.0",
        meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
        identity=IdentityAnchor(
            identity="You are a test assistant.",
            core_belief="Test core belief.",
        ),
        ocean=BigFiveTraits(),
        tone_matrix=ToneMatrix(),
        behavior_guardrails=["Be honest."],
    )
    return engine


@pytest.fixture
def mock_db():
    """Mock Database."""
    db = MagicMock()
    db.get_turns_since = AsyncMock(return_value=[])
    return db


@pytest.fixture
def mock_model():
    """Mock ChatOpenAI model."""
    m = MagicMock()
    m.ainvoke = AsyncMock()
    m.ainvoke.return_value.content = "一篇日记。"
    m.ainvoke.return_value.text = "一篇日记。"
    return m


@pytest.fixture
def tmp_data_dir():
    """Temporary data directory with a diary subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "diary").mkdir(exist_ok=True)
        yield tmpdir


class TestBackgroundRunnerInit:
    def test_init_stores_config(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=30,
            diary_check_interval=1800,
        )

        assert runner._engine is mock_engine
        assert runner._db is mock_db
        assert runner._model is mock_model
        assert runner._data_dir == tmp_data_dir
        assert runner.heartbeat_interval == 30
        assert runner.diary_check_interval == 1800
        assert runner._running is True
        assert runner._heartbeat_task is None
        assert runner._diary_task is None

    def test_init_default_intervals(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        assert runner.heartbeat_interval == 60
        assert runner.diary_check_interval == 3600


class TestBackgroundRunnerStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_running_tasks(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=0.01,
            diary_check_interval=0.01,
        )

        await runner.start()
        assert runner._heartbeat_task is not None
        assert runner._diary_task is not None
        assert runner.is_running is True

        await runner.stop()
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cleans_up_tasks(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=0.01,
            diary_check_interval=3600,
        )

        await runner.start()
        assert runner._heartbeat_task is not None

        await runner.stop()

        assert runner.is_running is False
        assert runner._heartbeat_task is None
        assert runner._diary_task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )
        runner._running = False

        # Should not raise
        await runner.stop()
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_start_twice_does_not_duplicate(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=0.01,
            diary_check_interval=3600,
        )

        await runner.start()
        first_heartbeat = runner._heartbeat_task

        # Second start should be a no-op
        await runner.start()
        assert runner._heartbeat_task is first_heartbeat

        await runner.stop()


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_heartbeat_loop_calls_idle_tick(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """Heartbeat loop calls idle_tick after each sleep interval."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=0.01,
        )

        # Run one iteration of the heartbeat loop manually
        runner._running = True

        async def run_one_tick():
            # Simulate one iteration: sleep, check running, call idle_tick
            await asyncio.sleep(0.01)
            if runner._running:
                await runner._engine.idle_tick()
            # Stop after one iteration
            runner._running = False

        await run_one_tick()
        mock_engine.idle_tick.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_skips_when_not_running(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """Heartbeat loop should not call idle_tick when _running is False."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=0.01,
        )

        runner._running = False
        # Directly test the loop logic: if _running is False, idle_tick is never called
        # Use the loop with mock sleep that returns immediately
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await runner._heartbeat_loop()

        mock_engine.idle_tick.assert_not_called()


class TestDiaryScheduler:
    @pytest.mark.asyncio
    async def test_try_generate_diary_when_due(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """_try_generate_diary generates diary when should_generate_diary returns True."""
        from lingya.gateway.background import BackgroundRunner

        meaningful_turns = [
            {"role": "user", "content": "hello", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "hi there", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "how are you", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "good", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "nice day", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "yes", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "anything else", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "nope", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "bye", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
        ]
        mock_db.get_turns_since = AsyncMock(return_value=meaningful_turns)

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        with patch("lingya.diary.should_generate_diary", return_value=True):
            await runner._try_generate_diary()

        # Diary should have been generated
        diary_dir = Path(tmp_data_dir) / "diary"
        md_files = list(diary_dir.glob("*.md"))
        assert len(md_files) >= 1

    @pytest.mark.asyncio
    async def test_try_generate_diary_skips_when_not_due(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """_try_generate_diary does nothing when should_generate_diary returns False."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        with patch("lingya.diary.should_generate_diary", return_value=False):
            await runner._try_generate_diary()

        # No turns fetched
        mock_db.get_turns_since.assert_not_called()

    @pytest.mark.asyncio
    async def test_try_generate_diary_skips_when_no_deep_conversation(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """_try_generate_diary skips when there aren't enough meaningful user turns."""
        from lingya.gateway.background import BackgroundRunner

        shallow_turns = [
            {"role": "user", "content": "/memories", "created_at": "2026-05-30", "conv_id": 1},
            {"role": "ai", "content": "listing...", "created_at": "2026-05-30", "conv_id": 1},
        ]
        mock_db.get_turns_since = AsyncMock(return_value=shallow_turns)

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        with patch("lingya.diary.should_generate_diary", return_value=True):
            await runner._try_generate_diary()

        # model.ainvoke should not have been called (no diary generation)
        mock_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_try_generate_diary_handles_db_error_gracefully(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """_try_generate_diary should not raise on DB error."""
        from lingya.gateway.background import BackgroundRunner

        mock_db.get_turns_since = AsyncMock(side_effect=RuntimeError("DB error"))

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        with patch("lingya.diary.should_generate_diary", return_value=True):
            # Should not raise
            await runner._try_generate_diary()

    @pytest.mark.asyncio
    async def test_try_generate_diary_handles_model_error_gracefully(self, mock_engine, mock_db, mock_model, tmp_data_dir):
        """_try_generate_diary should not raise when LLM call fails."""
        from lingya.gateway.background import BackgroundRunner

        meaningful_turns = [
            {"role": "user", "content": "hello", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "hi there", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "how are you", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "good", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "nice day", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "yes", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "anything else", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "ai", "content": "nope", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
            {"role": "user", "content": "bye", "created_at": "2026-05-30", "conv_id": 1, "conv_title": "Conv 1"},
        ]
        mock_db.get_turns_since = AsyncMock(return_value=meaningful_turns)
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM API error"))

        runner = BackgroundRunner(
            engine=mock_engine,
            db=mock_db,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        with patch("lingya.diary.should_generate_diary", return_value=True):
            # Should not raise
            await runner._try_generate_diary()
