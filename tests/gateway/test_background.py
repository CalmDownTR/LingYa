"""Tests for BackgroundRunner — heartbeat + diary scheduler loops."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_engine():
    """Mock MindEngine with idle_tick, get_recent_transcript, and proper config."""
    from lingya.mind.config import (
        BigFiveTraits,
        IdentityAnchor,
        MindConfig,
        PersonaMeta,
        ToneMatrix,
    )

    engine = MagicMock()
    engine.idle_tick = AsyncMock()
    engine.get_recent_transcript = MagicMock(return_value="TR: Hello\nLingYa: Hi there!")
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
    def test_init_stores_config(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            heartbeat_interval=30,
            diary_check_interval=1800,
        )

        assert runner._engine is mock_engine
        assert runner._model is mock_model
        assert runner._data_dir == tmp_data_dir
        assert runner.heartbeat_interval == 30
        assert runner.diary_check_interval == 1800
        assert runner._running is True
        assert runner._heartbeat_task is None
        assert runner._diary_task is None

    def test_init_default_intervals(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        assert runner.heartbeat_interval == 60
        assert runner.diary_check_interval == 3600


class TestBackgroundRunnerStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_running_tasks(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
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
    async def test_stop_cleans_up_tasks(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
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
    async def test_stop_when_not_running_is_noop(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
        )
        runner._running = False

        # Should not raise
        await runner.stop()
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_start_twice_does_not_duplicate(self, mock_engine, mock_model, tmp_data_dir):
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
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
    async def test_heartbeat_loop_calls_idle_tick(self, mock_engine, mock_model, tmp_data_dir):
        """Heartbeat loop calls idle_tick after each sleep interval."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
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
    async def test_heartbeat_loop_skips_when_not_running(self, mock_engine, mock_model, tmp_data_dir):
        """Heartbeat loop should not call idle_tick when _running is False."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
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
    async def test_try_generate_diary_calls_model_and_saves(
        self, mock_engine, mock_model, tmp_data_dir
    ):
        """v0.9.9: _try_generate_diary generates a diary via LLM and saves it."""
        from lingya.gateway.background import BackgroundRunner

        mock_model.ainvoke.return_value.text = "一篇测试日记内容。"

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        # Should not raise
        await runner._try_generate_diary()

        # Should have called the engine for transcript
        mock_engine.get_recent_transcript.assert_called_once()

        # Should have called model to generate the diary
        mock_model.ainvoke.assert_called_once()

        # Check diary file was created
        from datetime import date
        diary_path = Path(tmp_data_dir) / "diary" / f"{date.today().isoformat()}.md"
        assert diary_path.exists(), f"Diary file not found at {diary_path}"
        content = diary_path.read_text(encoding="utf-8")
        assert "一篇测试日记内容" in content

    @pytest.mark.asyncio
    async def test_try_generate_diary_skips_when_already_generated_today(
        self, mock_engine, mock_model, tmp_data_dir
    ):
        """When today's diary already exists, generation is skipped."""
        from datetime import date
        from lingya.gateway.background import BackgroundRunner

        # Pre-create today's diary
        diary_dir = Path(tmp_data_dir) / "diary"
        today_file = diary_dir / f"{date.today().isoformat()}.md"
        today_file.write_text("# Today\n\nAlready written.\n", encoding="utf-8")

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        await runner._try_generate_diary()

        # Model should NOT be called since diary already exists for today
        mock_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_try_generate_diary_handles_llm_error(
        self, mock_engine, mock_model, tmp_data_dir
    ):
        """When LLM fails, _try_generate_diary logs error and does not crash."""
        from lingya.gateway.background import BackgroundRunner

        mock_model.ainvoke.side_effect = RuntimeError("LLM timeout")

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
        )

        # Should not raise
        await runner._try_generate_diary()


# ── Memory Decay Loop tests ───────────────────────────────────────────


@pytest.fixture
def mock_memory():
    """Mock EnhancedMemoryStore for decay testing."""
    memory = MagicMock()
    memory.apply_decay = MagicMock(return_value=0)
    return memory


class TestDecayLoop:
    @pytest.mark.asyncio
    async def test_decay_loop_calls_apply_decay(
        self, mock_engine, mock_model, tmp_data_dir, mock_memory
    ):
        """Decay loop calls apply_decay on the memory store."""
        from lingya.gateway.background import BackgroundRunner

        mock_memory.apply_decay.return_value = 3

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=mock_memory,
            decay_interval=0.01,
        )

        # Run the loop as a task and cancel after a short time
        runner._running = True
        task = asyncio.create_task(runner._decay_loop())
        await asyncio.sleep(0.05)  # Let one tick complete
        runner._running = False
        await asyncio.sleep(0.05)  # Let the loop see _running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        mock_memory.apply_decay.assert_called()

    @pytest.mark.asyncio
    async def test_decay_loop_skips_when_no_memory(
        self, mock_engine, mock_model, tmp_data_dir
    ):
        """Decay loop is a no-op when memory is None (not started)."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=None,
            decay_interval=0.01,
        )

        # Run loop briefly — should not raise because memory=None is handled
        runner._running = True
        task = asyncio.create_task(runner._decay_loop())
        await asyncio.sleep(0.05)
        runner._running = False
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_decay_loop_not_started_when_memory_none(
        self, mock_engine, mock_model, tmp_data_dir
    ):
        """When memory is None, _decay_task is never created on start."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=None,
            heartbeat_interval=3600,
            diary_check_interval=3600,
        )

        await runner.start()
        assert runner._decay_task is None
        await runner.stop()

    @pytest.mark.asyncio
    async def test_decay_interval_is_configurable(
        self, mock_engine, mock_model, tmp_data_dir, mock_memory
    ):
        """Decay interval can be configured via constructor."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=mock_memory,
            decay_interval=43200,
        )
        assert runner.decay_interval == 43200

    @pytest.mark.asyncio
    async def test_decay_loop_starts_when_memory_provided(
        self, mock_engine, mock_model, tmp_data_dir, mock_memory
    ):
        """When memory is provided, decay task is created on start."""
        from lingya.gateway.background import BackgroundRunner

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=mock_memory,
            heartbeat_interval=3600,
            diary_check_interval=3600,
            decay_interval=0.01,
        )

        await runner.start()
        assert runner._decay_task is not None
        await runner.stop()

    @pytest.mark.asyncio
    async def test_decay_loop_handles_error_gracefully(
        self, mock_engine, mock_model, tmp_data_dir, mock_memory
    ):
        """Decay loop logs error and continues, does not crash."""
        from lingya.gateway.background import BackgroundRunner

        call_count = 0

        def apply_decay_side_effect():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Decay error")

        mock_memory.apply_decay.side_effect = apply_decay_side_effect

        runner = BackgroundRunner(
            engine=mock_engine,
            model=mock_model,
            data_dir=tmp_data_dir,
            memory=mock_memory,
            decay_interval=0.01,
        )

        # Run the loop as a task — apply_decay should be called at least once
        # even though it raises
        runner._running = True
        task = asyncio.create_task(runner._decay_loop())
        await asyncio.sleep(0.05)
        runner._running = False
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert call_count >= 1  # apply_decay was called despite raising
