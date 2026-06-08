"""Tests for MindEngine.idle_tick() — PAD baseline drift without user interaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lingya.mind.affect import ocean_to_pad_baseline
from lingya.mind.state import PADPoint

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_memory_for_engine():
    """Mock EnhancedMemoryStore."""
    m = MagicMock()
    m.store_with_importance = MagicMock(return_value="mem_test")
    m.score_importance = AsyncMock(return_value=5.0)
    m.update_importance = MagicMock()
    m.search_weighted = MagicMock(return_value=[])
    m.get_cumulative_importance = MagicMock(return_value=0.0)
    m.store = MagicMock(return_value="mem_test")
    m.search = MagicMock(return_value=[])
    m.list_all = MagicMock(return_value=[])
    return m


@pytest.fixture
def noop_llm():
    async def call(prompt: str) -> str:
        return "ok"
    return call


class TestMindEngineIdleTick:
    async def test_idle_tick_moves_pad_toward_baseline(self, mind_config, mock_memory_for_engine, noop_llm):
        """PAD should move closer to OCEAN-derived baseline after idle_tick."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        baseline = ocean_to_pad_baseline(mind_config.ocean)

        # Set PAD far from baseline to make the movement observable
        engine.state.current_pad = PADPoint(pleasure=0.8, arousal=-0.5, dominance=0.3)
        initial_p_distance = abs(engine.state.current_pad.pleasure - baseline.pleasure)
        initial_a_distance = abs(engine.state.current_pad.arousal - baseline.arousal)
        initial_d_distance = abs(engine.state.current_pad.dominance - baseline.dominance)

        await engine.idle_tick()

        new_p_distance = abs(engine.state.current_pad.pleasure - baseline.pleasure)
        new_a_distance = abs(engine.state.current_pad.arousal - baseline.arousal)
        new_d_distance = abs(engine.state.current_pad.dominance - baseline.dominance)

        # All dimensions should have moved toward baseline
        assert new_p_distance < initial_p_distance
        assert new_a_distance < initial_a_distance
        assert new_d_distance < initial_d_distance

    async def test_idle_tick_small_movement(self, mind_config, mock_memory_for_engine, noop_llm):
        """PAD change per idle tick should be very small."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )

        engine.state.current_pad = PADPoint(pleasure=0.9, arousal=-0.8, dominance=0.7)
        initial_pad = engine.state.current_pad.model_copy(deep=True)

        await engine.idle_tick()

        # Each dimension should change by a tiny amount (roughly < 0.05 with spring_k=0.01)
        p_change = abs(engine.state.current_pad.pleasure - initial_pad.pleasure)
        a_change = abs(engine.state.current_pad.arousal - initial_pad.arousal)
        d_change = abs(engine.state.current_pad.dominance - initial_pad.dominance)

        assert p_change < 0.05
        assert a_change < 0.05
        assert d_change < 0.05

    async def test_idle_tick_no_turn_counter_increment(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should NOT increment the turn counter."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        engine.state.turn_counter = 5
        await engine.idle_tick()
        assert engine.state.turn_counter == 5

    async def test_idle_tick_no_emotion_change(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should NOT add to recent_emotions."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        initial_emotion_count = len(engine.state.recent_emotions)
        await engine.idle_tick()
        assert len(engine.state.recent_emotions) == initial_emotion_count

    async def test_idle_tick_appends_to_pad_history(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should append current PAD to pad_history."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        initial_history_len = len(engine.state.pad_history)
        await engine.idle_tick()
        assert len(engine.state.pad_history) == initial_history_len + 1

    async def test_idle_tick_trims_pad_history_to_200(self, mind_config, mock_memory_for_engine, noop_llm):
        """pad_history should not exceed 200 entries after idle ticks."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        # Simulate a full history
        engine.state.pad_history = [
            PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        ] * 200

        await engine.idle_tick()
        assert len(engine.state.pad_history) == 100  # trimmed to 100 (last 100)

    async def test_idle_tick_persists_state_when_db_set(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should auto-persist state when _db is set."""
        from lingya.mind import MindEngine

        # Create a mock DB with async upsert_mind_state
        mock_db = MagicMock()
        mock_db.upsert_mind_state = AsyncMock()

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        engine.set_db(mock_db)
        await engine.idle_tick()
        mock_db.upsert_mind_state.assert_called_once()

    async def test_idle_tick_does_not_persist_without_db(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should not crash when _db is None."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        # _db is None by default — should not crash
        await engine.idle_tick()
        # Should complete without error

    async def test_multiple_idle_ticks_moves_closer_to_baseline(self, mind_config, mock_memory_for_engine, noop_llm):
        """After multiple idle ticks, PAD should be closer to baseline than after one."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        baseline = ocean_to_pad_baseline(mind_config.ocean)

        engine.state.current_pad = PADPoint(pleasure=0.8, arousal=-0.5, dominance=0.3)

        await engine.idle_tick()
        after_one_p_distance = abs(engine.state.current_pad.pleasure - baseline.pleasure)

        for _ in range(9):
            await engine.idle_tick()

        after_ten_p_distance = abs(engine.state.current_pad.pleasure - baseline.pleasure)
        # After 10 ticks, should be closer than after 1
        assert after_ten_p_distance < after_one_p_distance

    async def test_idle_tick_no_ipc_change(self, mind_config, mock_memory_for_engine, noop_llm):
        """idle_tick should NOT change IPC state."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory_for_engine,
            llm_call=noop_llm,
        )
        engine.state.ipc_agency = 0.7
        engine.state.ipc_communion = 0.3
        engine.state.ipc_state = "agentic"

        await engine.idle_tick()

        assert engine.state.ipc_agency == 0.7
        assert engine.state.ipc_communion == 0.3
        assert engine.state.ipc_state == "agentic"
