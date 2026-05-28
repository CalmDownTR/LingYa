from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_llm():
    """Mock LLM that returns merged OCC+IPC JSON."""
    async def call(prompt: str) -> str:
        if "w_goal" in prompt and "agency" in prompt:
            return '{"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
        if "importance" in prompt:
            return "7.0"
        return "ok"
    return call


@pytest.fixture
def mock_memory():
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


class TestMindEngine:
    async def test_engine_creation(self, mind_config):
        from lingya.mind import MindEngine

        async def noop_llm(prompt: str) -> str:
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=MagicMock(),
            llm_call=noop_llm,
        )
        assert engine.state.turn_counter == 0
        assert engine.state.current_pad.pleasure == mind_config.pad_baseline.pleasure

    async def test_process_event_increments_turn_counter(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "用户发来了一条问候",
        })
        assert engine.state.turn_counter == 1

    async def test_process_event_updates_pad(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        initial_pleasure = engine.state.current_pad.pleasure
        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "用户发来了一条问候",
        })
        # PAD should have changed from the event
        assert engine.state.current_pad.pleasure != initial_pleasure or len(engine.state.pad_history) > 0

    async def test_process_event_records_emotion(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "用户发来了一条问候",
        })
        assert len(engine.state.recent_emotions) > 0
        assert "emotion" in engine.state.recent_emotions[0]
        assert "intensity" in engine.state.recent_emotions[0]

    async def test_get_tone_params_returns_valid_ranges(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        tone = engine.get_tone_params()
        assert 0 <= tone["warmth"] <= 100
        assert 0 <= tone["formality"] <= 100
        assert 0 <= tone["humor"] <= 1

    async def test_get_prompt_fragment_contains_content(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        fragment = engine.get_prompt_fragment()
        assert len(fragment) > 0
        # Should mention current state
        assert "当前内部状态" in fragment or "互动姿态" in fragment

    async def test_save_and_load_state(self, mind_config, mock_llm, mock_memory, db):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.state.turn_counter = 42
        engine.state.ipc_agency = 0.7

        await engine.save_state(db)
        loaded_json = await db.get_mind_state()
        assert loaded_json is not None
        assert "42" in loaded_json

        # Create a new engine and load
        engine2 = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        restored = await engine2.load_state(db)
        assert restored is True
        assert engine2.state.turn_counter == 42
        assert engine2.state.ipc_agency == 0.7

    async def test_load_state_returns_false_when_no_saved_state(self, mind_config, mock_llm, mock_memory, db):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        result = await engine.load_state(db)
        assert result is False

    async def test_multiple_events_build_history(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        for i in range(5):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive" if i % 2 == 0 else "negative",
                "focus": "self",
                "description": f"Event {i}",
            })

        assert engine.state.turn_counter == 5
        assert len(engine.state.recent_emotions) == 5
        assert len(engine.state.pad_history) == 5
