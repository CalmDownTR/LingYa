from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_llm():
    """Mock LLM that returns merged OCC+IPC JSON with classification fields (v0.9.7)."""
    async def call(prompt: str) -> str:
        if "w_goal" in prompt and "agency" in prompt:
            return (
                '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                '"prospect": null, "agent": null, '
                '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
            )
        if "Score the importance" in prompt:
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
        from lingya.mind.affect import ocean_to_pad_baseline

        async def noop_llm(prompt: str) -> str:
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=MagicMock(),
            llm_call=noop_llm,
        )
        baseline = ocean_to_pad_baseline(mind_config.ocean)
        assert engine.state.turn_counter == 0
        assert engine.state.current_pad.pleasure == baseline.pleasure

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
        # Should mention tone directive
        assert "语气指令" in fragment or "度指令" in fragment

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


class TestIPCStateMachine:
    """Tests for IPC state transitions (dynamics.py)."""

    def test_legal_transition_works(self):
        from lingya.mind.dynamics import IPCState, next_ipc_state

        # WARM_LISTENING → NEUTRAL is a legal transition
        result = next_ipc_state(IPCState.WARM_LISTENING, IPCState.NEUTRAL)
        assert result == IPCState.NEUTRAL

    def test_illegal_transition_returns_neutral(self):
        from lingya.mind.dynamics import IPCState, next_ipc_state

        # CRISIS_INTERVENTION → PLAYFUL_COLLABORATION is NOT in the valid set
        # CRISIS_INTERVENTION valid: {WARM_LISTENING, NEUTRAL, PROFESSIONAL_DEFENSE}
        # Should route through NEUTRAL
        result = next_ipc_state(IPCState.CRISIS_INTERVENTION, IPCState.PLAYFUL_COLLABORATION)
        assert result == IPCState.NEUTRAL, (
            f"Expected NEUTRAL (routing through NEUTRAL), got {result}"
        )

    def test_same_state_returns_self(self):
        from lingya.mind.dynamics import IPCState, next_ipc_state

        result = next_ipc_state(IPCState.PLAYFUL_COLLABORATION, IPCState.PLAYFUL_COLLABORATION)
        assert result == IPCState.PLAYFUL_COLLABORATION

    def test_neutral_can_reach_playful(self):
        from lingya.mind.dynamics import IPCState, next_ipc_state

        # NEUTRAL → PLAYFUL_COLLABORATION is legal
        result = next_ipc_state(IPCState.NEUTRAL, IPCState.PLAYFUL_COLLABORATION)
        assert result == IPCState.PLAYFUL_COLLABORATION


class TestPadHistorySlidingWindow:
    """Tests for pad_history sliding window fix (v0.9.7 #3)."""

    async def test_pad_history_capped_at_200(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        # Process 250 events — pad_history should not drop to 100
        for i in range(250):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive" if i % 2 == 0 else "negative",
                "focus": "self",
                "description": f"Event {i}",
            })

        # After 250 events, pad_history should be at most 200 (the window size)
        # NOT 100 (the buggy truncation)
        assert len(engine.state.pad_history) <= 200
        # With the fix, it should hold close to 200 entries (the window)
        # Before fix: [-100:] means it drops to 100 after exceeding 200
        assert len(engine.state.pad_history) >= 200, (
            f"Expected ~200 entries in sliding window, got {len(engine.state.pad_history)}. "
            f"Bug: [-100:] truncates to half the window."
        )


class TestReflectionFailureRollback:
    """Tests for reflection failure rollback (v0.9.7 #4)."""

    async def test_failed_reflection_preserves_cumulative_importance(
        self, mind_config, mock_memory
    ):
        """When check_and_reflect fails, cumulative_importance must not reset."""
        from lingya.mind import MindEngine

        # Mock LLM that fails for reflection
        async def selective_mock(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "score_importance" in prompt or "Score the importance" in prompt:
                return "7.0"
            # Reflection call — simulate failure
            if "guiding questions" in prompt or "self-notion" in prompt:
                raise RuntimeError("Simulated LLM failure")
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=selective_mock,
        )
        # Set threshold low so reflection triggers immediately
        engine.state.reflection_threshold = 0.1
        engine.state.cumulative_importance = 5.0

        # Make search_weighted return something so reflection is attempted
        mock_memory.search_weighted = MagicMock(return_value=[
            {"text": "test memory", "importance": 7.0}
        ])

        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "Test event",
        })

        # After failed reflection, cumulative_importance should NOT be reset to 0
        assert engine.state.cumulative_importance > 0, (
            "cumulative_importance was reset even though reflection failed"
        )

    async def test_successful_reflection_resets_cumulative_importance(
        self, mind_config, mock_memory
    ):
        """When check_and_reflect succeeds, cumulative_importance resets."""
        from lingya.mind import MindEngine

        async def success_mock(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "score_importance" in prompt or "Score the importance" in prompt:
                return "7.0"
            if "guiding questions" in prompt:
                return "1. How does the user prefer to communicate?\n2. What topics interest the user?\n3. How does the user respond to humor?"
            if "self-notion" in prompt:
                return "User prefers direct communication."
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=success_mock,
        )
        engine.state.reflection_threshold = 0.1
        engine.state.cumulative_importance = 5.0

        mock_memory.search_weighted = MagicMock(return_value=[
            {"text": "test memory", "importance": 7.0}
        ])

        initial_threshold = engine.state.reflection_threshold
        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "Test event",
        })

        # After successful reflection, cumulative_importance should reset
        assert engine.state.cumulative_importance < 5.0, (
            "cumulative_importance was not reset after successful reflection"
        )
        # Threshold should increase
        assert engine.state.reflection_threshold > initial_threshold, (
            "reflection_threshold was not increased after successful reflection"
        )


class TestReloadConfig:
    """Tests for MindEngine.reload_config() — hot reload without restart."""

    async def test_reload_ocean_updates_config_and_state(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        original_openness = engine.config.ocean.openness
        new_openness = 0.9 if original_openness < 0.8 else 0.2

        await engine.reload_config({"ocean": {"openness": new_openness}})

        assert engine.config.ocean.openness == new_openness
        assert engine.state.current_ocean.openness == new_openness

    async def test_reload_ocean_recalculates_pad_baseline(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine
        from lingya.mind.affect import ocean_to_pad_baseline

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        # Drift ocean far from current
        await engine.reload_config({"ocean": {
            "openness": 0.9, "conscientiousness": 0.1,
            "extraversion": 0.9, "agreeableness": 0.1, "neuroticism": 0.9,
        }})
        expected_baseline = ocean_to_pad_baseline(engine.config.ocean)

        assert engine.state.current_pad.pleasure == expected_baseline.pleasure
        assert engine.state.current_pad.arousal == expected_baseline.arousal
        assert engine.state.current_pad.dominance == expected_baseline.dominance

    async def test_reload_identity_rebuilds_static_prompt(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        new_identity = "Test Identity 测试身份"
        await engine.reload_config({"identity": {"identity": new_identity}})

        assert engine.config.identity.identity == new_identity
        assert new_identity in engine._static_prompt

    async def test_reload_identity_partial_update_only_identity(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        original_belief = engine.config.identity.core_belief
        new_identity = "Partial Identity Update"

        await engine.reload_config({"identity": {"identity": new_identity}})

        assert engine.config.identity.identity == new_identity
        # core_belief should be unchanged
        assert engine.config.identity.core_belief == original_belief

    async def test_reload_tone_preset_changes_tone_matrix(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        await engine.reload_config({"tone_preset": "passionate"})

        assert engine.config.tone_matrix.warmth == 90
        assert engine.config.tone_matrix.formality == 30
        assert engine.config.tone_matrix.humor == 0.4
        # _current_tone should also be updated
        assert engine._current_tone.warmth == 90
        assert engine._current_tone.formality == 30

    async def test_reload_tone_preset_invalid_raises(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        with pytest.raises(ValueError, match="Unknown tone preset"):
            await engine.reload_config({"tone_preset": "nonexistent"})

    async def test_reset_restores_original_config(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        original_openness = engine.config.ocean.openness
        original_identity = engine.config.identity.identity

        # Mutate config
        await engine.reload_config({"ocean": {"openness": 0.99}})
        await engine.reload_config({"identity": {"identity": "changed"}})
        assert engine.config.ocean.openness == 0.99
        assert engine.config.identity.identity == "changed"

        # Reset
        await engine.reload_config({"reset": True})

        assert engine.config.ocean.openness == original_openness
        assert engine.config.identity.identity == original_identity

    async def test_reset_rebuilds_static_prompt(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine, build_static_prompt

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        await engine.reload_config({"identity": {"identity": "changed"}})
        await engine.reload_config({"reset": True})

        expected_prompt = build_static_prompt(mind_config)
        assert engine._static_prompt == expected_prompt

    async def test_reload_no_db_does_not_crash(self, mind_config, mock_llm, mock_memory):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        # _db is None by default
        assert engine._db is None

        # Should not raise
        await engine.reload_config({"ocean": {"openness": 0.7}})
        assert engine.config.ocean.openness == 0.7

    async def test_reload_with_db_persists(self, mind_config, mock_llm, mock_memory, db):
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.set_db(db)

        await engine.reload_config({"ocean": {"openness": 0.88}})

        # Load into a new engine to verify persistence
        engine2 = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        restored = await engine2.load_state(db)
        assert restored is True
        assert engine2.state.current_ocean.openness == 0.88

    async def test_tone_presets_exported_from_module(self):
        from lingya.mind import TONE_PRESETS

        assert isinstance(TONE_PRESETS, dict)
        assert "warm" in TONE_PRESETS
        assert "neutral" in TONE_PRESETS
        assert "cool" in TONE_PRESETS
        assert "passionate" in TONE_PRESETS
        assert "gentle" in TONE_PRESETS
        # Verify structure
        warm = TONE_PRESETS["warm"]
        assert "warmth" in warm
        assert "formality" in warm
        assert "humor" in warm
