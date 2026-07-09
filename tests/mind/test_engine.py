from __future__ import annotations

import asyncio
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


class TestIdentityGuardReanchor:
    """Tests for identity guard reanchor hint injection (v0.9.9 #1)."""

    async def test_prompt_fragment_injects_reanchor_hint_when_needed(
        self, mind_config, mock_llm, mock_memory
    ):
        """When reanchor_needed=True, get_prompt_fragment() must include the hint."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.state.reanchor_needed = True
        engine.state.reanchor_hint = "Remember: you are a test assistant, stay aligned."

        fragment = engine.get_prompt_fragment()

        assert "Remember: you are a test assistant" in fragment
        assert "身份重锚" in fragment or "reanchor" in fragment.lower()

    async def test_reanchor_flag_cleared_after_injection(
        self, mind_config, mock_llm, mock_memory
    ):
        """After get_prompt_fragment() injects the hint, reanchor_needed must be cleared."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.state.reanchor_needed = True
        engine.state.reanchor_hint = "Reanchor test hint."

        engine.get_prompt_fragment()

        assert engine.state.reanchor_needed is False, (
            "reanchor_needed should be cleared after one-time injection"
        )

    async def test_prompt_fragment_no_reanchor_when_not_needed(
        self, mind_config, mock_llm, mock_memory
    ):
        """When reanchor_needed=False, prompt should NOT contain reanchor hint."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.state.reanchor_needed = False
        engine.state.reanchor_hint = "Should not appear."

        fragment = engine.get_prompt_fragment()

        assert "Should not appear" not in fragment

    async def test_check_response_alignment_detects_drift(
        self, mind_config, mock_llm, mock_memory
    ):
        """check_response_alignment should return False when response drifts from identity."""
        from lingya.mind import MindEngine

        identity_text = mind_config.identity.identity

        # Mock embedding that returns orthogonal vectors (cosine similarity ~0)
        # Identity gets [1,0,0], anything else gets [0,1,0] → orthogonal → drift
        def mock_embedding(text: str) -> list[float]:
            if text == identity_text:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]  # Orthogonal → low similarity → drift detected

        async def reanchor_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "safety guard" in prompt or "re-anchoring" in prompt:
                return "Remember your identity as a test assistant."
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=reanchor_llm,
            embedding_fn=mock_embedding,
        )

        # Response text different from identity → orthogonal embedding → drift detected
        result = await engine.check_response_alignment("I am a completely different entity.")

        assert result is False, "Should detect drift when response is far from identity"
        assert engine.state.reanchor_needed is True
        assert len(engine.state.reanchor_hint) > 0

    async def test_check_response_alignment_no_drift(
        self, mind_config, mock_llm, mock_memory
    ):
        """check_response_alignment should return True when response is aligned."""
        from lingya.mind import MindEngine

        # Mock embedding that returns same vector (cosine similarity = 1.0)
        def mock_embedding(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]  # Always identical → high similarity

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
            embedding_fn=mock_embedding,
        )

        engine.state.reanchor_needed = False
        result = await engine.check_response_alignment("Any response text.")

        assert result is True, "Should not detect drift when embeddings are identical"

    async def test_check_response_alignment_returns_true_without_embedding_fn(
        self, mind_config, mock_llm, mock_memory
    ):
        """Without embedding_fn, check_response_alignment should always return True."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
            embedding_fn=None,
        )

        result = await engine.check_response_alignment("Any text.")
        assert result is True

    async def test_consecutive_reanchor_warning(
        self, mind_config, mock_llm, mock_memory, caplog
    ):
        """3 consecutive reanchor failures should log a warning."""
        from lingya.mind import MindEngine
        import logging

        identity_text = mind_config.identity.identity

        # Mock embedding that always detects drift
        def mock_embedding(text: str) -> list[float]:
            if text == identity_text:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]  # Orthogonal → drift

        async def reanchor_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "safety guard" in prompt or "re-anchoring" in prompt:
                return "Reanchor hint."
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=reanchor_llm,
            embedding_fn=mock_embedding,
        )

        caplog.set_level(logging.WARNING)

        # 3 consecutive drift detections
        for _ in range(3):
            await engine.check_response_alignment("Drifted response.")

        # Should have logged a warning about 3 consecutive reanchor failures
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        reanchor_warnings = [r for r in warnings if "reanchor" in r.message.lower()]
        assert len(reanchor_warnings) >= 1, (
            f"Expected at least 1 reanchor warning after 3 consecutive failures, "
            f"got {len(reanchor_warnings)} warnings: {[r.message for r in warnings]}"
        )

    async def test_successful_alignment_resets_reanchor_counter(
        self, mind_config, mock_llm, mock_memory
    ):
        """A successful alignment check should reset the consecutive failure counter."""
        from lingya.mind import MindEngine

        identity_text = mind_config.identity.identity
        call_count = 0

        def mock_embedding(text: str) -> list[float]:
            nonlocal call_count
            # First 2 calls: drift (orthogonal vectors)
            # 3rd call: aligned (same vector)
            if text == identity_text:
                return [1.0, 0.0, 0.0]
            call_count += 1
            if call_count <= 2:
                return [0.0, 1.0, 0.0]  # Orthogonal → drift
            return [1.0, 0.0, 0.0]  # Same → aligned

        async def reanchor_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "safety guard" in prompt or "re-anchoring" in prompt:
                return "Reanchor hint."
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=reanchor_llm,
            embedding_fn=mock_embedding,
        )

        # 2 drifts
        await engine.check_response_alignment("Drift 1")
        await engine.check_response_alignment("Drift 2")
        assert engine._reanchor_failure_count == 2

        # 1 successful alignment → counter resets
        await engine.check_response_alignment("Aligned response.")
        assert engine._reanchor_failure_count == 0, (
            f"Counter should reset after successful alignment, got {engine._reanchor_failure_count}"
        )


class TestConversationRecording:
    """Tests for conversation turn recording + transcript generation (v0.9.9 #2)."""

    async def test_record_conversation_turn_stores_turn(
        self, mind_config, mock_llm, mock_memory
    ):
        """record_conversation_turn should store user + assistant messages."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.record_conversation_turn("Hello", "Hi there!")
        engine.record_conversation_turn("How are you?", "I'm good, thanks.")

        assert len(engine._recent_turns) == 2
        assert engine._recent_turns[0]["user"] == "Hello"
        assert engine._recent_turns[0]["assistant"] == "Hi there!"
        assert "timestamp" in engine._recent_turns[0]

    async def test_recent_turns_capped_at_500(
        self, mind_config, mock_llm, mock_memory
    ):
        """The conversation ring buffer should cap at 500 turns."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        for i in range(600):
            engine.record_conversation_turn(f"User {i}", f"Assistant {i}")

        assert len(engine._recent_turns) == 500
        # Should keep the most recent
        assert engine._recent_turns[0]["user"] == "User 100"
        assert engine._recent_turns[-1]["user"] == "User 599"

    async def test_get_recent_transcript_formats_turns(
        self, mind_config, mock_llm, mock_memory
    ):
        """get_recent_transcript should format turns as readable transcript."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.record_conversation_turn("你好", "你好！")
        engine.record_conversation_turn("今天天气不错", "是的，很适合出去走走。")

        transcript = engine.get_recent_transcript(hours=24)

        assert "TR:" in transcript or "用户:" in transcript
        assert "LingYa:" in transcript or "灵芽:" in transcript
        assert "你好" in transcript
        assert "今天天气不错" in transcript

    async def test_get_recent_transcript_empty_returns_message(
        self, mind_config, mock_llm, mock_memory
    ):
        """When no turns are recorded, get_recent_transcript returns a placeholder."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        transcript = engine.get_recent_transcript(hours=24)

        assert len(transcript) > 0
        assert "无" in transcript or "没有" in transcript or "No" in transcript


class TestOceanDriftDamping:
    """Tests for OCEAN drift damping (v0.9.9 #3)."""

    async def test_original_ocean_snapshot_on_first_boot(
        self, mind_config, mock_llm, mock_memory
    ):
        """MindState.from_config must snapshot original_ocean from config."""
        from lingya.mind import MindEngine, MindState

        state = MindState.from_config(mind_config)
        assert state.original_ocean is not None, (
            "original_ocean must be set on first boot"
        )
        assert state.original_ocean.openness == mind_config.ocean.openness
        assert state.original_ocean.conscientiousness == mind_config.ocean.conscientiousness

    async def test_ocean_drift_regression_force(
        self, mind_config, mock_llm, mock_memory
    ):
        """ocean_drift with original_ocean should pull toward original values."""
        from lingya.mind.affect import ocean_drift
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        original = BigFiveTraits(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5,
        )
        # Current ocean has drifted far from original
        drifted = BigFiveTraits(
            openness=0.8, conscientiousness=0.8, extraversion=0.8,
            agreeableness=0.2, neuroticism=0.2,
        )

        # Create pad_history with extreme positive PAD (drives further from original)
        pad_history = [PADPoint(pleasure=0.9, arousal=0.5, dominance=0.9)] * 50

        # With regression force
        result_with_regression = ocean_drift(
            drifted, pad_history, original_ocean=original, epsilon=0.01,
        )

        # Without regression force
        result_without_regression = ocean_drift(
            drifted, pad_history, original_ocean=None, epsilon=0.01,
        )

        # The regression version should stay closer to original values
        # For openness: without regression goes higher (positive PAD), with regression less so
        assert result_with_regression.openness <= result_without_regression.openness, (
            "Regression force should dampen drift toward extremes"
        )

    async def test_ocean_drift_regression_is_weaker_than_event_drive(
        self, mind_config, mock_llm, mock_memory
    ):
        """Regression force must be far smaller than event-driven force."""
        from lingya.mind.affect import ocean_drift
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        original = BigFiveTraits()
        current = BigFiveTraits(openness=0.3, extraversion=0.3)  # Below original

        # PAD history that drives openness UP (positive pleasure → positive drift)
        pad_history = [PADPoint(pleasure=0.9, arousal=0.0, dominance=0.0)] * 50

        # With strong event drive, the net drift should still be able to move away from original
        # even with regression — just more slowly
        result = ocean_drift(
            current, pad_history, original_ocean=original, epsilon=0.05,
        )

        # Event drive should still dominate — openness should increase from 0.3
        # even though regression pulls back toward 0.5
        assert result.openness > 0.3, (
            f"Event drive should still move OCEAN, got openness={result.openness}"
        )

    async def test_ocean_drift_backward_compat_no_original(
        self, mind_config, mock_llm, mock_memory
    ):
        """When original_ocean is None, ocean_drift must behave exactly as before."""
        from lingya.mind.affect import ocean_drift
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        ocean = BigFiveTraits()
        pad_history = [PADPoint(pleasure=0.5, arousal=0.0, dominance=0.0)] * 30

        result = ocean_drift(ocean, pad_history, original_ocean=None)
        # Should not crash, should return valid BigFiveTraits
        assert 0.0 <= result.openness <= 1.0
        assert 0.0 <= result.extraversion <= 1.0


class TestToneMatrixEvolution:
    """Tests for tone_matrix evolution (v0.9.9 #4)."""

    async def test_tone_matrix_has_original_reference(
        self, mind_config, mock_llm, mock_memory
    ):
        """Engine must track original tone_matrix for cap enforcement."""
        from lingya.mind import MindEngine

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        assert engine._original_tone_matrix is not None
        assert engine._original_tone_matrix.warmth == mind_config.tone_matrix.warmth
        assert engine._original_tone_matrix.formality == mind_config.tone_matrix.formality

    async def test_tone_evolution_stays_within_cap(
        self, mind_config, mock_memory
    ):
        """After many events, tone warmth/formality must stay within ±20% of original."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        original_warmth = engine._original_tone_matrix.warmth
        original_formality = engine._original_tone_matrix.formality

        # Process 100 events to trigger multiple tone evolution cycles
        for i in range(100):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive" if i % 3 != 0 else "negative",
                "focus": "self",
                "description": f"Tone evolution test {i}",
            })

        current_warmth = engine.config.tone_matrix.warmth
        current_formality = engine.config.tone_matrix.formality

        # Must stay within ±20% of original
        warmth_min = max(0, int(original_warmth * 0.8))
        warmth_max = min(100, int(original_warmth * 1.2))
        formality_min = max(0, int(original_formality * 0.8))
        formality_max = min(100, int(original_formality * 1.2))

        assert warmth_min <= current_warmth <= warmth_max, (
            f"Warmth {current_warmth} outside cap [{warmth_min}, {warmth_max}]"
        )
        assert formality_min <= current_formality <= formality_max, (
            f"Formality {current_formality} outside cap [{formality_min}, {formality_max}]"
        )

    async def test_tone_evolution_humor_is_fixed(
        self, mind_config, mock_memory
    ):
        """Humor should NOT evolve — only warmth and formality change."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        original_humor = engine.config.tone_matrix.humor

        for _ in range(50):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": "Humor stability test",
            })

        # Humor should not have changed
        assert engine.config.tone_matrix.humor == original_humor, (
            f"Humor changed from {original_humor} to {engine.config.tone_matrix.humor}"
        )

    async def test_tone_evolution_rate_is_slow(
        self, mind_config, mock_memory
    ):
        """Tone evolution should be very slow — 1/10 of OCEAN drift rate."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        initial_warmth = engine.config.tone_matrix.warmth
        initial_formality = engine.config.tone_matrix.formality

        # 10 events = 1 evolution cycle
        for _ in range(10):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": "Rate test",
            })

        # After just 1 cycle, changes should be tiny (max 1-2 points)
        warmth_change = abs(engine.config.tone_matrix.warmth - initial_warmth)
        formality_change = abs(engine.config.tone_matrix.formality - initial_formality)

        assert warmth_change <= 2, (
            f"Warmth changed by {warmth_change}, expected <= 2 in one cycle"
        )
        assert formality_change <= 2, (
            f"Formality changed by {formality_change}, expected <= 2 in one cycle"
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


class TestConcurrencyLock:
    """Tests for asyncio.Lock protecting state mutations (v0.9.8 #1)."""

    async def test_concurrent_idle_tick_and_process_event_no_lost_updates(
        self, mind_config, mock_memory
    ):
        """Simultaneous idle_tick and process_event must not lose state updates."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "7.0"
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        # Set PAD off-baseline so idle_tick has visible effect
        from lingya.mind.state import PADPoint
        engine.state.current_pad = PADPoint(pleasure=0.8, arousal=-0.5, dominance=0.3)

        initial_history_len = len(engine.state.pad_history)

        # Fire process_event and idle_tick concurrently
        await asyncio.gather(
            engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": "Concurrent event",
            }),
            engine.idle_tick(),
        )

        # Both should have appended to pad_history (2 new entries)
        assert len(engine.state.pad_history) == initial_history_len + 2, (
            f"Expected {initial_history_len + 2} pad_history entries, "
            f"got {len(engine.state.pad_history)}. Concurrent updates may have been lost."
        )

    async def test_lock_prevents_race_on_save_state(
        self, mind_config, mock_memory
    ):
        """Lock should ensure save_state completes before the next mutation."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "7.0"
            return "ok"

        # Use a real async mock DB that tracks call count
        mock_db = MagicMock()
        save_call_count = 0

        async def tracked_upsert(state_json: str):
            nonlocal save_call_count
            save_call_count += 1
            # Small delay to simulate I/O
            await asyncio.sleep(0.001)

        mock_db.upsert_mind_state = tracked_upsert

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.set_db(mock_db)

        # Fire 5 concurrent pairs
        tasks = []
        for i in range(5):
            tasks.append(engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": f"Event {i}",
            }))
            tasks.append(engine.idle_tick())

        await asyncio.gather(*tasks)

        # Each process_event saves once, each idle_tick saves once = 10 saves
        # (reflection doesn't trigger since threshold is 150 and cumulative is small)
        assert save_call_count == 10, (
            f"Expected 10 save_state calls, got {save_call_count}"
        )

    async def test_concurrent_process_events_dont_corrupt_state(
        self, mind_config, mock_memory
    ):
        """Multiple concurrent process_event calls must not corrupt turn_counter."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            return "ok"

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        await asyncio.gather(*[
            engine.process_event({
                "event_type": "outcome",
                "valence": "positive" if i % 2 == 0 else "negative",
                "focus": "self",
                "description": f"Event {i}",
            })
            for i in range(10)
        ])

        # Turn counter should equal the number of events
        assert engine.state.turn_counter == 10, (
            f"Expected turn_counter=10, got {engine.state.turn_counter}"
        )


class TestReflectionThresholdCap:
    """Tests for reflection_threshold cap at 1000 (v0.9.8 #2)."""

    async def test_threshold_capped_at_1000(
        self, mind_config, mock_memory
    ):
        """After many successful reflections, threshold must not exceed 1000."""
        from lingya.mind import MindEngine

        async def success_mock(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            if "guiding questions" in prompt:
                return "1. How does the user prefer to communicate?\n2. What topics interest the user?"
            if "self-notion" in prompt:
                return "User prefers direct communication."
            return "ok"

        # Make search_weighted return something so reflection succeeds
        mock_memory.search_weighted = MagicMock(return_value=[
            {"text": "test memory", "importance": 7.0}
        ])

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=success_mock,
        )
        # Start from a value that would exceed 1000 after 50 reflections
        # 50 * 1.1 multipliers on 100 → 100 * 1.1^50 ≈ 11,739 without cap
        engine.state.reflection_threshold = 100.0

        for i in range(60):
            # Each event: set cumulative high enough to trigger reflection
            engine.state.cumulative_importance = engine.state.reflection_threshold + 1.0
            # Temporarily set threshold low enough that cumulative triggers it
            # The lock in process_event captures threshold before the LLM call,
            # so set cumulative > threshold for the check to pass
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": f"Reflection cap test {i}",
            })

        assert engine.state.reflection_threshold <= 1000.0, (
            f"reflection_threshold {engine.state.reflection_threshold} exceeds cap of 1000"
        )

    async def test_threshold_respects_cap_boundary(self, mind_config, mock_memory):
        """Threshold at exactly 1000 should stay at 1000 after next success."""
        from lingya.mind import MindEngine

        async def success_mock(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "5.0"
            if "guiding questions" in prompt:
                return "1. Test question?"
            if "self-notion" in prompt:
                return "Test notion."
            return "ok"

        mock_memory.search_weighted = MagicMock(return_value=[
            {"text": "test memory", "importance": 7.0}
        ])

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=success_mock,
        )
        # Set threshold near cap
        engine.state.reflection_threshold = 950.0
        engine.state.cumulative_importance = 1000.0

        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "Cap boundary test",
        })

        # 950 * 1.1 = 1045 → capped at 1000
        assert engine.state.reflection_threshold == 1000.0, (
            f"Expected threshold capped at 1000.0, got {engine.state.reflection_threshold}"
        )


class TestImportanceScoringObservability:
    """Tests for importance scoring observability (v0.9.8 #3)."""

    async def test_failed_importance_logs_warning_and_counts(self, mind_config, mock_memory):
        """LLM failure in deferred importance must log warning and increment counter."""
        from lingya.mind import MindEngine

        async def failing_importance_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                raise RuntimeError("Simulated scoring failure")
            return "ok"

        # score_importance must be an async mock that raises
        mock_memory.score_importance = AsyncMock(
            side_effect=RuntimeError("Simulated scoring failure")
        )

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=failing_importance_llm,
        )

        # Process an event — triggers deferred importance scoring
        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "Test event for observability",
        })

        # Allow the background task to complete
        await asyncio.sleep(0.1)

        health = engine.get_health()
        scoring = health["importance_scoring"]

        assert scoring["total"] == 1
        assert scoring["failures"] == 1
        assert scoring["success_rate"] == 0.0
        assert len(scoring["recent_failure_reasons"]) == 1
        assert "Simulated scoring failure" in scoring["recent_failure_reasons"][0]

    async def test_successful_importance_tracks_llm_score(self, mind_config, mock_memory):
        """Successful scoring must track pre-score and LLM-score averages."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "8.0"
            return "ok"

        mock_memory.score_importance = AsyncMock(return_value=8.0)

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )

        await engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "Test event",
        })

        # Allow background task to complete
        await asyncio.sleep(0.1)

        health = engine.get_health()
        scoring = health["importance_scoring"]

        assert scoring["total"] == 1
        assert scoring["failures"] == 0
        assert scoring["success_rate"] == 1.0
        assert scoring["avg_llm_score"] == 8.0

    async def test_mixed_success_and_failure_tracks_correctly(self, mind_config, mock_memory):
        """Mix of successes and failures should produce correct aggregate stats."""
        from lingya.mind import MindEngine

        call_count = 0

        async def flaky_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            return "ok"

        async def flaky_score_importance(text, llm_call):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Even calls fail
                raise RuntimeError(f"Failure #{call_count}")
            return 7.0

        mock_memory.score_importance = flaky_score_importance

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=flaky_llm,
        )

        for i in range(4):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": f"Event {i}",
            })

        await asyncio.sleep(0.1)

        health = engine.get_health()
        scoring = health["importance_scoring"]

        assert scoring["total"] == 4
        assert scoring["failures"] == 2
        assert scoring["success_rate"] == 0.5
        assert len(scoring["recent_failure_reasons"]) == 2

    async def test_failure_reasons_capped_at_10(self, mind_config, mock_memory):
        """Failure reasons list must not exceed 10 entries."""
        from lingya.mind import MindEngine

        call_count = 0

        async def always_failing_llm(prompt: str) -> str:
            nonlocal call_count
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                call_count += 1
                raise RuntimeError(f"Failure #{call_count}")
            return "ok"

        mock_memory.score_importance = AsyncMock(
            side_effect=[RuntimeError(f"Failure #{i}") for i in range(1, 16)]
        )

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=always_failing_llm,
        )

        for i in range(15):
            await engine.process_event({
                "event_type": "outcome",
                "valence": "positive",
                "focus": "self",
                "description": f"Event {i}",
            })

        await asyncio.sleep(0.2)

        health = engine.get_health()
        reasons = health["importance_scoring"]["recent_failure_reasons"]

        assert len(reasons) == 10, (
            f"Failure reasons should be capped at 10, got {len(reasons)}"
        )
        # Should contain the most recent failures (6-15)
        assert "Failure #15" in reasons[-1]

    async def test_get_health_includes_reflection_and_turn_info(self, mind_config, mock_memory):
        """get_health() must include reflection_threshold, cumulative_importance, turn_counter."""
        from lingya.mind import MindEngine

        async def mock_llm(prompt: str) -> str:
            if "w_goal" in prompt and "agency" in prompt:
                return (
                    '{"event_type": "outcome", "valence": "positive", "focus": "self", '
                    '"prospect": null, "agent": null, '
                    '"w_goal": 0.5, "p_expected": 0.3, "agency": 0.6, "communion": 0.5}'
                )
            if "Score the importance" in prompt:
                return "6.0"
            return "ok"

        mock_memory.score_importance = AsyncMock(return_value=6.0)

        engine = MindEngine(
            config=mind_config,
            memory_store=mock_memory,
            llm_call=mock_llm,
        )
        engine.state.turn_counter = 42
        engine.state.reflection_threshold = 200.0
        engine.state.cumulative_importance = 75.0

        health = engine.get_health()

        assert health["reflection_threshold"] == 200.0
        assert health["cumulative_importance"] == 75.0
        assert health["turn_counter"] == 42
