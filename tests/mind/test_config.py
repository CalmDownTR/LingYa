from __future__ import annotations

import pytest


class TestMindConfig:
    def test_load_mind_config_v2_format(self):
        from lingya.mind import load_mind_config

        config = load_mind_config("agent_config.yaml")
        assert config.version == "2.0.0"
        assert config.identity.identity == "冷峻、克制的终身学术观察者"
        assert config.identity.core_belief
        assert config.ocean.openness == 0.75
        assert config.ocean.conscientiousness == 0.80
        assert config.ocean.extraversion == 0.15
        assert config.ocean.agreeableness == 0.20
        assert config.ocean.neuroticism == 0.35
        assert config.tone_matrix.warmth == 15
        assert config.tone_matrix.formality == 85
        assert config.tone_matrix.humor == 0.05
        assert len(config.behavior_guardrails) == 4

    def test_mind_state_from_config(self, mind_config):
        from lingya.mind import MindState
        from lingya.mind.affect import ocean_to_pad_baseline

        state = MindState.from_config(mind_config)
        baseline = ocean_to_pad_baseline(mind_config.ocean)
        assert state.current_pad.pleasure == baseline.pleasure
        assert state.current_pad.arousal == baseline.arousal
        assert state.current_pad.dominance == baseline.dominance
        assert state.current_ocean.openness == mind_config.ocean.openness
        assert state.turn_counter == 0
        assert state.cumulative_importance == 0.0
        assert state.ipc_state == "neutral"

    def test_mind_state_round_trip(self, mind_config):
        from lingya.mind import MindState

        state = MindState.from_config(mind_config)
        state.turn_counter = 5
        state.ipc_agency = 0.7
        state.recent_emotions = [{"emotion": "joy", "intensity": 0.5}]

        d = state.to_dict()
        restored = MindState.from_dict(d)

        assert restored.turn_counter == 5
        assert restored.ipc_agency == 0.7
        assert len(restored.recent_emotions) == 1
        assert restored.recent_emotions[0]["emotion"] == "joy"

    def test_config_validation_clamps_bigfive(self):
        from lingya.mind.config import BigFiveTraits

        # These should all be valid
        ocean = BigFiveTraits(
            openness=0.75, conscientiousness=0.8, extraversion=0.15,
            agreeableness=0.2, neuroticism=0.35,
        )
        assert ocean.openness == 0.75

    def test_config_validation_clamps_pad(self):
        from lingya.mind.config import PADBaseline

        pad = PADBaseline(pleasure=-0.1, arousal=0.3, dominance=0.6)
        assert pad.pleasure == -0.1

    def test_static_prompt_build(self, mind_config):
        from lingya.mind import build_static_prompt

        prompt = build_static_prompt(mind_config)
        assert "底层原则" in prompt
        assert mind_config.identity.identity in prompt
        assert mind_config.identity.core_belief in prompt
        assert mind_config.behavior_guardrails[0] in prompt
