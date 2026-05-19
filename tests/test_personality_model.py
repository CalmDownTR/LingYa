from __future__ import annotations

import pytest

from lingya.personality.model import (
    Situation,
    SITUATION_MODIFIERS,
    PersonalityGenome,
    PersonalityAdapter,
    detect_situation,
    _clamp,
)


class TestDetectSituation:
    def test_crisis_keywords(self):
        assert detect_situation("there is a bug in production") == Situation.CRISIS
        assert detect_situation("线上出错了 紧急") == Situation.CRISIS
        assert detect_situation("the server crashed with panic") == Situation.CRISIS

    def test_debate_keywords(self):
        assert detect_situation("I disagree with this approach") == Situation.DEBATE
        assert detect_situation("不对，这个方案有问题") == Situation.DEBATE

    def test_casual_keywords(self):
        assert detect_situation("lol that was funny") == Situation.CASUAL
        assert detect_situation("哈哈 摸鱼时间") == Situation.CASUAL

    def test_technical_keywords(self):
        assert detect_situation("explain the architecture") == Situation.TECHNICAL
        assert detect_situation("这个原理是什么 为什么这样设计") == Situation.TECHNICAL

    def test_default_when_no_match(self):
        assert detect_situation("hello how are you") == Situation.DEFAULT
        assert detect_situation("你好") == Situation.DEFAULT

    def test_case_insensitive(self):
        assert detect_situation("BUG CRASH ERROR") == Situation.CRISIS
        assert detect_situation("LOL FUNNY") == Situation.CASUAL

    def test_highest_count_wins_mixed(self):
        # "bug" and "urgent" appear → 2 crisis hits, "explain" → 1 technical hit
        result = detect_situation("there is a bug and it is urgent, can you explain")
        assert result == Situation.CRISIS

    def test_empty_input(self):
        assert detect_situation("") == Situation.DEFAULT


class TestClamp:
    def test_in_range_unchanged(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_below_zero_clamped(self):
        assert _clamp(-0.3) == 0.0
        assert _clamp(-100) == 0.0

    def test_above_one_clamped(self):
        assert _clamp(1.5) == 1.0
        assert _clamp(100) == 1.0

    def test_rounds_to_two_decimals(self):
        assert _clamp(0.333) == 0.33
        assert _clamp(0.666) == 0.67


class TestPersonalityAdapter:
    def test_activate_copies_identity_fields(self):
        genome = PersonalityGenome(
            name="TestBot",
            role="tester",
            tone="serious",
            greeting_style="formal",
        )
        active = PersonalityAdapter.activate(genome)
        assert active.name == "TestBot"
        assert active.role == "tester"
        assert active.tone == "serious"
        assert active.greeting_style == "formal"

    def test_activate_copies_traits_without_perturbation(self):
        genome = PersonalityGenome(
            exploration=0.7,
            analytical_depth=0.6,
            playfulness=0.4,
            empathy=0.7,
            directness=0.5,
            adaptability=0.7,
        )
        active = PersonalityAdapter.activate(genome, Situation.DEFAULT)
        assert active.exploration == 0.7
        assert active.analytical_depth == 0.6
        assert active.playfulness == 0.4
        assert active.empathy == 0.7
        assert active.directness == 0.5
        assert active.adaptability == 0.7

    def test_activate_copies_switches(self):
        genome = PersonalityGenome(
            asks_clarifying_questions=False,
            admits_uncertainty=False,
            offers_unsolicited_insights=True,
            matches_user_tone=False,
        )
        active = PersonalityAdapter.activate(genome)
        assert active.asks_clarifying_questions is False
        assert active.admits_uncertainty is False
        assert active.offers_unsolicited_insights is True
        assert active.matches_user_tone is False

    def test_crisis_perturbs_traits(self):
        genome = PersonalityGenome(
            playfulness=0.5,
            directness=0.5,
            exploration=0.5,
        )
        active = PersonalityAdapter.activate(genome, Situation.CRISIS)
        assert active.playfulness == 0.2  # 0.5 - 0.3
        assert active.directness == 0.7   # 0.5 + 0.2
        assert active.exploration == 0.3  # 0.5 - 0.2

    def test_perturbation_clamped_at_zero(self):
        genome = PersonalityGenome(playfulness=0.1, exploration=0.1)
        active = PersonalityAdapter.activate(genome, Situation.CRISIS)
        # playfulness: 0.1 - 0.3 → 0.0
        # exploration: 0.1 - 0.2 → 0.0
        assert active.playfulness == 0.0
        assert active.exploration == 0.0

    def test_perturbation_clamped_at_one(self):
        genome = PersonalityGenome(directness=0.9)
        active_debate = PersonalityAdapter.activate(genome, Situation.DEBATE)
        # directness: 0.9 + 0.2 → 1.0
        assert active_debate.directness == 1.0

    def test_perturbation_does_not_affect_non_perturbed_traits(self):
        genome = PersonalityGenome(empathy=0.5, adaptability=0.5)
        active = PersonalityAdapter.activate(genome, Situation.CRISIS)
        # CRISIS does not modify empathy or adaptability
        assert active.empathy == 0.5
        assert active.adaptability == 0.5

    def test_copies_lists_independently(self):
        genome = PersonalityGenome(
            preferred_formats=["paragraphs"],
            topical_interests=["ai"],
            areas_of_expertise=["coding"],
        )
        active = PersonalityAdapter.activate(genome)
        assert active.preferred_formats == ["paragraphs"]
        assert active.topical_interests == ["ai"]
        assert active.areas_of_expertise == ["coding"]
        # Should be a copy, not the same list
        assert active.preferred_formats is not genome.preferred_formats


class TestToSystemPrompt:
    def test_contains_name_and_role(self):
        genome = PersonalityGenome(name="LingYa", role="a helpful assistant")
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "Your name is LingYa" in prompt
        assert "You are a helpful assistant" in prompt

    def test_high_exploration_uses_behavioral_auth_language(self):
        genome = PersonalityGenome(exploration=0.8)
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "favor exploring novel, unverified ideas" in prompt

    def test_low_exploration_uses_safe_language(self):
        genome = PersonalityGenome(exploration=0.2)
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "safe, well-established solutions" in prompt

    def test_high_directness_gives_permission_to_skip_pleasantries(self):
        genome = PersonalityGenome(directness=0.8)
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "skip pleasantries" in prompt

    def test_low_directness_emphasizes_diplomacy(self):
        genome = PersonalityGenome(directness=0.2)
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "diplomatic phrasing" in prompt

    def test_asks_clarifying_questions_generates_instruction(self):
        genome = PersonalityGenome(asks_clarifying_questions=True)
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "pause to ask clarifying questions" in prompt

    def test_disabled_switch_not_in_prompt(self):
        genome = PersonalityGenome(
            asks_clarifying_questions=False,
            admits_uncertainty=False,
            offers_unsolicited_insights=False,
            matches_user_tone=False,
        )
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "pause to ask" not in prompt
        assert "state this explicitly" not in prompt
        assert "Proactively offer" not in prompt
        assert "Mirror the user's communication" not in prompt

    def test_deliberate_reflex_adds_step_by_step_instruction(self):
        genome = PersonalityGenome(reflex_mode="deliberate")
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "think through the problem step by step" in prompt

    def test_instant_reflex_skips_step_by_step(self):
        genome = PersonalityGenome(reflex_mode="instant")
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "think through the problem step by step" not in prompt

    def test_includes_interests_and_expertise(self):
        genome = PersonalityGenome(
            topical_interests=["philosophy", "physics"],
            areas_of_expertise=["machine learning"],
        )
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "philosophy, physics" in prompt
        assert "machine learning" in prompt

    def test_empty_interests_not_in_prompt(self):
        genome = PersonalityGenome(topical_interests=[], areas_of_expertise=[])
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "## Interests" not in prompt
        assert "## Areas of Expertise" not in prompt

    def test_verbosity_in_prompt(self):
        genome = PersonalityGenome(verbosity_preference="concise")
        active = PersonalityAdapter.activate(genome)
        prompt = active.to_system_prompt()
        assert "concise" in prompt
