from __future__ import annotations

import pytest

from lingya.personality.model import (
    Situation,
    PersonalityAdapter,
    PersonalityGenome,
    detect_situation,
    _clamp,
)


class TestDetectSituation:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("there is a bug in production", Situation.CRISIS),
            ("线上出错了 紧急", Situation.CRISIS),
            ("the server crashed with panic", Situation.CRISIS),
            ("I disagree with this approach", Situation.DEBATE),
            ("不对，这个方案有问题", Situation.DEBATE),
            ("lol that was funny", Situation.CASUAL),
            ("哈哈 摸鱼时间", Situation.CASUAL),
            ("explain the architecture", Situation.TECHNICAL),
            ("这个原理是什么 为什么这样设计", Situation.TECHNICAL),
            ("hello how are you", Situation.DEFAULT),
            ("你好", Situation.DEFAULT),
            ("", Situation.DEFAULT),
        ],
    )
    def test_situation_detection(self, text, expected):
        assert detect_situation(text) == expected

    def test_case_insensitive(self):
        assert detect_situation("BUG CRASH ERROR") == Situation.CRISIS
        assert detect_situation("LOL FUNNY") == Situation.CASUAL

    def test_highest_count_wins_mixed(self):
        result = detect_situation("there is a bug and it is urgent, can you explain")
        assert result == Situation.CRISIS


class TestClamp:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 1.0),
            (-0.3, 0.0),
            (-100, 0.0),
            (1.5, 1.0),
            (100, 1.0),
        ],
    )
    def test_clamp(self, value, expected):
        assert _clamp(value) == expected

    @pytest.mark.parametrize("value,expected", [(0.333, 0.33), (0.666, 0.67)])
    def test_rounds_to_two_decimals(self, value, expected):
        assert _clamp(value) == expected


class TestPersonalityAdapter:
    def test_activate_copies_identity_fields(self, default_genome):
        active = PersonalityAdapter.activate(default_genome)
        assert active.name == default_genome.name
        assert active.role == default_genome.role
        assert active.tone == default_genome.tone
        assert active.greeting_style == default_genome.greeting_style

    def test_activate_copies_traits_without_perturbation(self, default_genome):
        active = PersonalityAdapter.activate(default_genome, Situation.DEFAULT)
        assert active.exploration == default_genome.exploration
        assert active.analytical_depth == default_genome.analytical_depth
        assert active.playfulness == default_genome.playfulness
        assert active.empathy == default_genome.empathy
        assert active.directness == default_genome.directness
        assert active.adaptability == default_genome.adaptability

    def test_activate_copies_switches(self, default_genome):
        active = PersonalityAdapter.activate(default_genome)
        assert active.asks_clarifying_questions == default_genome.asks_clarifying_questions
        assert active.admits_uncertainty == default_genome.admits_uncertainty
        assert active.offers_unsolicited_insights == default_genome.offers_unsolicited_insights
        assert active.matches_user_tone == default_genome.matches_user_tone

    @pytest.mark.parametrize(
        "situation,trait,base,expected",
        [
            (Situation.CRISIS, "playfulness", 0.5, 0.2),
            (Situation.CRISIS, "directness", 0.5, 0.7),
            (Situation.CRISIS, "exploration", 0.5, 0.3),
        ],
    )
    def test_crisis_perturbs_traits(self, situation, trait, base, expected):
        genome = PersonalityGenome(**{trait: base})
        active = PersonalityAdapter.activate(genome, situation)
        assert getattr(active, trait) == expected

    def test_perturbation_clamped_at_zero(self):
        genome = PersonalityGenome(playfulness=0.1, exploration=0.1)
        active = PersonalityAdapter.activate(genome, Situation.CRISIS)
        assert active.playfulness == 0.0
        assert active.exploration == 0.0

    def test_perturbation_clamped_at_one(self):
        genome = PersonalityGenome(directness=0.9)
        active = PersonalityAdapter.activate(genome, Situation.DEBATE)
        assert active.directness == 1.0

    def test_perturbation_does_not_affect_non_perturbed_traits(self):
        genome = PersonalityGenome(empathy=0.5, adaptability=0.5)
        active = PersonalityAdapter.activate(genome, Situation.CRISIS)
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
        assert active.preferred_formats is not genome.preferred_formats


class TestToSystemPrompt:
    @pytest.fixture
    def prompt(self, default_genome):
        return PersonalityAdapter.activate(default_genome).to_system_prompt()

    def test_contains_name_and_role(self, prompt):
        assert "Your name is LingYa" in prompt
        assert "You are a thoughtful, curious AI companion" in prompt

    def test_high_exploration_uses_behavioral_auth_language(self, genome_with_high_traits):
        prompt = PersonalityAdapter.activate(genome_with_high_traits).to_system_prompt()
        assert "favor exploring novel, unverified ideas" in prompt

    def test_low_exploration_uses_safe_language(self, genome_with_low_traits):
        prompt = PersonalityAdapter.activate(genome_with_low_traits).to_system_prompt()
        assert "safe, well-established solutions" in prompt

    def test_high_directness_gives_permission_to_skip_pleasantries(self, genome_with_high_traits):
        prompt = PersonalityAdapter.activate(genome_with_high_traits).to_system_prompt()
        assert "skip pleasantries" in prompt

    def test_low_directness_emphasizes_diplomacy(self, genome_with_low_traits):
        prompt = PersonalityAdapter.activate(genome_with_low_traits).to_system_prompt()
        assert "diplomatic phrasing" in prompt

    @pytest.mark.parametrize("switch", [
        "asks_clarifying_questions",
        "admits_uncertainty",
        "offers_unsolicited_insights",
        "matches_user_tone",
    ])
    def test_disabled_switch_not_in_prompt(self, switch):
        kwargs = {s: False for s in [
            "asks_clarifying_questions", "admits_uncertainty",
            "offers_unsolicited_insights", "matches_user_tone",
        ]}
        genome = PersonalityGenome(**kwargs)
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "pause to ask" not in prompt
        assert "state this explicitly" not in prompt
        assert "Proactively offer" not in prompt
        assert "Mirror the user's communication" not in prompt

    def test_deliberate_reflex_adds_step_by_step_instruction(self):
        genome = PersonalityGenome(reflex_mode="deliberate")
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "think through the problem step by step" in prompt

    def test_instant_reflex_skips_step_by_step(self):
        genome = PersonalityGenome(reflex_mode="instant")
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "think through the problem step by step" not in prompt

    def test_includes_interests_and_expertise(self):
        genome = PersonalityGenome(
            topical_interests=["philosophy", "physics"],
            areas_of_expertise=["machine learning"],
        )
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "philosophy, physics" in prompt
        assert "machine learning" in prompt

    def test_empty_interests_not_in_prompt(self):
        genome = PersonalityGenome(topical_interests=[], areas_of_expertise=[])
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "## Interests" not in prompt
        assert "## Areas of Expertise" not in prompt

    def test_verbosity_in_prompt(self):
        genome = PersonalityGenome(verbosity_preference="concise")
        prompt = PersonalityAdapter.activate(genome).to_system_prompt()
        assert "concise" in prompt
