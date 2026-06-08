from __future__ import annotations

import pytest


class TestOCCClassify:
    def test_joy_for_positive_self_outcome(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "focus": "self"}
        assert occ_classify(event) == "joy"

    def test_distress_for_negative_self_outcome(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "negative", "focus": "self"}
        assert occ_classify(event) == "distress"

    def test_hope_for_positive_prospect(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "prospect": "prospective"}
        assert occ_classify(event) == "hope"

    def test_fear_for_negative_prospect(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "negative", "prospect": "prospective"}
        assert occ_classify(event) == "fear"

    def test_satisfaction_for_confirmed_positive(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "prospect": "actual", "confirmed": True}
        assert occ_classify(event) == "satisfaction"

    def test_relief_for_disconfirmed_negative(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "prospect": "actual", "confirmed": False}
        assert occ_classify(event) == "relief"

    def test_fears_confirmed(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "negative", "prospect": "actual", "confirmed": True}
        assert occ_classify(event) == "fears-confirmed"

    def test_disappointment(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "negative", "prospect": "actual", "confirmed": False}
        assert occ_classify(event) == "disappointment"

    def test_pride_for_positive_self_action(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "positive", "agent": "self"}
        assert occ_classify(event) == "pride"

    def test_shame_for_negative_self_action(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "negative", "agent": "self"}
        assert occ_classify(event) == "shame"

    def test_admiration_for_positive_other_action(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "positive", "agent": "other"}
        assert occ_classify(event) == "admiration"

    def test_reproach_for_negative_other_action(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "negative", "agent": "other"}
        assert occ_classify(event) == "reproach"

    def test_love_for_positive_object(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "object", "valence": "positive"}
        assert occ_classify(event) == "love"

    def test_hate_for_negative_object(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "object", "valence": "negative"}
        assert occ_classify(event) == "hate"

    def test_happy_for_deserving_other(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "focus": "other", "deserving": True}
        assert occ_classify(event) == "happy-for"

    def test_gloating_for_undeserving_other(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "outcome", "valence": "positive", "focus": "other", "deserving": False}
        assert occ_classify(event) == "gloating"

    def test_compound_gratitude(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "positive", "focus": "self", "agent": "other"}
        assert occ_classify(event) in ("gratitude", "admiration")

    def test_compound_anger(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "action", "valence": "negative", "focus": "self", "agent": "other"}
        assert occ_classify(event) in ("anger", "reproach")

    def test_neutral_fallback(self):
        from lingya.mind.affect import occ_classify

        event = {"event_type": "unknown"}
        assert occ_classify(event) in ("neutral", "interest", "disgust")


class TestComputeIntensity:
    def test_high_goal_low_expected_gives_high_intensity(self):
        from lingya.mind.affect import compute_intensity

        intensity = compute_intensity(w_goal=0.9, p_expected=0.1)
        assert intensity > 0.7

    def test_low_goal_high_expected_gives_low_intensity(self):
        from lingya.mind.affect import compute_intensity

        intensity = compute_intensity(w_goal=0.1, p_expected=0.9)
        assert intensity < 0.1

    def test_intensity_clamped_to_1(self):
        from lingya.mind.affect import compute_intensity

        intensity = compute_intensity(w_goal=1.0, p_expected=0.0, e_residual=2.0)
        assert intensity <= 1.0


class TestPADEvolution:
    def test_evolve_pad_pulls_toward_occ_direction(self):
        from lingya.mind.affect import evolve_pad
        from lingya.mind.config import PADBaseline
        from lingya.mind.state import PADPoint

        current = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        occ_pull = PADPoint(pleasure=0.4, arousal=0.2, dominance=0.3)
        baseline = PADBaseline(pleasure=0.0, arousal=0.0, dominance=0.0)

        new_pad = evolve_pad(current, occ_pull, baseline)
        assert new_pad.pleasure > 0.0  # pulled positive
        assert new_pad.arousal > 0.0
        assert new_pad.dominance > 0.0

    def test_evolve_pad_spring_restores_to_baseline(self):
        from lingya.mind.affect import evolve_pad
        from lingya.mind.config import PADBaseline
        from lingya.mind.state import PADPoint

        current = PADPoint(pleasure=0.8, arousal=0.0, dominance=0.0)
        occ_pull = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        baseline = PADBaseline(pleasure=0.0, arousal=0.0, dominance=0.0)

        new_pad = evolve_pad(current, occ_pull, baseline)
        # Spring force should pull back toward 0
        assert new_pad.pleasure < 0.8

    def test_evolve_pad_clamped_to_bounds(self):
        from lingya.mind.affect import evolve_pad
        from lingya.mind.config import PADBaseline
        from lingya.mind.state import PADPoint

        current = PADPoint(pleasure=0.95, arousal=0.0, dominance=0.0)
        occ_pull = PADPoint(pleasure=0.5, arousal=0.0, dominance=0.0)
        baseline = PADBaseline(pleasure=0.0, arousal=0.0, dominance=0.0)

        new_pad = evolve_pad(current, occ_pull, baseline)
        assert -1.0 <= new_pad.pleasure <= 1.0
        assert -1.0 <= new_pad.arousal <= 1.0
        assert -1.0 <= new_pad.dominance <= 1.0

    def test_ocean_to_pad_baseline_produces_valid_range(self):
        from lingya.mind.affect import ocean_to_pad_baseline
        from lingya.mind.config import BigFiveTraits

        ocean = BigFiveTraits(
            openness=0.75, conscientiousness=0.8, extraversion=0.15,
            agreeableness=0.2, neuroticism=0.35,
        )
        pad = ocean_to_pad_baseline(ocean)
        assert -1.0 <= pad.pleasure <= 1.0
        assert -1.0 <= pad.arousal <= 1.0
        assert -1.0 <= pad.dominance <= 1.0

    def test_ocean_drift_no_effect_with_short_history(self):
        from lingya.mind.affect import ocean_drift
        from lingya.mind.config import BigFiveTraits

        ocean = BigFiveTraits()
        pad_history = []  # Not enough history

        result = ocean_drift(ocean, pad_history)
        assert result.openness == ocean.openness  # Unchanged

    def test_ocean_drift_with_sufficient_history(self):
        from lingya.mind.affect import ocean_drift
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        ocean = BigFiveTraits(openness=0.3)
        # PAD history consistently below OCEAN-derived baseline → slight drift
        pad_history = [PADPoint(pleasure=-0.5, arousal=0.0, dominance=0.0) for _ in range(25)]

        result = ocean_drift(ocean, pad_history)
        # Very small changes due to tiny epsilon and max_step cap
        assert abs(result.openness - ocean.openness) < 0.01


class TestOCCEmotionsDict:
    def test_all_22_emotions_have_pad_vectors(self):
        from lingya.mind.affect import OCC_EMOTIONS

        expected = {
            "joy", "distress", "happy-for", "resentment", "gloating", "sorry-for",
            "hope", "fear", "satisfaction", "relief", "fears-confirmed", "disappointment",
            "pride", "shame", "admiration", "reproach",
            "gratification", "remorse", "anger", "gratitude",
            "love", "hate",
            "interest", "surprise", "disgust", "neutral",
        }
        assert set(OCC_EMOTIONS.keys()) == expected

    def test_pad_vectors_in_valid_range(self):
        from lingya.mind.affect import OCC_EMOTIONS

        for emotion, (p, a, d) in OCC_EMOTIONS.items():
            assert -1.0 <= p <= 1.0, f"{emotion}: pleasure {p} out of range"
            assert -1.0 <= a <= 1.0, f"{emotion}: arousal {a} out of range"
            assert -1.0 <= d <= 1.0, f"{emotion}: dominance {d} out of range"
