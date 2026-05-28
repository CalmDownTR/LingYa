from __future__ import annotations

import pytest


class TestDetectStage:
    def test_initial_stage_for_early_turns(self):
        from lingya.mind.tone import ConversationStage, detect_stage
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        assert detect_stage(1, pad, []) == ConversationStage.INITIAL
        assert detect_stage(2, pad, []) == ConversationStage.INITIAL
        assert detect_stage(3, pad, []) == ConversationStage.INITIAL

    def test_deep_stage_for_later_turns(self):
        from lingya.mind.tone import ConversationStage, detect_stage
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        assert detect_stage(4, pad, []) == ConversationStage.DEEP
        assert detect_stage(10, pad, []) == ConversationStage.DEEP

    def test_crisis_stage_for_negative_high_arousal(self):
        from lingya.mind.tone import ConversationStage, detect_stage
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=-0.7, arousal=0.8, dominance=-0.3)
        assert detect_stage(5, pad, []) == ConversationStage.CRISIS

    def test_error_stage_for_recent_reproach(self):
        from lingya.mind.tone import ConversationStage, detect_stage
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        emotions = [{"emotion": "joy"}, {"emotion": "reproach"}]
        assert detect_stage(5, pad, emotions) == ConversationStage.ERROR

    def test_error_stage_for_recent_anger(self):
        from lingya.mind.tone import ConversationStage, detect_stage
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.0)
        emotions = [{"emotion": "interest"}, {"emotion": "anger"}]
        assert detect_stage(5, pad, emotions) == ConversationStage.ERROR


class TestPadToTone:
    def test_high_dominance_produces_high_formality(self):
        from lingya.mind.tone import pad_to_tone
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=0.8)
        tone = pad_to_tone(pad)
        assert tone["formality"] > 80

    def test_high_pleasure_produces_high_warmth(self):
        from lingya.mind.tone import pad_to_tone
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.8, arousal=0.0, dominance=0.0)
        tone = pad_to_tone(pad)
        assert tone["warmth"] > 80

    def test_moderate_arousal_enables_humor(self):
        from lingya.mind.tone import pad_to_tone
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.3, dominance=0.0)
        tone = pad_to_tone(pad)
        assert tone["humor"] > 0.3

    def test_extreme_arousal_kills_humor(self):
        from lingya.mind.tone import pad_to_tone
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.9, dominance=0.0)
        tone = pad_to_tone(pad)
        assert tone["humor"] < 0.3

    def test_values_in_valid_ranges(self):
        from lingya.mind.tone import pad_to_tone
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.5, arousal=-0.3, dominance=0.2)
        tone = pad_to_tone(pad)
        assert 0 <= tone["warmth"] <= 100
        assert 0 <= tone["formality"] <= 100
        assert 0 <= tone["humor"] <= 1


class TestApplyOceanModulation:
    """Tests for the OCEAN personality modulation layer (Phase 6).

    apply_ocean_modulation modulates the *deviation* of raw tone values
    from the neutral midpoint (50 for warmth/formality, raw value for humor).
    It uses deviation-modulation, not direct multiplication, so the
    mathematical invariant is:

        output = midpoint + (raw - midpoint) × gain

    where gain is a function of OCEAN traits and (for formality) PAD dominance.
    This preserves the midpoint as a fixed point regardless of gain values.
    """

    def test_high_agreeableness_suppresses_warmth_deviation(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        # Low pleasure should reduce warmth, but high A suppresses this
        pad = PADPoint(pleasure=-0.5, arousal=0.0, dominance=0.0)
        raw = pad_to_tone(pad)

        high_a = BigFiveTraits(agreeableness=0.85)
        low_a = BigFiveTraits(agreeableness=0.15)

        mod_high = apply_ocean_modulation(raw, high_a)
        mod_low = apply_ocean_modulation(raw, low_a)

        # Both should be below 50 (negative pleasure), but high A stays warmer
        assert mod_high["warmth"] > mod_low["warmth"]

    def test_high_agreeableness_inverts_formality_for_low_dominance(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        # Low dominance normally → low formality (blunt). High A inverts this.
        pad = PADPoint(pleasure=0.0, arousal=0.0, dominance=-0.5)
        raw = pad_to_tone(pad)
        # raw formality should be < 50 (low dominance → casual/blunt)
        assert raw["formality"] < 50

        high_a = BigFiveTraits(agreeableness=0.85)
        low_a = BigFiveTraits(agreeableness=0.15)

        mod_high = apply_ocean_modulation(raw, high_a)
        mod_low = apply_ocean_modulation(raw, low_a)

        # High A: low dominance → nervous polite → formality ABOVE 50
        assert mod_high["formality"] > 50
        # Low A: low dominance → blunt/casual → formality stays below 50
        assert mod_low["formality"] < 50

    def test_neuroticism_amplifies_warmth_and_formality_deviations(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        # Positive PAD → raw tone above 50. Use low A so f_mod is non-zero.
        pad = PADPoint(pleasure=0.6, arousal=0.0, dominance=0.6)
        raw = pad_to_tone(pad)

        high_n = BigFiveTraits(agreeableness=0.2, neuroticism=0.9)
        low_n = BigFiveTraits(agreeableness=0.2, neuroticism=0.1)

        mod_high = apply_ocean_modulation(raw, high_n)
        mod_low = apply_ocean_modulation(raw, low_n)

        # High N amplifies positive deviations further above 50
        assert mod_high["warmth"] > mod_low["warmth"]
        assert mod_high["formality"] > mod_low["formality"]

    def test_neuroticism_does_not_affect_humor(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.3, arousal=0.3, dominance=0.0)
        raw = pad_to_tone(pad)

        high_n = BigFiveTraits(neuroticism=0.9)
        low_n = BigFiveTraits(neuroticism=0.1)

        mod_high = apply_ocean_modulation(raw, high_n)
        mod_low = apply_ocean_modulation(raw, low_n)

        # Humor should be identical regardless of N
        assert mod_high["humor"] == mod_low["humor"]

    def test_extraversion_amplifies_humor(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        pad = PADPoint(pleasure=0.0, arousal=0.3, dominance=0.0)
        raw = pad_to_tone(pad)

        high_e = BigFiveTraits(extraversion=0.9)
        low_e = BigFiveTraits(extraversion=0.1)

        mod_high = apply_ocean_modulation(raw, high_e)
        mod_low = apply_ocean_modulation(raw, low_e)

        assert mod_high["humor"] > mod_low["humor"]

    def test_all_outputs_clamped_to_valid_ranges(self):
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        # Test all 8 corners of PAD space with extreme OCEAN
        corners = [
            PADPoint(pleasure=-1.0, arousal=-1.0, dominance=-1.0),
            PADPoint(pleasure=-1.0, arousal=-1.0, dominance=1.0),
            PADPoint(pleasure=-1.0, arousal=1.0, dominance=-1.0),
            PADPoint(pleasure=-1.0, arousal=1.0, dominance=1.0),
            PADPoint(pleasure=1.0, arousal=-1.0, dominance=-1.0),
            PADPoint(pleasure=1.0, arousal=-1.0, dominance=1.0),
            PADPoint(pleasure=1.0, arousal=1.0, dominance=-1.0),
            PADPoint(pleasure=1.0, arousal=1.0, dominance=1.0),
        ]
        # All traits at extremes
        ocean = BigFiveTraits(
            openness=1.0, conscientiousness=1.0,
            extraversion=1.0, agreeableness=1.0, neuroticism=1.0,
        )

        for pad in corners:
            raw = pad_to_tone(pad)
            mod = apply_ocean_modulation(raw, ocean)
            assert 0.0 <= mod["warmth"] <= 100.0, f"warmth {mod['warmth']} out of range at PAD={pad}"
            assert 0.0 <= mod["formality"] <= 100.0, f"formality {mod['formality']} out of range at PAD={pad}"
            assert 0.0 <= mod["humor"] <= 1.0, f"humor {mod['humor']} out of range at PAD={pad}"

    def test_deviation_modulation_invariant(self):
        """Midpoint 50 is a fixed point: when raw = 50, output = 50 regardless of OCEAN."""
        from lingya.mind.tone import apply_ocean_modulation
        from lingya.mind.config import BigFiveTraits

        raw = {"warmth": 50.0, "formality": 50.0, "humor": 0.5}
        ocean_extreme = BigFiveTraits(
            openness=1.0, conscientiousness=1.0,
            extraversion=1.0, agreeableness=1.0, neuroticism=1.0,
        )
        ocean_minimal = BigFiveTraits(
            openness=0.0, conscientiousness=0.0,
            extraversion=0.0, agreeableness=0.0, neuroticism=0.0,
        )

        mod_extreme = apply_ocean_modulation(raw, ocean_extreme)
        mod_minimal = apply_ocean_modulation(raw, ocean_minimal)

        # Warmth and formality at midpoint should be unchanged
        assert mod_extreme["warmth"] == 50.0
        assert mod_extreme["formality"] == 50.0
        assert mod_minimal["warmth"] == 50.0
        assert mod_minimal["formality"] == 50.0

    def test_high_a_low_dominance_produces_nervous_politeness(self):
        """User's canonical example: high A + low D → heightened formality."""
        from lingya.mind.tone import apply_ocean_modulation, pad_to_tone
        from lingya.mind.config import BigFiveTraits
        from lingya.mind.state import PADPoint

        # PAD = (-0.5, 0.6, -0.3): low pleasure, high arousal, low dominance
        pad = PADPoint(pleasure=-0.5, arousal=0.6, dominance=-0.3)
        raw = pad_to_tone(pad)

        high_a = BigFiveTraits(agreeableness=0.85, neuroticism=0.5, extraversion=0.7)
        low_a = BigFiveTraits(agreeableness=0.15, neuroticism=0.5, extraversion=0.7)

        mod_high = apply_ocean_modulation(raw, high_a)
        mod_low = apply_ocean_modulation(raw, low_a)

        # High A: suppressed warmth loss, elevated formality (nervous polite)
        assert mod_high["warmth"] > mod_low["warmth"]
        assert mod_high["formality"] > mod_low["formality"]

        # Low A: warmth tracks pleasure closely (less dampened than high A)
        # formality stays below 50 (blunt/casual, no inversion)
        assert mod_low["formality"] < 50  # stays casual/blunt


class TestComputeDynamicTone:
    def test_blends_base_with_mapped_and_stage_delta(self):
        from lingya.mind.tone import compute_dynamic_tone, ConversationStage
        from lingya.mind.config import BigFiveTraits, ToneMatrix
        from lingya.mind.state import PADPoint

        base = ToneMatrix(warmth=50, formality=50, humor=0.1)
        ocean = BigFiveTraits()
        pad = PADPoint(pleasure=0.3, arousal=0.0, dominance=0.3)
        stage = ConversationStage.DEEP

        result = compute_dynamic_tone(pad, stage, base, ocean)
        assert 0 <= result.warmth <= 100
        assert 0 <= result.formality <= 100
        assert 0.0 <= result.humor <= 1.0
        # DEEP stage should increase warmth slightly
        assert result.warmth > 50

    def test_crisis_stage_warms_and_relaxes(self):
        from lingya.mind.tone import compute_dynamic_tone, ConversationStage
        from lingya.mind.config import BigFiveTraits, ToneMatrix
        from lingya.mind.state import PADPoint

        base = ToneMatrix(warmth=50, formality=80, humor=0.1)
        ocean = BigFiveTraits()
        pad = PADPoint(pleasure=-0.5, arousal=0.6, dominance=0.0)
        stage = ConversationStage.CRISIS

        result = compute_dynamic_tone(pad, stage, base, ocean)
        # CRISIS should increase warmth, decrease formality
        assert result.warmth > 50
        assert result.formality < 80

    def test_ocean_modulation_flows_through_to_final_tone(self):
        from lingya.mind.tone import compute_dynamic_tone, ConversationStage
        from lingya.mind.config import BigFiveTraits, ToneMatrix
        from lingya.mind.state import PADPoint

        # Same PAD, same base, same stage — only OCEAN differs
        base = ToneMatrix(warmth=50, formality=50, humor=0.1)
        pad = PADPoint(pleasure=-0.5, arousal=0.6, dominance=-0.3)
        stage = ConversationStage.NEUTRAL

        high_a = BigFiveTraits(agreeableness=0.85, neuroticism=0.5, extraversion=0.7)
        low_a = BigFiveTraits(agreeableness=0.15, neuroticism=0.5, extraversion=0.7)

        result_high_a = compute_dynamic_tone(pad, stage, base, high_a)
        result_low_a = compute_dynamic_tone(pad, stage, base, low_a)

        # Different personalities should produce meaningfully different tones
        assert result_high_a.warmth != result_low_a.warmth
        assert result_high_a.formality != result_low_a.formality
        # High A should be warmer and more formal in distress
        assert result_high_a.warmth > result_low_a.warmth
        assert result_high_a.formality > result_low_a.formality
