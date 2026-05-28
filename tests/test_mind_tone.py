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


class TestComputeDynamicTone:
    def test_blends_base_with_mapped_and_stage_delta(self):
        from lingya.mind.tone import compute_dynamic_tone, ConversationStage
        from lingya.mind.config import ToneMatrix
        from lingya.mind.state import PADPoint

        base = ToneMatrix(warmth=50, formality=50, humor=0.1)
        pad = PADPoint(pleasure=0.3, arousal=0.0, dominance=0.3)
        stage = ConversationStage.DEEP

        result = compute_dynamic_tone(pad, stage, base)
        assert 0 <= result.warmth <= 100
        assert 0 <= result.formality <= 100
        assert 0.0 <= result.humor <= 1.0
        # DEEP stage should increase warmth slightly
        assert result.warmth > 50

    def test_crisis_stage_warms_and_relaxes(self):
        from lingya.mind.tone import compute_dynamic_tone, ConversationStage
        from lingya.mind.config import ToneMatrix
        from lingya.mind.state import PADPoint

        base = ToneMatrix(warmth=50, formality=80, humor=0.1)
        pad = PADPoint(pleasure=-0.5, arousal=0.6, dominance=0.0)
        stage = ConversationStage.CRISIS

        result = compute_dynamic_tone(pad, stage, base)
        # CRISIS should increase warmth, decrease formality
        assert result.warmth > 50
        assert result.formality < 80
