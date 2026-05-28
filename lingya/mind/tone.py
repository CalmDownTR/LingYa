"""Stage-aware tone matrix — continuous PAD→tone mapping.

Replaces bucketing.py's if-elif with smooth, continuous functions.
"""

from __future__ import annotations

from enum import Enum

from lingya.mind.config import ToneMatrix
from lingya.mind.state import PADPoint


class ConversationStage(Enum):
    INITIAL = "initial"   # turns 1-3
    DEEP = "deep"         # turns 4+
    CRISIS = "crisis"     # user extremely negative pleasure + high arousal
    ERROR = "error"       # recent reproach/anger toward agent
    NEUTRAL = "neutral"   # fallback


def detect_stage(
    turn_count: int,
    pad: PADPoint,
    recent_emotions: list[dict],
) -> ConversationStage:
    """Detect conversation stage from turn count, PAD position, and recent emotions."""
    # CRISIS: extreme negative pleasure + high arousal
    if pad.pleasure < -0.5 and pad.arousal > 0.5:
        return ConversationStage.CRISIS

    # ERROR: recent reproach or anger toward agent
    for e in recent_emotions[-3:]:
        if e.get("emotion") in ("reproach", "anger", "disgust"):
            return ConversationStage.ERROR

    # INITIAL: early turns
    if turn_count <= 3:
        return ConversationStage.INITIAL

    # DEEP: turns 4+
    if turn_count >= 4:
        return ConversationStage.DEEP

    return ConversationStage.NEUTRAL


def pad_to_tone(pad: PADPoint) -> dict[str, float]:
    """Continuous PAD→tone mapping.

    dominance → formality (positive correlation)
    pleasure → warmth (positive correlation)
    arousal → humor (inverted-U: moderate enables, extreme kills)

    Returns dict with warmth, formality, humor each in valid ranges.
    """
    # Dominance → formality: map from [-1,1] to [0,100]
    formality = 50.0 + pad.dominance * 50.0

    # Pleasure → warmth: map from [-1,1] to [0,100]
    warmth = 50.0 + pad.pleasure * 50.0

    # Arousal → humor: inverted-U, peak at arousal=0.3, drops at extremes
    humor = max(0.0, min(1.0, 0.3 + (0.5 - abs(pad.arousal - 0.3)) * 0.7))

    return {
        "warmth": max(0.0, min(100.0, warmth)),
        "formality": max(0.0, min(100.0, formality)),
        "humor": humor,
    }


def stage_tone_delta(stage: ConversationStage) -> dict[str, float]:
    """Per-stage tone adjustments (added to base tone)."""
    deltas: dict[ConversationStage, dict[str, float]] = {
        ConversationStage.INITIAL: {"warmth": -5.0, "formality": 5.0, "humor": -0.05},
        ConversationStage.DEEP: {"warmth": 5.0, "formality": -5.0, "humor": 0.05},
        ConversationStage.CRISIS: {"warmth": 10.0, "formality": -10.0, "humor": -0.1},
        ConversationStage.ERROR: {"warmth": -5.0, "formality": 10.0, "humor": -0.05},
        ConversationStage.NEUTRAL: {"warmth": 0.0, "formality": 0.0, "humor": 0.0},
    }
    return deltas.get(stage, {"warmth": 0.0, "formality": 0.0, "humor": 0.0})


def compute_dynamic_tone(
    pad: PADPoint,
    stage: ConversationStage,
    base: ToneMatrix,
) -> ToneMatrix:
    """Compute final tone by blending PAD-mapped tone with base + stage delta."""
    mapped = pad_to_tone(pad)
    delta = stage_tone_delta(stage)

    warmth = int(max(0.0, min(100.0, mapped["warmth"] * 0.3 + base.warmth * 0.7 + delta["warmth"])))
    formality = int(max(0.0, min(100.0, mapped["formality"] * 0.3 + base.formality * 0.7 + delta["formality"])))
    humor = max(0.0, min(1.0, mapped["humor"] * 0.3 + base.humor * 0.7 + delta["humor"]))

    return ToneMatrix(warmth=warmth, formality=formality, humor=humor)
