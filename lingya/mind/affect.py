"""OCC 22-emotion decision tree + PAD/OCEAN evolution.

Deterministic, rule-based emotion classification. No LLM for emotion labels.
LLM is only used for cognitive appraisal (w_goal, p_expected estimation)
and merged OCC+IPC (w_goal, p_expected, agency, communion) in a single call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from lingya.mind.config import BigFiveTraits, PADBaseline
from lingya.mind.state import PADPoint

# ── OCC 22-emotion → PAD pull vectors (Mehrabian literature) ──────────────
# Each tuple is (pleasure, arousal, dominance) pull direction.

OCC_EMOTIONS: dict[str, tuple[float, float, float]] = {
    # Event-outcome (self)
    "joy": (0.40, 0.20, 0.30),
    "distress": (-0.40, 0.30, -0.30),
    # Event-outcome (other)
    "happy-for": (0.30, 0.10, 0.10),
    "resentment": (-0.20, 0.20, -0.10),
    "gloating": (0.20, 0.10, 0.20),
    "sorry-for": (-0.20, -0.10, -0.10),
    # Prospect-based
    "hope": (0.20, 0.10, 0.00),
    "fear": (-0.30, 0.40, -0.40),
    "satisfaction": (0.30, -0.10, 0.20),
    "relief": (0.40, -0.20, 0.10),
    "fears-confirmed": (-0.30, 0.10, -0.20),
    "disappointment": (-0.20, -0.10, -0.10),
    # Agent attribution (self)
    "pride": (0.30, 0.20, 0.40),
    "shame": (-0.30, 0.10, -0.30),
    # Agent attribution (other)
    "admiration": (0.20, 0.10, -0.10),
    "reproach": (-0.20, 0.10, 0.10),
    # Compound
    "gratification": (0.30, 0.00, 0.20),
    "remorse": (-0.30, 0.00, -0.20),
    "anger": (-0.30, 0.40, 0.30),
    "gratitude": (0.30, 0.00, -0.10),
    # Object attraction
    "love": (0.40, 0.10, 0.00),
    "hate": (-0.40, 0.30, 0.20),
    # Fallback
    "interest": (0.10, 0.20, 0.10),
    "surprise": (0.00, 0.40, -0.10),
    "disgust": (-0.20, 0.10, 0.10),
    "neutral": (0.00, 0.00, 0.00),
}


@dataclass
class OCCResult:
    emotion: str
    intensity: float
    pad_pull: PADPoint
    w_goal: float
    p_expected: float


# ── OCC Decision Tree ────────────────────────────────────────────────────

def occ_classify(event: dict[str, Any]) -> str:
    """Deterministic OCC 22-emotion classification from event structure.

    Expected event keys:
        event_type: "outcome" | "action" | "object"
        valence: "positive" | "negative"
        focus: "self" | "other" | None
        prospect: "prospective" | "actual" | None
        agent: "self" | "other" | None
        confirmed: bool | None   (for prospect events)
        disconfirmed: bool | None (for prospect events)
    """
    event_type = event.get("event_type", "outcome")
    valence = event.get("valence", "neutral")
    focus = event.get("focus")
    prospect = event.get("prospect")
    agent = event.get("agent")

    # ── Object attraction ──
    if event_type == "object":
        return "love" if valence == "positive" else "hate"

    # ── Agent-based ──
    if event_type == "action":
        if agent == "self":
            return "pride" if valence == "positive" else "shame"
        if agent == "other":
            return "admiration" if valence == "positive" else "reproach"
        # Compound: other's action affects self
        if focus == "self":
            if valence == "positive":
                return "gratitude"
            return "anger"

    # ── Event outcome ──
    if event_type == "outcome":
        # Prospect-based
        if prospect == "prospective":
            return "hope" if valence == "positive" else "fear"
        if prospect == "actual":
            if valence == "positive":
                return "satisfaction" if event.get("confirmed") else "relief"
            return "fears-confirmed" if event.get("confirmed") else "disappointment"

        # Outcome for self
        if focus == "self":
            return "joy" if valence == "positive" else "distress"

        # Outcome for other
        if focus == "other":
            if valence == "positive":
                # Deservingness check
                if event.get("deserving"):
                    return "happy-for"
                return "gloating"  # positive outcome for undeserving other → gloating
            if event.get("deserving"):
                return "sorry-for"
            return "resentment"  # negative outcome for deserving other → resentment

    # Fallback
    if valence == "positive":
        return "interest"
    if valence == "negative":
        return "disgust"
    return "neutral"


def compute_intensity(
    w_goal: float,
    p_expected: float,
    e_residual: float = 1.0,
) -> float:
    """Intensity = w_goal * (1 - p_expected) * e_residual.

    High goal-relevance + unexpected → high intensity.
    Clamped to [0, 1].
    """
    raw = w_goal * (1.0 - p_expected) * e_residual
    return max(0.0, min(1.0, raw))


# ── Cognitive Appraisal (LLM) ────────────────────────────────────────────

COGNITIVE_APPRAISAL_PROMPT = """\
Analyze this event from a conversation and return two numbers:

Event: {event_description}

1. w_goal (0-1): How relevant is this event to the agent's goals and concerns?
   0 = completely irrelevant, 1 = critically important
2. p_expected (0-1): How expected was this event from the agent's perspective?
   0 = completely surprising, 1 = fully expected

Return ONLY a JSON object: {{"w_goal": <float>, "p_expected": <float>}}"""


async def cognitive_appraisal(
    event: dict[str, Any],
    llm_call: Callable[[str], Awaitable[str]],
) -> tuple[float, float]:
    """Use LLM to estimate goal-relevance and expectedness of an event."""
    import json

    description = event.get("description", str(event))
    prompt = COGNITIVE_APPRAISAL_PROMPT.format(event_description=description)

    try:
        response = await llm_call(prompt)
        # Extract JSON from response (may have markdown wrapping)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response[:-3]
        data = json.loads(response)
        w_goal = max(0.0, min(1.0, float(data.get("w_goal", 0.5))))
        p_expected = max(0.0, min(1.0, float(data.get("p_expected", 0.5))))
        return w_goal, p_expected
    except Exception:
        # Fallback: moderate relevance, moderate surprise
        return 0.3, 0.5


# ── Full OCC Process ─────────────────────────────────────────────────────

async def occ_process(
    event: dict[str, Any],
    llm_call: Callable[[str], Awaitable[str]],
) -> OCCResult:
    """Full OCC pipeline: appraisal → classify → intensity → PAD pull.

    Args:
        event: Dict with event_type, valence, focus, prospect, agent, etc.
        llm_call: Async callable for LLM cognitive appraisal.

    Returns:
        OCCResult with emotion label, intensity, and scaled PAD pull vector.
    """
    w_goal, p_expected = await cognitive_appraisal(event, llm_call)
    emotion = occ_classify(event)
    intensity = compute_intensity(w_goal, p_expected)
    pull = OCC_EMOTIONS.get(emotion, OCC_EMOTIONS["neutral"])
    pad_pull = PADPoint(
        pleasure=pull[0] * intensity,
        arousal=pull[1] * intensity,
        dominance=pull[2] * intensity,
    )
    return OCCResult(
        emotion=emotion,
        intensity=intensity,
        pad_pull=pad_pull,
        w_goal=w_goal,
        p_expected=p_expected,
    )


# ── Merged OCC + IPC (single LLM call) ──────────────────────────────────


@dataclass
class OCCIPCResult:
    emotion: str
    intensity: float
    pad_pull: PADPoint
    w_goal: float
    p_expected: float
    agency: float
    communion: float


OCC_IPC_MERGED_PROMPT = """\
Analyze this event from a conversation and return a JSON object with these fields:

Event: {event_description}

Recent emotional context:
{emotion_context}

Classification fields (for emotion labeling):
- event_type: "outcome" (something happened to someone), "action" (someone did something), or "object" (attraction/aversion to something)
- valence: "positive" or "negative" — is this event good or bad from the agent's perspective?
- focus: "self" (about the agent), "other" (about the user/someone else), or null
- prospect: "prospective" (about the future), "actual" (about past/present), or null
- agent: "self" (agent caused it), "other" (user/someone else caused it), or null

Appraisal fields (for intensity + IPC):
- w_goal (0-1): How relevant is this event to the agent's goals and concerns?
  0 = completely irrelevant, 1 = critically important
- p_expected (0-1): How expected was this event from the agent's perspective?
  0 = completely surprising, 1 = fully expected
- agency (0-1): How much should the agent lead, direct, or assert in response?
  0 = completely passive/following, 1 = highly directive/assertive
- communion (0-1): How much should the agent connect, empathize, or build rapport?
  0 = cold/distant/formal, 1 = warm/intimate/connected

Return ONLY a JSON object: {{"event_type": "<string>", "valence": "<string>", "focus": "<string|null>", "prospect": "<string|null>", "agent": "<string|null>", "w_goal": <float>, "p_expected": <float>, "agency": <float>, "communion": <float>}}"""


_OCC_IPC_TIMEOUT = 1.5  # seconds


async def occ_ipc_process(
    event: dict[str, Any],
    recent_emotions: list[dict],
    llm_call: Callable[[str], Awaitable[str]],
) -> OCCIPCResult:
    """Single LLM call for cognitive appraisal + IPC estimation + OCC classification.

    Merges what were previously three serial LLM calls into one.
    Returns OCC results (emotion, intensity, pad_pull) plus IPC axes (agency, communion).
    The LLM now also returns event_type/valence/focus/prospect/agent so the
    deterministic OCC decision tree receives correct classification input.
    Falls back to neutral values on timeout or parse error.
    """
    import json

    description = event.get("description", str(event))
    emotion_context = (
        "\n".join(
            f"Turn {e.get('turn', '?')}: {e.get('emotion', 'neutral')} (intensity={e.get('intensity', 0.5):.2f})"
            for e in recent_emotions[-6:]
        )
        if recent_emotions
        else "No recent emotional context."
    )

    prompt = OCC_IPC_MERGED_PROMPT.format(
        event_description=description,
        emotion_context=emotion_context,
    )

    try:
        response = await asyncio.wait_for(llm_call(prompt), timeout=_OCC_IPC_TIMEOUT)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response[:-3]
        data = json.loads(response)
        w_goal = max(0.0, min(1.0, float(data.get("w_goal", 0.5))))
        p_expected = max(0.0, min(1.0, float(data.get("p_expected", 0.5))))
        agency = max(0.0, min(1.0, float(data.get("agency", 0.5))))
        communion = max(0.0, min(1.0, float(data.get("communion", 0.5))))
        # Parse OCC classification fields from LLM (v0.9.7 fix: no longer hardcoded)
        occ_event_type = data.get("event_type", event.get("event_type", "outcome"))
        occ_valence = data.get("valence", event.get("valence", "neutral"))
        occ_focus = data.get("focus") or event.get("focus")
        occ_prospect = data.get("prospect") or event.get("prospect")
        occ_agent = data.get("agent") or event.get("agent")
    except Exception:
        w_goal, p_expected = 0.3, 0.5
        agency, communion = 0.5, 0.5
        occ_event_type = event.get("event_type", "outcome")
        occ_valence = event.get("valence", "neutral")
        occ_focus = event.get("focus")
        occ_prospect = event.get("prospect")
        occ_agent = event.get("agent")

    # Build classified event dict from LLM-returned fields (v0.9.7)
    classified_event: dict[str, Any] = {
        "event_type": occ_event_type,
        "valence": occ_valence,
        "focus": occ_focus,
        "prospect": occ_prospect,
        "agent": occ_agent,
    }
    emotion = occ_classify(classified_event)
    intensity = compute_intensity(w_goal, p_expected)
    pull = OCC_EMOTIONS.get(emotion, OCC_EMOTIONS["neutral"])
    pad_pull = PADPoint(
        pleasure=pull[0] * intensity,
        arousal=pull[1] * intensity,
        dominance=pull[2] * intensity,
    )
    return OCCIPCResult(
        emotion=emotion,
        intensity=intensity,
        pad_pull=pad_pull,
        w_goal=w_goal,
        p_expected=p_expected,
        agency=agency,
        communion=communion,
    )


# ── PAD Evolution (Phase 3) ──────────────────────────────────────────────

def ocean_to_pad_baseline(ocean: BigFiveTraits) -> PADBaseline:
    """Convert OCEAN traits to PAD baseline using Mehrabian formulas.

    Literature-derived coefficients mapping Big Five to PAD space.
    """
    P = (
        0.21 * ocean.extraversion
        + 0.27 * ocean.agreeableness
        - 0.20 * ocean.neuroticism
        + 0.14 * ocean.openness
        + 0.08 * ocean.conscientiousness
    )
    # Normalize from [0,1] scale to [-1,1]
    P = (P - 0.25) * 2.0

    A = (
        0.29 * ocean.neuroticism
        + 0.16 * ocean.extraversion
        + 0.10 * ocean.openness
        + 0.05 * ocean.conscientiousness
        - 0.05 * ocean.agreeableness
    )
    A = (A - 0.25) * 2.0

    D = (
        0.27 * ocean.extraversion
        + 0.18 * ocean.conscientiousness
        + 0.17 * ocean.openness
        - 0.15 * ocean.neuroticism
        - 0.03 * ocean.agreeableness
    )
    D = (D - 0.22) * 2.0

    return PADBaseline(
        pleasure=max(-1.0, min(1.0, P)),
        arousal=max(-1.0, min(1.0, A)),
        dominance=max(-1.0, min(1.0, D)),
    )


def evolve_pad(
    current: PADPoint,
    occ_pull: PADPoint,
    baseline: PADBaseline,
    spring_k: float = 0.2,
) -> PADPoint:
    """Apply OCC emotion pull + spring-restore toward baseline.

    new = current + pull_weight * occ_pull + spring_force
    spring_force = -spring_k * (current - baseline)
    Clamped to [-1, 1].
    """
    pull_weight = 0.3
    spring_force_pleasure = -spring_k * (current.pleasure - baseline.pleasure)
    spring_force_arousal = -spring_k * (current.arousal - baseline.arousal)
    spring_force_dominance = -spring_k * (current.dominance - baseline.dominance)

    new_pleasure = (
        current.pleasure
        + pull_weight * occ_pull.pleasure
        + spring_force_pleasure
    )
    new_arousal = (
        current.arousal
        + pull_weight * occ_pull.arousal
        + spring_force_arousal
    )
    new_dominance = (
        current.dominance
        + pull_weight * occ_pull.dominance
        + spring_force_dominance
    )

    return PADPoint(
        pleasure=max(-1.0, min(1.0, new_pleasure)),
        arousal=max(-1.0, min(1.0, new_arousal)),
        dominance=max(-1.0, min(1.0, new_dominance)),
    )


def ocean_drift(
    ocean: BigFiveTraits,
    pad_history: list[PADPoint],
    epsilon: float = 0.001,
    min_history: int = 20,
    max_step: float = 0.005,
) -> BigFiveTraits:
    """Long-term PAD deviation → tiny OCEAN adjustment.

    PAD baseline is derived from OCEAN itself (closed loop). Drift only
    triggers with sufficient history. Epsilon and max_step cap ensure
    personality changes are extremely gradual.

    max_step limits the per-cycle absolute change in each OCEAN dimension.
    """
    if len(pad_history) < min_history:
        return ocean

    baseline = ocean_to_pad_baseline(ocean)

    # Average PAD over history
    n = len(pad_history)
    avg_p = sum(p.pleasure for p in pad_history) / n
    avg_a = sum(p.arousal for p in pad_history) / n
    avg_d = sum(p.dominance for p in pad_history) / n

    # Deviation from baseline
    dp = avg_p - baseline.pleasure
    da = avg_a - baseline.arousal
    dd = avg_d - baseline.dominance

    # Tiny adjustments with step cap
    def step(current: float, delta: float) -> float:
        clamped_delta = max(-max_step, min(max_step, delta))
        return max(0.0, min(1.0, current + clamped_delta))

    return BigFiveTraits(
        openness=step(ocean.openness, epsilon * dp * 0.15),
        conscientiousness=step(ocean.conscientiousness, epsilon * dd * 0.18),
        extraversion=step(ocean.extraversion, epsilon * (dp * 0.21 + dd * 0.27)),
        agreeableness=step(ocean.agreeableness, epsilon * dp * 0.27),
        neuroticism=step(ocean.neuroticism, epsilon * da * 0.29),
    )
