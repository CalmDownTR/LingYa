"""IPC (Interpersonal Circumplex) dual-axis state machine.

Uses LLM few-shot to estimate agency and communion from recent turns.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from enum import Enum


class IPCState(Enum):
    PROFESSIONAL_DEFENSE = "professional_defense"  # high agency, low communion
    WARM_LISTENING = "warm_listening"               # low agency, high communion
    CRISIS_INTERVENTION = "crisis_intervention"     # high agency, high communion
    PLAYFUL_COLLABORATION = "playful_collaboration" # moderate both
    NEUTRAL = "neutral"


# Valid state transitions
IPC_TRANSITIONS: dict[IPCState, set[IPCState]] = {
    IPCState.PROFESSIONAL_DEFENSE: {IPCState.WARM_LISTENING, IPCState.NEUTRAL, IPCState.CRISIS_INTERVENTION},
    IPCState.WARM_LISTENING: {IPCState.PROFESSIONAL_DEFENSE, IPCState.NEUTRAL, IPCState.PLAYFUL_COLLABORATION},
    IPCState.CRISIS_INTERVENTION: {IPCState.WARM_LISTENING, IPCState.NEUTRAL, IPCState.PROFESSIONAL_DEFENSE},
    IPCState.PLAYFUL_COLLABORATION: {IPCState.WARM_LISTENING, IPCState.NEUTRAL},
    IPCState.NEUTRAL: {IPCState.PROFESSIONAL_DEFENSE, IPCState.WARM_LISTENING, IPCState.PLAYFUL_COLLABORATION},
}


IPC_ESTIMATION_PROMPT = """\
Analyze the following conversation turns and estimate the agent's interpersonal stance on two axes:

- Agency (0-1): How much the agent leads, directs, asserts, or takes control.
  0 = completely passive/following, 1 = highly directive/assertive
- Communion (0-1): How much the agent connects, empathizes, or builds rapport.
  0 = cold/distant/formal, 1 = warm/intimate/connected

Recent conversation:
{transcript}

Return ONLY a JSON object: {{"agency": <float>, "communion": <float>}}"""


async def estimate_ipc(
    last_turns: list[dict],
    llm_call: Callable[[str], Awaitable[str]],
) -> tuple[float, float]:
    """Estimate agency and communion from recent conversation turns via LLM few-shot."""
    if not last_turns:
        return 0.5, 0.5

    transcript = "\n".join(
        f"{t.get('role', 'unknown')}: {t.get('content', '')[:300]}"
        for t in last_turns[-6:]
    )

    try:
        response = await llm_call(IPC_ESTIMATION_PROMPT.format(transcript=transcript))
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response[:-3]
        data = json.loads(response)
        agency = max(0.0, min(1.0, float(data.get("agency", 0.5))))
        communion = max(0.0, min(1.0, float(data.get("communion", 0.5))))
        return agency, communion
    except Exception:
        return 0.5, 0.5


def ipc_to_state(agency: float, communion: float) -> IPCState:
    """Map (agency, communion) coordinates to IPC state."""
    if agency > 0.6 and communion > 0.6:
        return IPCState.CRISIS_INTERVENTION
    if agency > 0.6 and communion < 0.4:
        return IPCState.PROFESSIONAL_DEFENSE
    if agency < 0.4 and communion > 0.6:
        return IPCState.WARM_LISTENING
    if 0.4 <= agency <= 0.7 and 0.4 <= communion <= 0.7:
        return IPCState.PLAYFUL_COLLABORATION
    return IPCState.NEUTRAL


def next_ipc_state(current: IPCState, target: IPCState) -> IPCState:
    """Enforce valid transitions. If target not reachable from current, go through NEUTRAL."""
    if target == current:
        return current
    valid = IPC_TRANSITIONS.get(current, set())
    if target in valid:
        return target
    # Invalid direct transition — route through NEUTRAL
    if IPCState.NEUTRAL in valid:
        return target  # Next turn it'll transition
    return current
