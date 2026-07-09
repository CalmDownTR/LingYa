"""IPC (Interpersonal Circumplex) dual-axis state machine.

Agency and communion are now estimated via the merged OCC+IPC LLM call
in affect.py:occ_ipc_process — no separate LLM call needed.
"""

from __future__ import annotations

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
        return IPCState.NEUTRAL
    return IPCState.NEUTRAL
