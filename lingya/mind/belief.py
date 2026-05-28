"""Belief anchoring with OCEAN-modulated update probability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from lingya.mind.config import BigFiveTraits


def belief_update_probability(ocean: BigFiveTraits) -> float:
    """Probability of accepting a belief challenge.

    High agreeableness → higher update probability (more open to change).
    High conscientiousness → lower update probability (more stubborn).
    Returns value in [0, 1].
    """
    p = 0.3 + 0.4 * ocean.agreeableness - 0.2 * ocean.conscientiousness
    return max(0.05, min(0.95, p))


BELIEF_UPDATE_PROMPT = """\
You are an agent's belief system. Your core belief is:

"{core_belief}"

A conversation has challenged this belief in the following way:

"{challenge}"

Your current agreeableness is {agreeableness:.2f} (0=stubborn, 1=malleable).
Your current conscientiousness is {conscientiousness:.2f} (0=flexible, 1=rigid).

Should you update your belief? Reply with a JSON object:
{{"update": true/false, "new_belief": "<revised belief or null>"}}

If you update, the new_belief should integrate the challenge while maintaining coherence.
If you don't update, set new_belief to null."""


async def belief_update_decision(
    core_belief: str,
    challenge: str,
    ocean: BigFiveTraits,
    llm_call: Callable[[str], Awaitable[str]],
) -> tuple[bool, str | None]:
    """Decide whether to update a belief based on a challenge.

    Returns (did_update, new_belief_or_None).
    """
    base_prob = belief_update_probability(ocean)

    prompt = BELIEF_UPDATE_PROMPT.format(
        core_belief=core_belief,
        challenge=challenge,
        agreeableness=ocean.agreeableness,
        conscientiousness=ocean.conscientiousness,
    )

    try:
        import json

        response = await llm_call(prompt)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response[:-3]
        data = json.loads(response)
        should_update = bool(data.get("update", False))
        # Only update if both LLM agrees AND probability check passes
        if should_update and base_prob > 0.5:
            return True, data.get("new_belief")
        return False, None
    except Exception:
        return False, None
