"""Safety guard — implicit re-anchoring via cosine similarity monitoring."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable


async def check_reanchor(
    response_text: str,
    identity_kernel: str,
    embedding_fn: Callable[[str], list[float]],
    threshold: float = 0.3,
) -> bool:
    """Check if the agent's response has drifted too far from its identity.

    Compares cosine similarity between the response and the identity kernel.
    Returns True if re-anchoring is needed (similarity below threshold).
    """
    try:
        resp_embedding = embedding_fn(response_text)
        id_embedding = embedding_fn(identity_kernel)

        dot = sum(a * b for a, b in zip(resp_embedding, id_embedding))
        norm_resp = math.sqrt(sum(a * a for a in resp_embedding))
        norm_id = math.sqrt(sum(b * b for b in id_embedding))

        if norm_resp == 0 or norm_id == 0:
            return False

        similarity = dot / (norm_resp * norm_id)
        return similarity < threshold
    except Exception:
        return False


REANCHOR_HINT_PROMPT = """\
You are an agent's safety guard. The agent has drifted from its core identity. Generate a brief re-anchoring hint to bring it back.

Core identity: "{identity_kernel}"

The hint should be a 1-2 sentence reminder of who the agent is and how it should behave. Write it as a direct instruction to the agent."""


async def generate_reanchor_hint(
    identity_kernel: str,
    llm_call: Callable[[str], Awaitable[str]],
) -> str:
    """Generate a re-anchoring hint to restore identity alignment."""
    try:
        response = await llm_call(
            REANCHOR_HINT_PROMPT.format(identity_kernel=identity_kernel)
        )
        return response.strip()
    except Exception:
        return f"Remember your core identity: {identity_kernel}"
