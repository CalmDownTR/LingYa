"""Memory tools — decoupled from daemon assembly.

create_memory_tools(memory_store) returns a list of @tool-decorated functions
ready for registration with create_agent.
"""

from __future__ import annotations

from langchain_core.tools import tool

from lingya.protocols import IMemoryStore


def create_memory_tools(memory_store: IMemoryStore) -> list:
    """Create memory_store and memory_search tools bound to *memory_store*."""

    @tool
    def remember(text: str) -> str:
        """Remember important information about the user.

        Use this tool when the user shares personal preferences, identity,
        emotional states, or context useful for future interactions.
        Examples: "I like rainy days", "I'm afraid of loneliness",
        "I'm a freelancer".

        Do NOT use for transient information (e.g. "I'm running late"),
        one-time tasks, small talk, or credentials/API keys.
        """
        return memory_store.store(text)

    @tool
    def recall(query: str) -> str:
        """Search for prior memories about the user.

        Use this tool when the user asks "do you remember...", or when you
        need to recall context from past conversations to answer accurately.
        Returns matching memories with their text content.
        """
        results = memory_store.search(query)
        if not results:
            return "(No matching memories found)"
        lines = []
        for r in results:
            lines.append(f"[{r['id']}] {r['text']}")
        return "\n".join(lines)

    return [remember, recall]
