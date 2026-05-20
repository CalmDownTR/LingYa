from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool


def create_agent_tools(memory_manager, url_loader: Callable):
    """Create agent tools with closure-bound dependencies.

    Uses closures to inject the agent instance's MemoryManager and URL loader
    into tool functions. This avoids global state and keeps tools testable.
    """

    @tool
    async def search_memory(query: str) -> str:
        """Search your long-term memory bank for relevant past context (conversations, facts, user preferences)."""
        results = await memory_manager.retrieve_context(query)
        if not results:
            return "No relevant memories found."
        parts = [f"[{r.metadata.get('source', 'unknown')}] {r.text}" for r in results]
        return "\n".join(parts)

    @tool
    async def save_memory(content: str, source: str) -> str:
        """Save important facts or user preferences to your long-term memory for future recall."""
        ids = await memory_manager.save_to_long_term(content, source)
        return f"Saved {len(ids)} chunks from {source}."

    @tool
    async def fetch_url(url: str) -> str:
        """Fetch and read the content of a web page by URL. Use this when the user asks about a URL or web content."""
        ids = await url_loader(memory_manager, url)
        return f"Fetched and ingested {len(ids)} chunks from {url}."

    return [search_memory, save_memory, fetch_url]
