from __future__ import annotations

from langchain_core.tools import tool

from lingya.ingestion.chunker import chunk_text
from lingya.memory.long_term import LongTermMemory, MemoryEntry


def create_memory_tools(long_term: LongTermMemory):
    """Create memory tools bound to a LongTermMemory instance.

    These are exposed to the agent alongside MCP and deepagents built-in tools.
    """

    @tool
    async def search_memory(query: str) -> str:
        """Search long-term memory for relevant past context (conversations, facts, user preferences)."""
        results = await long_term.search(query)
        if not results:
            return "No relevant memories found."
        parts = [f"[{r.metadata.get('source', 'unknown')}] {r.text}" for r in results]
        return "\n".join(parts)

    @tool
    async def save_memory(content: str, source: str) -> str:
        """Save important facts or user preferences to long-term memory for future recall."""
        chunks = chunk_text(content)
        if not chunks:
            return "Nothing to save."

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        entries: list[MemoryEntry] = []
        for i, chunk in enumerate(chunks):
            import uuid
            chunk_id = f"agent_saved_{i}_{uuid.uuid4().hex[:8]}"
            entries.append(MemoryEntry(
                id=chunk_id,
                text=chunk,
                metadata={
                    "type": "agent_saved",
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": now,
                },
            ))

        await long_term.store(entries)
        return f"Saved {len(entries)} chunks from {source}."

    return [search_memory, save_memory]
