from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from lingya.config import MemoryConfig
from lingya.ingestion.chunker import chunk_text, count_tokens

from .long_term import LongTermMemory, MemoryEntry
from .short_term import Message, ShortTermMemory

if TYPE_CHECKING:
    pass


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig,
        summarize: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.config = config
        self._summarize = summarize
        self.short_term = ShortTermMemory(
            max_messages=config.short_term_max_messages,
        )
        self.long_term = LongTermMemory(
            persist_dir=config.chroma_persist_dir,
            embedding_model_name=config.embedding_model,
        )
        self._turn_count: int = 0

    async def add_message(self, message: Message) -> None:
        self.short_term.add(message)
        self._turn_count += 1

    async def retrieve_context(self, query: str) -> list[MemoryEntry]:
        return await self.long_term.search(query, top_k=self.config.long_term_top_k)

    async def compress_context(self, max_tokens: int) -> str:
        """Compress old messages to fit within max_tokens budget.

        Pops oldest messages, summarizes them via LLM, and prepends the summary
        as a system message. Does NOT store to ChromaDB — this is a same-session
        context window management concern only.
        """
        if not self._summarize:
            return ""

        # Pop messages until estimated tokens within budget, keeping at least 2
        while self.estimate_message_tokens() > max_tokens and len(self.short_term) > 2:
            excess = max(1, len(self.short_term) // 4)
            old_messages = self.short_term.pop_compressible(excess)
            conversation_text = "\n".join(
                f"{m.role}: {m.content}" for m in old_messages
            )

            summary = await self._summarize(
                system_prompt=(
                    "You are a conversation summarizer. Summarize the key points, "
                    "topics, and any important details from this conversation. "
                    "Be concise but thorough. Write in the same language as the conversation."
                ),
                user_message=f"Summarize this conversation excerpt:\n\n{conversation_text}",
                max_tokens=512,
            )

            self.short_term.prepend(Message(
                role="system",
                content=f"[Compressed context]: {summary}",
            ))

        return ""

    async def save_to_long_term(self, content: str, source: str) -> list[str]:
        """Chunk and store content into long-term memory for future retrieval."""
        chunks = chunk_text(content)
        if not chunks:
            return []

        now = datetime.now(timezone.utc).isoformat()
        entries: list[MemoryEntry] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"agent_saved_{i}_{_uuid_safe()}"
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

        await self.long_term.store(entries)
        return [e.id for e in entries]

    async def ingest_content(self, text: str, source: str, content_type: str) -> list[str]:
        """Ingest external content (e.g. from /fetch). Kept for CLI compatibility."""
        chunks = chunk_text(text)
        if not chunks:
            return []

        now = datetime.now(timezone.utc).isoformat()
        entries: list[MemoryEntry] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"ingest_{content_type}_{i}_{_uuid_safe()}"
            entries.append(MemoryEntry(
                id=chunk_id,
                text=chunk,
                metadata={
                    "type": content_type,
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": now,
                },
            ))

        await self.long_term.store(entries)
        return [e.id for e in entries]

    def estimate_message_tokens(self) -> int:
        """Estimate total tokens across all deque messages using tiktoken."""
        total = 0
        for m in self.short_term.get_messages():
            total += count_tokens(m.content)
        return total


def _uuid_safe() -> str:
    import uuid
    return str(uuid.uuid4())[:8]
