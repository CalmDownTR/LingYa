from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from lingya.config import MemoryConfig
from lingya.ingestion.chunker import chunk_text
from lingya.memory.short_term import Message

from .long_term import LongTermMemory, MemoryEntry
from .short_term import ShortTermMemory

if TYPE_CHECKING:
    from lingya.llm.base import BaseLLMBackend


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig,
        llm: BaseLLMBackend,
    ) -> None:
        self.config = config
        self.llm = llm
        self.short_term = ShortTermMemory(
            max_messages=config.short_term_max_messages,
            compression_trigger=config.compression_trigger_messages,
        )
        self.long_term = LongTermMemory(
            persist_dir=config.chroma_persist_dir,
            embedding_model_name=config.embedding_model,
        )
        self._compressed_summary: str = ""
        self._turn_count: int = 0

    async def add_message(self, message: Message) -> None:
        self.short_term.add(message)
        self._turn_count += 1

    async def retrieve_context(self, query: str) -> list[MemoryEntry]:
        return await self.long_term.search(query, top_k=self.config.long_term_top_k)

    async def compress_if_needed(self) -> None:
        if not self.config.compression_enabled:
            return
        if not self.short_term.should_compress():
            return

        # Keep the most recent ~half of max messages
        keep_count = self.config.short_term_max_messages // 2
        excess = len(self.short_term) - keep_count
        if excess <= 0:
            return

        old_messages = self.short_term.pop_compressible(excess)
        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in old_messages
        )

        summary = await self.llm.generate_simple(
            system_prompt="You are a conversation summarizer. Summarize the key points, topics, and any important details from this conversation. Be concise but thorough. Write in the same language as the conversation.",
            user_message=f"Summarize this conversation excerpt:\n\n{conversation_text}",
            max_tokens=512,
        )

        self._compressed_summary = summary

        # Store the compressed summary in long-term memory
        entry = MemoryEntry(
            id=f"compressed_{self._turn_count}",
            text=summary,
            metadata={
                "type": "compressed_conversation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_turns": len(old_messages),
            },
        )
        await self.long_term.store([entry])

    def build_context_for_llm(self) -> tuple[str, list[dict]]:
        """Returns (context_string_for_system_prompt, messages_for_api)."""
        context_parts: list[str] = []

        if self._compressed_summary:
            context_parts.append(f"## Recent Context Summary\n{self._compressed_summary}")

        context = "\n\n".join(context_parts) if context_parts else ""
        messages = [
            {"role": m.role, "content": m.content}
            for m in self.short_term.get_messages()
        ]

        return context, messages

    async def ingest_content(self, text: str, source: str, content_type: str) -> list[str]:
        chunks = chunk_text(text)
        if not chunks:
            return []

        now = datetime.now(timezone.utc).isoformat()
        entries: list[MemoryEntry] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"ingest_{content_type}_{i}_{uuid_safe()}"
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


def uuid_safe() -> str:
    import uuid
    return str(uuid.uuid4())[:8]
