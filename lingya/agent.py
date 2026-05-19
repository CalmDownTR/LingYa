from __future__ import annotations

from datetime import datetime, timezone

from lingya.config import Config
from lingya.llm.factory import create_backend
from lingya.memory.manager import MemoryManager
from lingya.memory.short_term import Message
from lingya.personality.engine import PersonalityEngine
from lingya.storage.db import Database


class LingYaAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.db_path)
        self.llm = create_backend(config.llm)
        self.memory = MemoryManager(config.memory, self.llm)
        self.personality = PersonalityEngine(config.personality, self.llm, self.db)
        self._conv_id: int | None = None

    async def initialize(self) -> None:
        await self.db.initialize()
        await self.personality.load()
        await self._ensure_conversation()

    async def shutdown(self) -> None:
        await self.db.close()

    async def _ensure_conversation(self) -> None:
        if self._conv_id is not None:
            return
        title = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self._conv_id = await self.db.create_conversation(title)

    async def list_sessions(self) -> list[dict]:
        return await self.db.list_conversations()

    async def new_session(self, title: str | None = None) -> str:
        if title is None:
            title = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self._conv_id = await self.db.create_conversation(title)
        return f"Created session #{self._conv_id}: {title}"

    async def switch_session(self, session_id: int) -> str:
        conv = await self.db.get_conversation(session_id)
        if conv is None:
            return f"Session #{session_id} does not exist."
        self._conv_id = session_id
        return f"Switched to session #{session_id}: {conv['title']}"

    async def handle_input(self, user_input: str) -> str:
        # 1. Add user message to short-term memory
        user_msg = Message(role="user", content=user_input)
        await self.memory.add_message(user_msg)
        await self.db.log_turn(self._conv_id, "user", user_input)  # type: ignore[arg-type]

        # 2. Retrieve relevant long-term memories
        retrieved = await self.memory.retrieve_context(user_input)
        memories_text = ""
        if retrieved:
            parts = []
            for entry in retrieved:
                src = entry.metadata.get("source", "unknown")
                parts.append(f"[{src}] {entry.text}")
            memories_text = "\n".join(parts)

        # 3. Build context
        compressed_summary, recent_messages = self.memory.build_context_for_llm()

        # Construct full system prompt
        system_prompt = self.personality.get_system_prompt(user_input)

        if memories_text:
            system_prompt += f"\n\n## Relevant Past Memories\n{memories_text}"

        if compressed_summary:
            system_prompt += f"\n\n## Recent Context Summary\n{compressed_summary}"

        # 4. Call LLM
        try:
            response = await self.llm.generate(
                system_prompt=system_prompt,
                messages=recent_messages,
            )
        except Exception as e:
            error_msg = f"[Error calling LLM: {e}]"
            # Store error as assistant message too
            assistant_msg = Message(role="assistant", content=error_msg)
            await self.memory.add_message(assistant_msg)
            return error_msg

        # 5. Store assistant response
        assistant_msg = Message(role="assistant", content=response.text)
        await self.memory.add_message(assistant_msg)
        await self.db.log_turn(self._conv_id, "assistant", response.text)  # type: ignore[arg-type]

        # 6. Maybe compress short-term memory
        await self.memory.compress_if_needed()

        # 7. Maybe evolve personality (v0.2.0 stub)
        await self.personality.maybe_evolve(
            recent_summary=compressed_summary or self.memory.short_term.get_conversation_text()
        )

        return response.text

    async def ingest_and_learn(self, content: str, source: str, content_type: str) -> str:
        chunk_ids = await self.memory.ingest_content(content, source, content_type)
        if chunk_ids:
            return f"Ingested {len(chunk_ids)} chunks from {source} ({content_type})."
        return "No content was ingested."

    async def reflect(self) -> str:
        conversation = self.memory.short_term.get_conversation_text()
        if not conversation.strip():
            return "Nothing to reflect on yet."

        prompt = (
            "Analyze this conversation and provide insights:\n"
            "1. What are the main themes?\n"
            "2. What have you learned about the user?\n"
            "3. Any patterns in the interaction?\n\n"
            f"Conversation:\n{conversation}"
        )

        result = await self.llm.generate_simple(
            system_prompt="You are a reflective analyst. Provide honest, insightful analysis.",
            user_message=prompt,
            max_tokens=512,
        )
        return result
