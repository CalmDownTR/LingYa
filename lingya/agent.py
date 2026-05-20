from __future__ import annotations

import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import SecretStr

from lingya.config import Config
from lingya.ingestion.loader import ingest_url
from lingya.memory.manager import MemoryManager
from lingya.memory.short_term import Message
from lingya.personality.engine import PersonalityEngine
from lingya.storage.db import Database
from lingya.tools import create_agent_tools


class LingYaAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.db_path)

        self._model = ChatOpenAI(
            model=config.llm.model,
            api_key=SecretStr(os.environ[config.llm.api_key_env]),
            base_url=config.llm.api_base_url,
            temperature=config.llm.temperature,
        )

        self.memory = MemoryManager(
            config.memory,
            summarize=self._summarize,
        )

        self.personality = PersonalityEngine(config.personality, self.db, self._model)

        self._tools = create_agent_tools(self.memory, ingest_url)
        # No checkpointer — each ainvoke is a fresh snapshot; deque is the sole source of truth
        self._graph = create_agent(self._model, tools=self._tools)

        self._conv_id: int | None = None

    # ── summarization callback for MemoryManager ──

    async def _summarize(
        self, system_prompt: str, user_message: str, max_tokens: int
    ) -> str:
        model = self._model.bind(max_tokens=max_tokens)
        result = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        return str(result.content)  # AIMessage.content is str | list; always str for text

    # ── lifecycle ──

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

    # ── session management ──

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

    # ── core agent loop ──

    async def handle_input(self, user_input: str) -> str:
        # 1. Store user message in deque (clean: only user/assistant final messages)
        await self.memory.add_message(Message(role="user", content=user_input))
        await self.db.log_turn(self._conv_id, "user", user_input)  # type: ignore[arg-type]

        # 2. Pre-flight: token budget check — compress if needed
        await self._ensure_context_fits()

        # 3. Build dynamic system prompt (personality + compression summaries)
        system_prompt = self._build_system_prompt()

        # 4. Convert deque to LangChain messages (clean snapshot, no tool internals)
        input_messages = self._deque_to_langchain_messages()

        # 5. SystemMessage first, then history
        formatted = [SystemMessage(content=system_prompt)] + input_messages

        # 6. Stateless agent execution — LangGraph manages tool call loop internally
        try:
            result = await self._graph.ainvoke(
                {"messages": formatted},
                config={"recursion_limit": self.config.memory.max_agent_iterations + 1},
            )
        except Exception as e:
            error_msg = f"[Agent error: {e}]"
            await self.memory.add_message(Message(role="assistant", content=error_msg))
            return error_msg

        # 7. Extract final AI response only; ephemeral tool_history stays in result["messages"]
        final_ai = result["messages"][-1]
        response_text = final_ai.content or ""

        # 8. Persist only the final answer back to deque
        await self.memory.add_message(Message(role="assistant", content=response_text))
        await self.db.log_turn(self._conv_id, "assistant", response_text)  # type: ignore[arg-type]

        # Personality evolution (stub)
        await self.personality.maybe_evolve(
            recent_summary=self.memory.short_term.get_conversation_text()
        )

        return response_text

    # ── helpers ──

    async def _ensure_context_fits(self) -> None:
        if not self.config.memory.compression_enabled:
            return

        message_tokens = self.memory.estimate_message_tokens()
        # Reserve 4000 tokens for system prompt + tool definitions + response headroom
        available = self.config.memory.max_context_tokens - 4000
        trigger = int(available * self.config.memory.compression_trigger_ratio)

        if message_tokens > trigger:
            await self.memory.compress_context(trigger)

    def _build_system_prompt(self) -> str:
        parts = [self.personality.personality.to_system_prompt()]

        # Append deque system messages (compression summaries)
        for m in self.memory.short_term.get_messages():
            if m.role == "system":
                parts.append(m.content)

        parts.append(
            "## Available Actions\n"
            "You have access to these tools. Use them when appropriate:\n"
            "- search_memory(query) — Search your long-term memory for relevant past information.\n"
            "- save_memory(content, source) — Save important facts or preferences to long-term memory.\n"
            "- fetch_url(url) — Fetch and read the content of a web page.\n\n"
            "If you can answer the user directly without using any tools, just respond naturally."
        )

        return "\n\n".join(parts)

    def _deque_to_langchain_messages(self) -> list:
        messages: list = []
        for m in self.memory.short_term.get_messages():
            if m.role == "user":
                messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                messages.append(AIMessage(content=m.content))
            # system messages are handled in _build_system_prompt
        return messages

    # ── ingestion & reflection (unchanged contract) ──

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

        result = await self._model.ainvoke([
            SystemMessage(content="You are a reflective analyst. Provide honest, insightful analysis."),
            HumanMessage(content=prompt),
        ])
        return str(result.content)
