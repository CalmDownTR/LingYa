"""MessageRouter — pure dict-in/dict-out message routing.

Does NOT know about WebSocket. Testable without network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class MessageRouter:
    """Routes messages to handlers. Pure dict-in/dict-out."""

    def __init__(
        self,
        engine: Any,  # MindEngine
        memory: Any,  # EnhancedMemoryStore
        db: Any,  # Database
        data_dir: str,
        agent: Any = None,  # deep agent (create_deep_agent)
        thread_id: str = "ws-default",
    ) -> None:
        self._engine = engine
        self._memory = memory
        self._db = db
        self._data_dir = data_dir
        self._agent = agent
        self._thread_id = thread_id

    async def route(self, message: dict) -> dict:
        """Route a message and return a response dict.

        Message format: {"type": "...", "payload": {...}}
        Response format: {"type": "..._response", "payload": {...}}
          or {"type": "error", "payload": {"message": "..."}}
        """
        msg_type = message.get("type", "")
        payload = message.get("payload", {})

        handlers: dict[str, Any] = {
            "ping": self._handle_ping,
            "mind": self._handle_mind,
            "diary": self._handle_diary,
            "memory": self._handle_memory,
            "chat": self._handle_chat,
        }

        handler = handlers.get(msg_type)
        if handler is None:
            return {
                "type": "error",
                "payload": {"message": f"Unknown message type: {msg_type}"},
            }

        try:
            return await handler(payload)
        except Exception as e:
            return {"type": "error", "payload": {"message": str(e)}}

    # ── Route handlers ──────────────────────────────────────────────

    async def _handle_ping(self, payload: dict) -> dict:
        """Return pong with ISO 8601 timestamp."""
        return {
            "type": "pong",
            "payload": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }

    async def _handle_mind(self, payload: dict) -> dict:
        """Return mind state or tone params based on query type."""
        query = payload.get("query", "state")
        engine = self._engine

        if query == "tone":
            tone = engine.get_tone_params()
            return {"type": "mind_state", "payload": {"tone": tone}}

        # Full state dump
        pad = engine.state.current_pad
        last_emotion = (
            engine.state.recent_emotions[-1]
            if engine.state.recent_emotions
            else {"emotion": "neutral", "intensity": 0.0}
        )
        tone = engine.get_tone_params()
        ocean = engine.state.current_ocean

        return {
            "type": "mind_state",
            "payload": {
                "pad": {
                    "pleasure": pad.pleasure,
                    "arousal": pad.arousal,
                    "dominance": pad.dominance,
                },
                "emotion": last_emotion["emotion"],
                "emotion_intensity": last_emotion["intensity"],
                "ipc_state": engine.state.ipc_state,
                "ipc_agency": engine.state.ipc_agency,
                "ipc_communion": engine.state.ipc_communion,
                "tone": tone,
                "ocean": {
                    "openness": ocean.openness,
                    "conscientiousness": ocean.conscientiousness,
                    "extraversion": ocean.extraversion,
                    "agreeableness": ocean.agreeableness,
                    "neuroticism": ocean.neuroticism,
                },
                "turn_counter": engine.state.turn_counter,
            },
        }

    async def _handle_diary(self, payload: dict) -> dict:
        """List or read diary entries."""
        from lingya.diary import list_diaries, read_diary

        diary_dir = Path(self._data_dir) / "diary"
        action = payload.get("action", "list")

        if action == "read":
            index = payload.get("index", 0)
            result = read_diary(diary_dir, index)
            if result is None:
                return {
                    "type": "error",
                    "payload": {"message": f"No diary at index {index}"},
                }
            diary_date, content = result
            return {
                "type": "diary_response",
                "payload": {
                    "action": "read",
                    "date": diary_date.isoformat(),
                    "content": content,
                },
            }

        # action == "list"
        diaries = list_diaries(diary_dir)
        return {
            "type": "diary_response",
            "payload": {
                "action": "list",
                "diaries": [
                    {"date": d["date"].isoformat(), "preview": d["preview"]}
                    for d in diaries
                ],
            },
        }

    async def _handle_memory(self, payload: dict) -> dict:
        """Search, list, or recover memories."""
        action = payload.get("action", "search")

        if action == "list":
            memories = self._memory.list_all()
            return {
                "type": "memory_response",
                "payload": {"action": "list", "memories": memories},
            }

        if action == "recover":
            mem_id = payload.get("id", "")
            if not mem_id:
                return {
                    "type": "error",
                    "payload": {"message": "Missing memory id"},
                }
            recovered = self._memory.recover(mem_id)
            return {
                "type": "memory_response",
                "payload": {
                    "action": "recover",
                    "recovered": recovered,
                    "id": mem_id,
                },
            }

        # action == "search"
        query = payload.get("query", "")
        results = self._memory.search(query)
        return {
            "type": "memory_response",
            "payload": {"action": "search", "results": results},
        }

    async def _handle_chat(self, payload: dict) -> dict:
        """Process a chat message through the agent + mind engine pipeline."""
        text = payload.get("text", "")
        if not text:
            return {"type": "error", "payload": {"message": "Empty message"}}

        if self._agent is None:
            return {
                "type": "error",
                "payload": {"message": "Agent not initialized"},
            }

        # 1. Get dynamic tone fragment from engine
        fragment = self._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=text)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))

        # 2. Invoke agent
        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                {"configurable": {"thread_id": self._thread_id}},
            )
        except Exception as e:
            return {"type": "error", "payload": {"message": str(e)}}

        # 3. Extract response text
        msgs = result.get("messages", [])
        ais = [m for m in msgs if isinstance(m, AIMessage)]
        response_text = ais[-1].text if ais else ""

        # 4. Process through MindEngine (same as CLI does)
        await self._engine.process_event({
            "event_type": "outcome",
            "valence": "neutral",
            "focus": "self",
            "description": text,
            "content": text,
        })
        if response_text:
            await self._engine.check_response_alignment(response_text)

        # 5. Return response with tone
        return {
            "type": "chat_response",
            "payload": {
                "text": response_text,
                "tone": self._engine.get_tone_params(),
            },
        }
