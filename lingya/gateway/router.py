"""MessageRouter — pure dict-in/dict-out message routing.

Does NOT know about WebSocket. Testable without network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MessageRouter:
    """Routes messages to handlers. Pure dict-in/dict-out."""

    def __init__(
        self,
        engine: Any,  # MindEngine
        memory: Any,  # EnhancedMemoryStore
        db: Any,  # Database
        data_dir: str,
    ) -> None:
        self._engine = engine
        self._memory = memory
        self._db = db
        self._data_dir = data_dir

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
        """Search or list memories."""
        action = payload.get("action", "search")

        if action == "list":
            memories = self._memory.list_all()
            return {
                "type": "memory_response",
                "payload": {"action": "list", "memories": memories},
            }

        # action == "search"
        query = payload.get("query", "")
        results = self._memory.search(query)
        return {
            "type": "memory_response",
            "payload": {"action": "search", "results": results},
        }

    async def _handle_chat(self, payload: dict) -> dict:
        """Chat placeholder — returns 'not implemented' for now (Phase B)."""
        return {
            "type": "error",
            "payload": {"message": "Chat not yet implemented"},
        }
