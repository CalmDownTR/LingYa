"""MessageRouter — pure dict-in/dict-out message routing.

Does NOT know about WebSocket or HTTP. Testable without network.
Session/settings/chat business logic extracted to dedicated services (v0.9.5).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MessageRouter:
    """Routes messages to handlers. Pure dict-in/dict-out.

    Business logic is delegated to SessionService, SettingsService, and
    ChatHandler. This class is responsible only for dispatch and lightweight
    handlers (ping, stats, mind, diary, memory).
    """

    def __init__(
        self,
        engine: Any,
        memory: Any,
        data_dir: str,
        session_service: Any,
        settings_service: Any,
        chat_handler: Any,
    ) -> None:
        self._engine = engine
        self._memory = memory
        self._data_dir = data_dir
        self._session_service = session_service
        self._settings_service = settings_service
        self._chat_handler = chat_handler

    # ── Properties (backward compat) ─────────────────────────────────

    @property
    def _agent(self):
        """Backward compat: delegate to ChatHandler._agent."""
        return self._chat_handler._agent

    @property
    def _thread_id(self) -> str:
        """Backward compat: delegate to SessionService.thread_id."""
        return self._session_service.thread_id

    @_thread_id.setter
    def _thread_id(self, value: str) -> None:
        self._session_service.thread_id = value

    # ── Delegation helpers (backward compat for REST endpoints) ──────
    # Underbar-prefixed methods are called directly by server.py REST handlers.
    # They delegate to the corresponding services introduced in v0.9.5.

    async def _handle_session(self, payload: dict) -> dict:
        """Delegate to SessionService."""
        return await self._session_service.handle_session(payload)

    async def _handle_settings(self, payload: dict) -> dict:
        """Delegate to SettingsService."""
        return await self._settings_service.handle_settings(payload)

    async def _handle_chat(self, payload: dict, emit=None) -> dict:
        """Delegate to ChatHandler."""
        return await self._chat_handler.handle_chat(payload, emit=emit)

    async def _handle_chat_streaming(self, messages, config, user_text):
        """Delegate to ChatHandler internal streaming method."""
        async for event in self._chat_handler._chat_streaming(messages, config, user_text):
            yield event

    # ── Routing ─────────────────────────────────────────────────────

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
            "stats": self._handle_stats,
            "session": self._session_service.handle_session,
            "settings": self._settings_service.handle_settings,
            "chat": self._chat_handler.handle_chat,
        }

        handler = handlers.get(msg_type)
        if handler is None:
            return {
                "type": "error",
                "payload": {"message": f"Unknown message type: {msg_type}"},
            }

        try:
            response = await handler(payload)
        except Exception as e:
            response = {"type": "error", "payload": {"message": str(e)}}
        return response

    # ── Lightweight handlers ──────────────────────────────────────

    async def _handle_ping(self, payload: dict) -> dict:
        """Return pong with ISO 8601 timestamp."""
        return {
            "type": "pong",
            "payload": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }

    async def _handle_stats(self, payload: dict) -> dict:
        """Stats migrated to OpenTelemetry — return deprecation notice."""
        return {
            "type": "stats_response",
            "payload": {
                "deprecated": True,
                "hint": "Stats migrated to OpenTelemetry. Set otel.enabled=true in config.yaml.",
            },
        }

    async def _handle_mind(self, payload: dict) -> dict:
        """Return mind state, tone params, or health based on query type."""
        query = payload.get("query", "state")
        engine = self._engine

        if query == "tone":
            tone = engine.get_tone_params()
            return {"type": "mind_state", "payload": {"tone": tone}}

        if query == "health":
            health = engine.get_health()
            return {"type": "mind_health", "payload": health}

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
