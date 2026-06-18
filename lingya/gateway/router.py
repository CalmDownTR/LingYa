"""MessageRouter — pure dict-in/dict-out message routing.

Does NOT know about WebSocket. Testable without network.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from lingya.protocols import IMemoryStore


class MessageRouter:
    """Routes messages to handlers. Pure dict-in/dict-out."""

    def __init__(
        self,
        engine: Any,  # MindEngine
        memory: IMemoryStore | Any,
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
            "session": self._handle_session,
            "stats": self._handle_stats,
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

    # ── Route handlers ──────────────────────────────────────────────

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

    async def _handle_session(self, payload: dict) -> dict:
        """Manage conversation sessions — start new, list, or switch."""
        action = payload.get("action", "new")

        if action == "new":
            self._thread_id = f"ws-{uuid.uuid4().hex[:8]}"
            return {
                "type": "session_response",
                "payload": {
                    "action": "new",
                    "thread_id": self._thread_id,
                },
            }

        return {
            "type": "error",
            "payload": {"message": f"Unknown session action: {action}"},
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

    async def _handle_chat(
        self,
        payload: dict,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """Process a chat message through the agent + mind engine pipeline.

        When *emit* is provided, the agent runs via ``astream_events(version="v3")``
        and streaming events are pushed through *emit* as they happen.
        When *emit* is None (backward compat), falls back to ``agent.ainvoke()``.

        Returns the final ``chat_response`` dict.
        """
        text = payload.get("text", "")
        if not text:
            return {"type": "error", "payload": {"message": "Empty message"}}

        if self._agent is None:
            return {"type": "error", "payload": {"message": "Agent not initialized"}}

        # 1. Get dynamic tone fragment from engine
        fragment = self._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=text)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))

        config = {"configurable": {"thread_id": self._thread_id}}

        if emit is not None:
            return await self._handle_chat_streaming(messages, config, text, emit)
        else:
            return await self._handle_chat_invoke(messages, config, text)

    async def _handle_chat_streaming(
        self,
        messages: list,
        config: dict,
        user_text: str,
        emit: Callable[[dict], Awaitable[None]],
    ) -> dict:
        """Run agent via astream_events(version="v3") and emit streaming events."""
        from lingya.transformers import create_lingya_transformer

        accumulated_text = ""

        try:
            run = await self._agent.astream_events(
                {"messages": messages},
                config,
                version="v3",
                transformers=[create_lingya_transformer],
            )

            async for event in run:
                method = event["method"]

                if method == "messages":
                    payload_data, _metadata = event["params"]["data"]
                    if isinstance(payload_data, dict) and "event" in payload_data:
                        if payload_data["event"] == "content-block-delta":
                            delta = payload_data.get("delta", {})
                            if delta.get("type") == "text-delta":
                                chunk = delta.get("text", "")
                                accumulated_text += chunk
                                await emit({
                                    "type": "event",
                                    "event": "chat.delta",
                                    "payload": {"content": chunk},
                                })

                elif method == "lingya_inner":
                    inner_event = event["params"]["data"]
                    await emit({
                        "type": "event",
                        "event": inner_event["type"],
                        "payload": inner_event["payload"],
                    })

        except Exception as e:
            return {"type": "error", "payload": {"message": str(e)}}

        # 4. Process through MindEngine
        t_engine = time.monotonic()
        await self._engine.process_event({
            "event_type": "outcome",
            "valence": "neutral",
            "focus": "self",
            "description": user_text,
            "content": user_text,
        })
        engine_ms = round((time.monotonic() - t_engine) * 1000, 1)
        if accumulated_text:
            await self._engine.check_response_alignment(accumulated_text)

        # Emit mind.transition
        tone = self._engine.get_tone_params()
        pad = self._engine.state.current_pad
        last_emotion = (
            self._engine.state.recent_emotions[-1]
            if self._engine.state.recent_emotions
            else {"emotion": "neutral", "intensity": 0.0}
        )
        await emit({
            "type": "event",
            "event": "mind.transition",
            "payload": {
                "pad": {
                    "pleasure": pad.pleasure,
                    "arousal": pad.arousal,
                    "dominance": pad.dominance,
                },
                "occ_emotion": last_emotion["emotion"],
                "ipc": f"{self._engine.state.ipc_state} (agency={self._engine.state.ipc_agency:.2f}, communion={self._engine.state.ipc_communion:.2f})",
            },
        })

        return {
            "type": "chat_response",
            "payload": {
                "text": accumulated_text,
                "tone": tone,
                "meta": {"engine_ms": engine_ms},
            },
        }

    async def _handle_chat_invoke(
        self,
        messages: list,
        config: dict,
        user_text: str,
    ) -> dict:
        """Fallback path: agent.ainvoke() for backward compatibility."""
        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                config,
            )
        except Exception as e:
            return {"type": "error", "payload": {"message": str(e)}}

        # Extract response text
        msgs = result.get("messages", [])
        ais = [m for m in msgs if isinstance(m, AIMessage)]
        response_text = ais[-1].text if ais else ""

        # Process through MindEngine
        t_engine = time.monotonic()
        await self._engine.process_event({
            "event_type": "outcome",
            "valence": "neutral",
            "focus": "self",
            "description": user_text,
            "content": user_text,
        })
        engine_ms = round((time.monotonic() - t_engine) * 1000, 1)
        if response_text:
            await self._engine.check_response_alignment(response_text)

        return {
            "type": "chat_response",
            "payload": {
                "text": response_text,
                "tone": self._engine.get_tone_params(),
                "meta": {"engine_ms": engine_ms},
            },
        }
