"""MessageRouter — pure dict-in/dict-out message routing.

Does NOT know about WebSocket. Testable without network.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from lingya.protocols import IMemoryStore


logger = logging.getLogger(__name__)


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
        # Persisted current thread_id survives daemon restarts.
        # Falls back to the constructor arg if no persisted value exists.
        self._current_session_file = Path(data_dir) / "current_session.txt"
        self._thread_id = self._load_persisted_thread_id() or thread_id

    # ── Persistence helpers ────────────────────────────────────────

    def _load_persisted_thread_id(self) -> str | None:
        """Load the persisted current thread_id, if any. Returns None on miss."""
        try:
            content = self._current_session_file.read_text(encoding="utf-8").strip()
            return content or None
        except (FileNotFoundError, OSError):
            return None

    def _persist_thread_id(self, thread_id: str) -> None:
        """Persist the current thread_id so it survives daemon restarts."""
        try:
            self._current_session_file.parent.mkdir(parents=True, exist_ok=True)
            self._current_session_file.write_text(thread_id, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to persist current thread_id: %s", exc)

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
            "settings": self._handle_settings,
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
        """Manage conversation sessions — new, switch, delete, list, current."""
        action = payload.get("action", "new")

        if action == "new":
            self._thread_id = f"ws-{uuid.uuid4().hex[:8]}"
            self._persist_thread_id(self._thread_id)
            return {
                "type": "session_response",
                "payload": {
                    "action": "new",
                    "thread_id": self._thread_id,
                },
            }

        if action == "switch":
            thread_id = payload.get("thread_id")
            if not thread_id:
                return {
                    "type": "error",
                    "payload": {"message": "Missing thread_id for switch"},
                }
            exists = await self._thread_exists(thread_id)
            if not exists:
                return {
                    "type": "error",
                    "payload": {"message": f"Session {thread_id} not found"},
                }
            old_id = self._thread_id
            self._thread_id = thread_id
            self._persist_thread_id(self._thread_id)
            return {
                "type": "session_response",
                "payload": {
                    "action": "switch",
                    "thread_id": thread_id,
                    "previous": old_id,
                },
            }

        if action == "delete":
            thread_id = payload.get("thread_id")
            if not thread_id:
                return {
                    "type": "error",
                    "payload": {"message": "Missing thread_id for delete"},
                }
            if thread_id == self._thread_id:
                return {
                    "type": "error",
                    "payload": {"message": "Cannot delete current session"},
                }
            await self._delete_thread(thread_id)
            return {
                "type": "session_response",
                "payload": {
                    "action": "delete",
                    "thread_id": thread_id,
                    "deleted": True,
                },
            }

        if action == "list":
            sessions = await self._list_sessions()
            return {
                "type": "session_response",
                "payload": {"action": "list", "sessions": sessions},
            }

        if action == "current":
            info = await self._session_info(self._thread_id)
            return {
                "type": "session_response",
                "payload": {"action": "current", "session": info},
            }

        if action == "history":
            thread_id = payload.get("thread_id", self._thread_id)
            messages = await self._load_history(thread_id)
            return {
                "type": "session_response",
                "payload": {"action": "history", "thread_id": thread_id, "messages": messages},
            }

        return {
            "type": "error",
            "payload": {"message": f"Unknown session action: {action}"},
        }

    # ── Session helpers ────────────────────────────────────────────

    async def _thread_exists(self, thread_id: str) -> bool:
        """Check if a thread_id exists in the checkpoints table."""
        try:
            cursor = await self._db.conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return (row[0] if row else 0) > 0
        except Exception:
            return False

    async def _delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread_id.

        Checkpoints are the sole source of truth for session data —
        conversations/turns tables are legacy and no longer written to.
        """
        try:
            await self._db.conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            await self._db.conn.commit()
            logger.info("Session deleted: thread_id=%s", thread_id)
        except Exception:
            logger.exception("Failed to delete session: thread_id=%s", thread_id)
            raise

    async def _list_sessions(self) -> list[dict]:
        """List all sessions ordered by most recent activity.

        LangGraph's checkpoint_id is a time-ordered UUID (v1), so
        MAX(checkpoint_id) per thread_id gives the last-activity order
        without needing a separate timestamp column.
        """
        try:
            cursor = await self._db.conn.execute(
                "SELECT thread_id, MAX(checkpoint_id) AS last_cp "
                "FROM checkpoints GROUP BY thread_id ORDER BY last_cp DESC"
            )
            rows = await cursor.fetchall()
        except Exception:
            logger.exception("Failed to list sessions from checkpoints table")
            return []

        sessions = []
        for row in rows:
            tid = row[0] if isinstance(row, tuple) else row["thread_id"]
            last_cp = row[1] if isinstance(row, tuple) else row["last_cp"]
            # Count checkpoints for this thread
            cnt_cur = await self._db.conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (tid,),
            )
            cnt_row = await cnt_cur.fetchone()
            count = cnt_row[0] if cnt_row else 0
            message_count = max(0, count - 1)  # Subtract initial checkpoint
            # Generate a readable label
            short_id = tid[-8:] if len(tid) > 8 else tid
            label = f"会话 {short_id}"
            sessions.append({
                "thread_id": tid,
                "label": label,
                "message_count": message_count,
                "last_activity": last_cp,
                "is_current": tid == self._thread_id,
            })
        return sessions

    async def _session_info(self, thread_id: str) -> dict | None:
        """Get info for a specific session."""
        try:
            cursor = await self._db.conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            count = row[0] if row else 0
            short_id = thread_id[-8:] if len(thread_id) > 8 else thread_id
            return {
                "thread_id": thread_id,
                "label": f"会话 {short_id}",
                "message_count": max(0, count - 1),
                "is_current": thread_id == self._thread_id,
            }
        except Exception:
            return None

    async def _load_history(self, thread_id: str) -> list[dict]:
        """Load conversation history for a thread_id from LangGraph checkpointer.

        In recent LangChain versions, ``AIMessage.content`` may be a list of
        content blocks (e.g. ``[{"type": "text", "text": "..."}]``)
        rather than a plain string — especially when the message was produced
        via ``astream_events(version="v3")``.  This helper normalises both
        shapes to a plain string so the frontend always receives
        ``{"role": "her", "content": "..."}`` with *content* being a string.
        """
        if self._agent is None:
            return []
        try:
            state = await self._agent.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
        except Exception:
            # Log the error so silent failures are debuggable — returning []
            # makes the frontend treat this as "empty session" otherwise.
            logger.exception(
                "Failed to load history for thread_id=%s", thread_id
            )
            return []

        if state is None or not state.values:
            return []

        raw_messages = state.values.get("messages", [])
        messages: list[dict] = []
        for msg in raw_messages:
            type_name = msg.__class__.__name__
            if type_name == "HumanMessage":
                messages.append({
                    "role": "user",
                    "content": getattr(msg, "content", ""),
                })
            elif type_name == "AIMessage":
                messages.append({
                    "role": "her",
                    "content": self._extract_text_content(msg),
                })
            # Skip SystemMessage and ToolMessage
        return messages

    @staticmethod
    def _extract_text_content(msg) -> str:
        """Normalise *msg.content* to a plain string.

        Handles:
        - ``str`` → returned as-is.
        - ``list[dict]`` (LangChain ContentBlock format) → extracts all
          ``"text"`` fields and joins them.
        - Anything else → coerced via ``str()``.
        """
        raw = getattr(msg, "content", "")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: list[str] = []
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts) if parts else ""
        return str(raw)

    @staticmethod
    def _extract_text_content_from_value(value) -> str:
        """Normalise an arbitrary *value* to a plain string.

        Like :meth:`_extract_text_content` but operates on a raw value
        (e.g. ``accumulated_text`` that might have been corrupted) rather
        than an ``AIMessage`` attribute.  Also handles the case where
        ``json.dumps(default=str)`` has already converted a ContentBlock
        list into its Python repr string.
        """
        if isinstance(value, str):
            # Could be a repr-of-list like "[{'type': 'text', ...}]"
            stripped = value.strip()
            if stripped.startswith("[") and "'type'" in stripped:
                try:
                    parsed = eval(stripped, {"__builtins__": {}}, {})
                    if isinstance(parsed, list):
                        parts: list[str] = []
                        for block in parsed:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                text = block.get("text", "")
                                if isinstance(text, str):
                                    parts.append(text)
                        if parts:
                            return "\n".join(parts)
                except Exception:
                    pass
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for block in value:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts) if parts else ""
        return str(value) if value is not None else ""

    async def _handle_settings(self, payload: dict) -> dict:
        """Handle settings get/update/reset operations."""
        from lingya.mind.engine import TONE_PRESETS

        action = payload.get("action", "get")
        engine = self._engine

        if action == "get":
            c = engine.config
            return {
                "type": "settings_response",
                "payload": {
                    "ocean": {
                        "openness": c.ocean.openness,
                        "conscientiousness": c.ocean.conscientiousness,
                        "extraversion": c.ocean.extraversion,
                        "agreeableness": c.ocean.agreeableness,
                        "neuroticism": c.ocean.neuroticism,
                    },
                    "tone": {
                        "warmth": c.tone_matrix.warmth,
                        "formality": c.tone_matrix.formality,
                        "humor": c.tone_matrix.humor,
                    },
                    "identity": {
                        "identity": c.identity.identity,
                        "core_belief": c.identity.core_belief,
                    },
                    "available_presets": list(TONE_PRESETS.keys()),
                },
            }

        if action == "update_ocean":
            ocean = payload.get("ocean", {})
            key_map = {
                "O": "openness", "C": "conscientiousness",
                "E": "extraversion", "A": "agreeableness", "N": "neuroticism",
            }
            mapped: dict[str, float] = {}
            for k, v in ocean.items():
                full_key = key_map.get(k, k)
                val = float(v)
                if not (0.0 <= val <= 1.0):
                    return {
                        "type": "error",
                        "payload": {"message": f"{k}={val} 超出 0-1 范围"},
                    }
                mapped[full_key] = val
            await engine.reload_config({"ocean": mapped})
            full_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]
            new_ocean = {k: getattr(engine.config.ocean, k) for k in full_names}
            return {
                "type": "settings_response",
                "payload": {"ok": True, "ocean": new_ocean},
            }

        if action == "update_identity":
            identity_data = payload.get("identity", {})
            update: dict = {}
            if "identity" in identity_data:
                update["identity"] = identity_data["identity"]
            if "core_belief" in identity_data:
                update["core_belief"] = identity_data["core_belief"]
            if update:
                await engine.reload_config({"identity": update})
            return {"type": "settings_response", "payload": {"ok": True}}

        if action == "update_tone":
            preset = payload.get("preset", "")
            if preset not in TONE_PRESETS:
                valid = list(TONE_PRESETS.keys())
                return {
                    "type": "error",
                    "payload": {"message": f"Unknown preset: {preset}. Valid: {valid}"},
                }
            await engine.reload_config({"tone_preset": preset})
            return {
                "type": "settings_response",
                "payload": {"ok": True, "tone_preset": preset},
            }

        if action == "reset":
            await engine.reload_config({"reset": True})
            c = engine.config
            full_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]
            return {
                "type": "settings_response",
                "payload": {
                    "ok": True,
                    "ocean": {k: getattr(c.ocean, k) for k in full_names},
                },
            }

        return {
            "type": "error",
            "payload": {"message": f"Unknown settings action: {action}"},
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
            final = None
            async for event_dict in self._handle_chat_streaming(
                messages, config, text
            ):
                if event_dict.get("type") in ("chat_response", "error"):
                    final = event_dict
                else:
                    await emit(event_dict)
            return final
        else:
            return await self._handle_chat_invoke(messages, config, text)

    async def _handle_chat_streaming(
        self,
        messages: list,
        config: dict,
        user_text: str,
    ):
        """Run agent via astream_events and yield streaming events + final response.

        Yields ``{"type": "event", "event": "...", "payload": {...}}`` dicts
        for each streaming event, then a final ``{"type": "chat_response", ...}``
        or ``{"type": "error", ...}`` dict.

        The caller decides how to deliver these — emit callback (backward compat),
        SSE frames (FastAPI), or any other transport.
        """
        from lingya.transformers import create_lingya_transformer

        accumulated_text = ""

        # Filter out _subagent_factory from compiled stream_transformers
        # to avoid SubagentTransformer key conflict in the v3 StreamMux.
        # LingYa does not use subagents. See ADR-004 Amendment 2.
        _saved_st = self._agent.stream_transformers
        self._agent.stream_transformers = tuple(
            t for t in _saved_st
            if not (callable(t) and getattr(t, "__name__", "") == "_subagent_factory")
        )

        try:
            run = await self._agent.astream_events(
                {"messages": messages},
                config,
                version="v3",
                transformers=[create_lingya_transformer],
            )

            # Start MindEngine processing concurrently — runs while LLM streams.
            # process_event does 1 LLM call (~1.5s timeout) + DB save + event publish.
            # By the time the stream ends, it's likely already done, eliminating
            # the visible gap between last token and "complete" signal.
            t_engine = time.monotonic()
            engine_task = asyncio.create_task(
                self._engine.process_event({
                    "event_type": "outcome",
                    "valence": "neutral",
                    "focus": "self",
                    "description": user_text,
                    "content": user_text,
                })
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
                                yield {
                                    "type": "event",
                                    "event": "chat.delta",
                                    "payload": {"content": chunk},
                                }

                elif method == "lingya_inner":
                    inner_event = event["params"]["data"]
                    yield {
                        "type": "event",
                        "event": inner_event["type"],
                        "payload": inner_event["payload"],
                    }

            # Wait for engine — process_event does OCC+IPC LLM call
            # (affect.py:_OCC_IPC_TIMEOUT=1.5s) + DB save + event publish.
            # 2.0s gives the LLM call full headroom + 0.5s for persistence.
            try:
                await asyncio.wait_for(engine_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass
            engine_ms = round((time.monotonic() - t_engine) * 1000, 1)

            # Fire-and-forget: response alignment check runs in background.
            # Result (reanchor hint) affects subsequent turns, not this one.
            if accumulated_text:
                asyncio.create_task(
                    self._engine.check_response_alignment(accumulated_text)
                )

            # Yield mind.transition
            tone = self._engine.get_tone_params()
            pad = self._engine.state.current_pad
            last_emotion = (
                self._engine.state.recent_emotions[-1]
                if self._engine.state.recent_emotions
                else {"emotion": "neutral", "intensity": 0.0}
            )
            yield {
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
            }

            # Yield final response — defensive normalisation in case
            # accumulated_text somehow contains non-string data.
            yield {
                "type": "chat_response",
                "payload": {
                    "text": (
                        accumulated_text
                        if isinstance(accumulated_text, str)
                        else self._extract_text_content_from_value(accumulated_text)
                    ),
                    "tone": tone,
                    "meta": {"engine_ms": engine_ms},
                },
            }

        except Exception as e:
            logger.exception("_handle_chat_streaming failed")
            yield {"type": "error", "payload": {"message": str(e)}}
        finally:
            self._agent.stream_transformers = _saved_st

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

        # Extract response text — AIMessage.content may be a ContentBlock
        # list in newer LangChain versions; normalise to plain string.
        msgs = result.get("messages", [])
        ais = [m for m in msgs if isinstance(m, AIMessage)]
        response_text = (
            self._extract_text_content(ais[-1]) if ais else ""
        )

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
