"""SessionService — thread/session CRUD, persistence, and history loading.

Extracted from MessageRouter (v0.9.5 router.py split). Owns thread_id
lifecycle and all checkpoint-table operations.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionService:
    """Manages conversation sessions: thread CRUD, persistence, history.

    Owns the current thread_id and persists it to disk so it survives
    daemon restarts. All checkpoint-table operations go through this class.
    """

    def __init__(self, db: Any, data_dir: str, thread_id: str = "ws-default") -> None:
        self._db = db
        self._current_session_file = Path(data_dir) / "current_session.txt"
        self._thread_id = self._load_persisted_thread_id() or thread_id
        self._agent: Any = None

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

    # ── thread_id property ─────────────────────────────────────────

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @thread_id.setter
    def thread_id(self, value: str) -> None:
        self._thread_id = value

    # ── Session dispatch ──────────────────────────────────────────

    async def handle_session(self, payload: dict) -> dict:
        """Manage conversation sessions — new, switch, delete, list, current, history."""
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
        """Delete all checkpoints for a thread_id."""
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
        """List all sessions ordered by most recent activity."""
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
            cnt_cur = await self._db.conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (tid,),
            )
            cnt_row = await cnt_cur.fetchone()
            count = cnt_row[0] if cnt_row else 0
            message_count = max(0, count - 1)
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

        Requires the agent reference for state loading — set via
        set_agent() before any history reads.
        """
        if self._agent is None:
            return []
        try:
            state = await self._agent.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
        except Exception:
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
        return messages

    # ── Agent reference (for history loading) ─────────────────────

    def set_agent(self, agent: Any) -> None:
        """Set the agent reference for history loading."""
        self._agent = agent

    # ── Content extraction (static) ────────────────────────────────

    @staticmethod
    def _extract_text_content(msg) -> str:
        """Normalise msg.content to a plain string."""
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
        """Normalise an arbitrary value to a plain string."""
        if isinstance(value, str):
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
