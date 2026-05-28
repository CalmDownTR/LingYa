from __future__ import annotations

from pathlib import Path

import aiosqlite

from .migrations import MIGRATIONS


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._run_migrations()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "Database not initialized"
        return self._conn

    async def _run_migrations(self) -> None:
        await self.conn.execute("PRAGMA journal_mode=WAL")
        # Get current version
        cursor = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        exists = await cursor.fetchone()
        current_version = 0
        if exists:
            cur = await self.conn.execute("SELECT MAX(version) FROM schema_version")
            row = await cur.fetchone()
            if row and row[0] is not None:
                current_version = row[0]

        for i, sql in enumerate(MIGRATIONS[current_version:], start=current_version + 1):
            await self.conn.execute(sql)
            await self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (i,)
            )

        await self.conn.commit()

    # -- Conversations --

    async def create_conversation(self, title: str) -> int:
        cur = await self.conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        await self.conn.commit()
        return cur.lastrowid

    async def update_conversation_timestamp(self, conv_id: int) -> None:
        await self.conn.execute(
            "UPDATE conversations SET updated_at=datetime('now') WHERE id=?",
            (conv_id,),
        )
        await self.conn.commit()

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT id, title, created_at, updated_at
               FROM conversations
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in await cur.fetchall()]

    async def get_conversation(self, conv_id: int) -> dict | None:
        cur = await self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    # -- Turns --

    async def add_turn(self, conversation_id: int, role: str, content: str) -> None:
        await self.conn.execute(
            "INSERT INTO turns (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        await self.conn.commit()

    async def get_turns(
        self, conversation_id: int, limit: int = 6
    ) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT role, content
               FROM turns
               WHERE conversation_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (conversation_id, limit),
        )
        rows = await cur.fetchall()
        # Reverse to chronological order
        return [dict(r) for r in reversed(rows)]

    async def get_turns_since(
        self, since_date: str, limit: int = 200
    ) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT t.role, t.content, t.created_at,
                      c.id as conv_id, c.title as conv_title
               FROM turns t
               JOIN conversations c ON t.conversation_id = c.id
               WHERE t.created_at > ?
               ORDER BY t.id ASC
               LIMIT ?""",
            (since_date, limit),
        )
        return [dict(row) for row in await cur.fetchall()]

    # -- Mind State --

    async def upsert_mind_state(self, state_json: str) -> None:
        """INSERT or REPLACE the singleton mind state row."""
        await self.conn.execute(
            """INSERT INTO mind_state (id, state_json, updated_at)
               VALUES (1, ?, datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
               state_json = excluded.state_json,
               updated_at = excluded.updated_at""",
            (state_json,),
        )
        await self.conn.commit()

    async def get_mind_state(self) -> str | None:
        """Return state_json or None if no saved state."""
        cur = await self.conn.execute(
            "SELECT state_json FROM mind_state WHERE id = 1"
        )
        row = await cur.fetchone()
        return row["state_json"] if row else None

