from __future__ import annotations

import json
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

    # -- Personality --

    async def get_personality(self) -> dict | None:
        cur = await self.conn.execute("SELECT data FROM personality WHERE id = 1")
        row = await cur.fetchone()
        return json.loads(row["data"]) if row else None

    async def save_personality(self, data: dict) -> None:
        await self.conn.execute(
            """INSERT INTO personality (id, data, updated_at)
               VALUES (1, ?, datetime('now'))
               ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
            (json.dumps(data, ensure_ascii=False),),
        )
        await self.conn.commit()

    # -- Conversations --

    async def create_conversation(self, title: str) -> int:
        cur = await self.conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        await self.conn.commit()
        return cur.lastrowid

    async def log_turn(self, conv_id: int, role: str, content: str) -> None:
        await self.conn.execute(
            "INSERT INTO turns (conversation_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )
        await self.conn.execute(
            "UPDATE conversations SET updated_at=datetime('now') WHERE id=?",
            (conv_id,),
        )
        await self.conn.commit()

    async def get_conversation_turns(self, conv_id: int) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT role, content, created_at FROM turns WHERE conversation_id=? ORDER BY id",
            (conv_id,),
        )
        return [dict(row) for row in await cur.fetchall()]

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        cur = await self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in await cur.fetchall()]

    # -- Settings --

    async def get_setting(self, key: str) -> str | None:
        cur = await self.conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None

    async def set_setting(self, key: str, value: str) -> None:
        await self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        await self.conn.commit()

    # -- Reflection log --

    async def log_reflection(
        self, old_personality: dict | None, new_personality: dict, reason: str
    ) -> None:
        await self.conn.execute(
            "INSERT INTO reflection_log (old_personality, new_personality, reason) VALUES (?, ?, ?)",
            (
                json.dumps(old_personality, ensure_ascii=False) if old_personality else None,
                json.dumps(new_personality, ensure_ascii=False),
                reason,
            ),
        )
        await self.conn.commit()
