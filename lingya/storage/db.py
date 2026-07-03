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

