from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lingya.storage.db import Database


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        yield db_path


class TestDatabaseSession:
    @pytest.mark.asyncio
    async def test_create_conversation_returns_id(self, tmp_db):
        db = Database(tmp_db)
        await db.initialize()
        try:
            id1 = await db.create_conversation("Session A")
            id2 = await db.create_conversation("Session B")
            assert isinstance(id1, int)
            assert isinstance(id2, int)
            assert id2 > id1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_get_conversation_existing_and_missing(self, tmp_db):
        db = Database(tmp_db)
        await db.initialize()
        try:
            cid = await db.create_conversation("Test")
            conv = await db.get_conversation(cid)
            assert conv is not None
            assert conv["title"] == "Test"

            missing = await db.get_conversation(999)
            assert missing is None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_list_conversations_with_turn_counts(self, tmp_db):
        db = Database(tmp_db)
        await db.initialize()
        try:
            c1 = await db.create_conversation("S1")
            c2 = await db.create_conversation("S2")

            await db.log_turn(c1, "user", "hello")
            await db.log_turn(c1, "assistant", "hi")

            await db.log_turn(c2, "user", "one turn only")

            sessions = await db.list_conversations()
            assert len(sessions) == 2

            s1 = next(s for s in sessions if s["id"] == c1)
            s2 = next(s for s in sessions if s["id"] == c2)
            assert s1["turn_count"] == 2
            assert s2["turn_count"] == 1
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_log_turn_persists(self, tmp_db):
        db = Database(tmp_db)
        await db.initialize()
        try:
            cid = await db.create_conversation("Test")
            await db.log_turn(cid, "user", "Are you there?")
            await db.log_turn(cid, "assistant", "Yes, I am.")

            turns = await db.get_conversation_turns(cid)
            assert len(turns) == 2
            assert turns[0]["role"] == "user"
            assert turns[0]["content"] == "Are you there?"
            assert turns[1]["role"] == "assistant"
            assert turns[1]["content"] == "Yes, I am."
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_conversation_timestamps(self, tmp_db):
        db = Database(tmp_db)
        await db.initialize()
        try:
            cid = await db.create_conversation("Timestamps")
            conv = await db.get_conversation(cid)
            assert conv is not None
            assert "created_at" in conv
            assert "updated_at" in conv
            assert isinstance(conv["created_at"], str)
        finally:
            await db.close()
