from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


class TestDatabaseSession:
    async def test_create_conversation_returns_id(self, db):
        id1 = await db.create_conversation("Session A")
        id2 = await db.create_conversation("Session B")
        assert isinstance(id1, int)
        assert isinstance(id2, int)
        assert id2 > id1

    async def test_get_conversation_existing_and_missing(self, db):
        cid = await db.create_conversation("Test")
        conv = await db.get_conversation(cid)
        assert conv is not None
        assert conv["title"] == "Test"

        missing = await db.get_conversation(999)
        assert missing is None

    async def test_list_conversations(self, db):
        c1 = await db.create_conversation("S1")
        c2 = await db.create_conversation("S2")

        sessions = await db.list_conversations()
        assert len(sessions) == 2
        ids = {s["id"] for s in sessions}
        assert ids == {c1, c2}
        for s in sessions:
            assert "title" in s
            assert "created_at" in s
            assert "updated_at" in s

    async def test_update_conversation_timestamp(self, db):
        cid = await db.create_conversation("Test")
        before = (await db.get_conversation(cid))["updated_at"]
        await asyncio.sleep(1.1)
        await db.update_conversation_timestamp(cid)
        after = (await db.get_conversation(cid))["updated_at"]
        assert after > before

    async def test_conversation_timestamps(self, db):
        cid = await db.create_conversation("Timestamps")
        conv = await db.get_conversation(cid)
        assert conv is not None
        assert "created_at" in conv
        assert "updated_at" in conv
        assert isinstance(conv["created_at"], str)
