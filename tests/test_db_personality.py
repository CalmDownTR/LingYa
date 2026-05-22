from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


class TestDatabasePersonality:
    async def test_get_personality_returns_none_when_empty(self, db):
        assert await db.get_personality() is None

    @pytest.mark.parametrize(
        "data",
        [
            {"name": "TestBot", "exploration": 0.85, "version": 1},
            {"name": "灵牙", "role": "一个有思想的AI伙伴"},
            {
                "name": "TestBot",
                "traits": {"exploration": 0.9, "playfulness": 0.3},
                "interests": ["AI", "哲学", "music"],
                "version": 1,
            },
        ],
    )
    async def test_save_and_get_personality_roundtrip(self, db, data):
        await db.save_personality(data)
        loaded = await db.get_personality()
        assert loaded is not None
        # Check all top-level keys are preserved
        for key in data:
            assert loaded[key] == data[key]

    async def test_save_personality_updates_existing(self, db):
        await db.save_personality({"name": "V1", "version": 1})
        await db.save_personality({"name": "V2", "version": 2})
        loaded = await db.get_personality()
        assert loaded["name"] == "V2"
        assert loaded["version"] == 2

    async def test_personality_and_conversations_coexist(self, db):
        await db.save_personality({"name": "Bot", "version": 1})
        cid = await db.create_conversation("Test Session")
        loaded_p = await db.get_personality()
        loaded_c = await db.get_conversation(cid)
        assert loaded_p is not None
        assert loaded_p["name"] == "Bot"
        assert loaded_c is not None
        assert loaded_c["title"] == "Test Session"
