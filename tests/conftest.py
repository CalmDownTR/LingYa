from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def tmp_db_path():
    """Yields a path to a temporary SQLite database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test.db")


@pytest.fixture
def db(tmp_db_path):
    """Initialized Database instance backed by a temp file."""
    from lingya.storage.db import Database

    async def _init():
        database = Database(tmp_db_path)
        await database.initialize()
        return database

    async def _close(database):
        await database.close()

    database = asyncio.run(_init())
    yield database
    asyncio.run(_close(database))


@pytest.fixture
def mock_db():
    """Mock Database for unit tests that don't need real SQLite."""
    db = MagicMock()
    return db
