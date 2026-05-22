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
    db.get_personality = AsyncMock(return_value=None)
    db.save_personality = AsyncMock()
    return db


@pytest.fixture
def default_genome():
    """Default PersonalityGenome (LingYa defaults)."""
    from lingya.personality.model import PersonalityGenome

    return PersonalityGenome()


@pytest.fixture
def genome_with_high_traits():
    """Genome with all traits at 0.9."""
    from lingya.personality.model import PersonalityGenome

    return PersonalityGenome(
        exploration=0.9,
        analytical_depth=0.9,
        playfulness=0.9,
        empathy=0.9,
        directness=0.9,
        adaptability=0.9,
    )


@pytest.fixture
def genome_with_low_traits():
    """Genome with all traits at 0.1."""
    from lingya.personality.model import PersonalityGenome

    return PersonalityGenome(
        exploration=0.1,
        analytical_depth=0.1,
        playfulness=0.1,
        empathy=0.1,
        directness=0.1,
        adaptability=0.1,
    )
