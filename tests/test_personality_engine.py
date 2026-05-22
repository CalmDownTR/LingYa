from __future__ import annotations

import json

import pytest

from lingya.config import PersonalityConfig
from lingya.personality.engine import PersonalityEngine
from lingya.personality.model import ActivePersonality

@pytest.fixture
def default_config():
    return PersonalityConfig()


@pytest.fixture
def engine(mock_db, default_config):
    return PersonalityEngine(config=default_config, db=mock_db, llm=None)


class TestPersonalityEngineInit:
    def test_default_genome_is_lingya(self, engine):
        assert engine._genome.name == "LingYa"

    def test_turn_counter_starts_at_zero(self, engine):
        assert engine._turn_since_reflection == 0


class TestPersonalityEngineGetSystemPrompt:
    @pytest.mark.parametrize("text", ["Hello", "", "hi"])
    def test_returns_string_containing_name(self, engine, text):
        prompt = engine.get_system_prompt(text)
        assert isinstance(prompt, str)
        assert "LingYa" in prompt

    @pytest.mark.parametrize(
        "text",
        [
            "I have a crash! It's broken!",
            "I disagree with what you said",
            "lol 哈哈 that's funny",
            "Explain the architecture of this code",
        ],
    )
    def test_various_situations(self, engine, text):
        prompt = engine.get_system_prompt(text)
        assert "LingYa" in prompt

    def test_system_prompt_reflects_genome_changes(self, engine):
        engine._genome.name = "TestBot"
        engine._genome.playfulness = 0.9
        prompt = engine.get_system_prompt("hi")
        assert "TestBot" in prompt
        assert "humor" in prompt.lower()


class TestPersonalityEngineLoad:
    pytestmark = pytest.mark.asyncio

    async def test_load_from_db(self, mock_db, default_config):
        mock_db.get_personality.return_value = {"name": "CustomBot", "version": 1}
        eng = PersonalityEngine(config=default_config, db=mock_db, llm=None)
        await eng.load()
        assert eng._genome.name == "CustomBot"

    async def test_load_no_data_keeps_default(self, engine, mock_db):
        mock_db.get_personality.return_value = None
        await engine.load()
        assert engine._genome.name == "LingYa"

    async def test_load_from_seed_file(self, mock_db, tmp_path):
        seed_path = tmp_path / "seed.json"
        seed_path.write_text(json.dumps({"name": "SeedBot", "version": 1}))

        config = PersonalityConfig(seed_personality=str(seed_path))
        eng = PersonalityEngine(config=config, db=mock_db, llm=None)
        await eng.load()
        assert eng._genome.name == "SeedBot"

    async def test_seed_file_not_found_silently_falls_back(self, mock_db):
        config = PersonalityConfig(seed_personality="/nonexistent/seed.json")
        eng = PersonalityEngine(config=config, db=mock_db, llm=None)
        await eng.load()
        assert eng._genome.name == "LingYa"

    async def test_seed_file_invalid_json_silently_falls_back(self, mock_db, tmp_path):
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json")

        config = PersonalityConfig(seed_personality=str(bad_path))
        eng = PersonalityEngine(config=config, db=mock_db, llm=None)
        await eng.load()
        assert eng._genome.name == "LingYa"

    async def test_db_takes_priority_over_seed(self, mock_db, tmp_path):
        mock_db.get_personality.return_value = {"name": "DBBot", "version": 1}

        seed_path = tmp_path / "seed.json"
        seed_path.write_text(json.dumps({"name": "SeedBot", "version": 1}))

        config = PersonalityConfig(seed_personality=str(seed_path))
        eng = PersonalityEngine(config=config, db=mock_db, llm=None)
        await eng.load()
        assert eng._genome.name == "DBBot"


class TestPersonalityEngineSave:
    pytestmark = pytest.mark.asyncio

    async def test_save_updates_timestamp_and_calls_db(self, engine, mock_db):
        engine._genome.name = "UpdatedBot"
        await engine.save()
        mock_db.save_personality.assert_awaited_once()
        saved_data = mock_db.save_personality.call_args[0][0]
        assert saved_data["name"] == "UpdatedBot"
        assert "last_updated" in saved_data


class TestPersonalityEngineMaybeEvolve:
    pytestmark = pytest.mark.asyncio

    async def test_returns_false_before_interval(self, engine):
        engine._turn_since_reflection = 0
        result = await engine.maybe_evolve("some summary")
        assert result is False
        assert engine._turn_since_reflection == 1

    async def test_resets_counter_after_interval(self, engine):
        engine._turn_since_reflection = 9
        result = await engine.maybe_evolve("some summary")
        assert result is False
        assert engine._turn_since_reflection == 0


class TestPersonalityEnginePersonalityProperty:
    def test_returns_active_personality(self, engine):
        active = engine.personality
        assert isinstance(active, ActivePersonality)
        assert active.name == "LingYa"
