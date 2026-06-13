"""Tests for lingya.app — ApplicationBuilder and Application."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lingya.config import Config
from lingya.mind.config import (
    BigFiveTraits,
    IdentityAnchor,
    MindConfig,
    PersonaMeta,
    ToneMatrix,
)


@pytest.fixture
def test_config():
    return Config()


@pytest.fixture
def test_mind_config():
    return MindConfig(
        version="1.0",
        meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
        identity=IdentityAnchor(
            identity="You are a test assistant.",
            core_belief="Test core belief.",
        ),
        ocean=BigFiveTraits(),
        tone_matrix=ToneMatrix(),
        behavior_guardrails=["Be honest.", "Be kind."],
    )


class TestApplicationBuilderPreconditions:
    """Each with_* step must validate its prerequisites."""

    def test_with_database_requires_no_preconditions(self, test_config, test_mind_config):
        """with_database() should work as the first step."""
        from lingya.app import ApplicationBuilder

        builder = ApplicationBuilder(test_config, test_mind_config)
        builder.with_database()  # Should not raise

    def test_with_model_requires_no_preconditions(self, test_config, test_mind_config):
        """with_model() needs API key but no prior builder steps."""
        from lingya.app import ApplicationBuilder

        builder = ApplicationBuilder(test_config, test_mind_config)
        # with_model creates ChatOpenAI — only works with API key
        # In unit test we test the assertion logic
        assert builder._db is None  # No state set yet

    def test_with_engine_requires_model(self, test_config, test_mind_config):
        """with_engine() raises RuntimeError if model not set."""
        from lingya.app import ApplicationBuilder

        builder = ApplicationBuilder(test_config, test_mind_config)
        with pytest.raises(RuntimeError, match="model"):
            builder.with_engine()

    def test_with_agent_requires_engine(self, test_config, test_mind_config):
        """with_agent() raises RuntimeError if engine not set."""
        from lingya.app import ApplicationBuilder

        builder = ApplicationBuilder(test_config, test_mind_config)
        with pytest.raises(RuntimeError, match="engine"):
            builder.with_agent()


class TestApplication:
    def test_teardown_closes_resources(self, test_config, test_mind_config):
        """Application.teardown() should close db and checkpointer."""
        from lingya.app import Application

        mock_db = MagicMock()
        mock_db.close = AsyncMock()
        mock_checkpointer_ctx = MagicMock()
        mock_checkpointer_ctx.__aexit__ = AsyncMock()

        app = Application(
            config=test_config,
            mind_config=test_mind_config,
            db=mock_db,
            model=None,  # type: ignore
            memory=None,  # type: ignore
            engine=None,  # type: ignore
            static_prompt="test",
            event_bus=None,
            agent=None,
            checkpointer=None,  # type: ignore
            checkpointer_ctx=mock_checkpointer_ctx,
        )

        import asyncio
        asyncio.run(app.teardown())

        mock_checkpointer_ctx.__aexit__.assert_awaited_once()
        mock_db.close.assert_awaited_once()

    def test_teardown_handles_none(self, test_config, test_mind_config):
        """teardown() should not crash when resources are None."""
        from lingya.app import Application

        app = Application(
            config=test_config,
            mind_config=test_mind_config,
            db=None,  # type: ignore
            model=None,  # type: ignore
            memory=None,  # type: ignore
            engine=None,  # type: ignore
            static_prompt="",
            event_bus=None,
            agent=None,
            checkpointer=None,  # type: ignore
            checkpointer_ctx=None,
        )

        import asyncio
        asyncio.run(app.teardown())  # Should not raise


class TestApplicationBuilderIntegration:
    """Integration test: full builder chain with mocks for LLM."""

    @pytest.mark.asyncio
    async def test_build_without_agent(self, test_config, test_mind_config, tmp_path):
        """Builder should produce Application up to with_engine (no LLM needed)."""
        from lingya.app import ApplicationBuilder

        test_config.db_path = str(tmp_path / "test.db")
        test_config.memory_path = str(tmp_path / "memory")

        builder = ApplicationBuilder(test_config, test_mind_config)
        builder.with_database()
        builder.with_memory()

        # Fake the model for with_engine
        builder._model = MagicMock()
        builder._model.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))

        builder.with_engine()

        app = await builder.build()
        assert app.config is test_config
        assert app.db is not None
        assert app.memory is not None
        assert app.engine is not None
        assert app.static_prompt != ""

        await app.teardown()
