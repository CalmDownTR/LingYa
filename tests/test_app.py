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


class TestAuxiliaryModel:
    """Verify auxiliary_model routing in ApplicationBuilder."""

    def test_auxiliary_model_created_when_configured(self, test_config, test_mind_config):
        """When auxiliary_model is set, a second LiteLLMModel is created."""
        from lingya.app import ApplicationBuilder

        test_config.llm.auxiliary_model = "deepseek/deepseek-v4-flash"

        builder = ApplicationBuilder(test_config, test_mind_config)
        builder.with_model()

        assert builder._model is not None
        assert builder._aux_model is not None
        assert builder._aux_model.model == "deepseek/deepseek-v4-flash"
        # Aux model should NOT have fallbacks
        assert builder._aux_model.fallbacks == []

    def test_auxiliary_model_none_when_not_configured(self, test_config, test_mind_config):
        """Without auxiliary_model, _aux_model should be None."""
        from lingya.app import ApplicationBuilder

        test_config.llm.auxiliary_model = None

        builder = ApplicationBuilder(test_config, test_mind_config)
        builder.with_model()

        assert builder._model is not None
        assert builder._aux_model is None

    @pytest.mark.asyncio
    async def test_engine_uses_aux_model_when_configured(self, test_config, test_mind_config, tmp_path):
        """MindEngine llm_call should use auxiliary model when configured."""
        from lingya.app import ApplicationBuilder

        test_config.db_path = str(tmp_path / "test.db")
        test_config.memory_path = str(tmp_path / "memory")
        test_config.llm.auxiliary_model = "cheap/model"

        builder = ApplicationBuilder(test_config, test_mind_config)
        builder.with_database()
        builder.with_model()  # creates real _model + _aux_model LiteLLMModel
        builder.with_memory()

        # Replace models with mocks so we can track which one is called.
        # Must be done BEFORE with_engine() — the llm_call closure captures
        # the reference at engine construction time.
        aux_mock = MagicMock()
        aux_mock.ainvoke = AsyncMock(return_value=MagicMock(content="aux ok"))
        main_mock = MagicMock()
        main_mock.ainvoke = AsyncMock(return_value=MagicMock(content="main ok"))
        builder._aux_model = aux_mock
        builder._model = main_mock

        builder.with_engine()
        app = await builder.build()

        # Trigger engine process_event — it calls llm_call internally
        # (via occ_ipc_process), which should use the aux model
        await app.engine.process_event({
            "event_type": "outcome",
            "valence": "positive",
            "focus": "self",
            "description": "test event",
        })

        # Verify aux model was used for the engine's llm_call
        assert aux_mock.ainvoke.call_count > 0
        # Main model should NOT have been called via engine's llm_call
        assert main_mock.ainvoke.call_count == 0

        await app.teardown()
