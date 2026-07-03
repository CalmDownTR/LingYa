"""Tests for v0.8.3 auto-observability — Traceloop replaces manual OTel scaffolding.

Verifies the old scaffolding is fully removed:
- ApplicationBuilder no longer has with_observability()
- Application no longer has tracer attribute
- MindEngine no longer accepts tracer parameter
- BackgroundRunner no longer accepts tracer parameter

Traceloop.init() integration tests are in test_daemon.py (require SDK).
"""

from __future__ import annotations

import pytest


class TestWithObservabilityRemoved:
    """ApplicationBuilder.with_observability() must be gone."""

    def test_application_builder_has_no_with_observability(self):
        """ApplicationBuilder should not have with_observability method."""
        from lingya.app import ApplicationBuilder
        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        config = Config()
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test", created_at="2025-01-01"),
            identity=IdentityAnchor(identity="test", core_belief="test"),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )
        builder = ApplicationBuilder(config, mind_config)
        assert not hasattr(builder, "with_observability"), (
            "with_observability() should be removed — Traceloop.init() "
            "handles instrumentation automatically"
        )

    def test_application_has_no_tracer_attribute(self):
        """Application dataclass should not have tracer field."""
        from lingya.app import Application
        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        config = Config()
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test", created_at="2025-01-01"),
            identity=IdentityAnchor(identity="test", core_belief="test"),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        app = Application(
            config=config,
            mind_config=mind_config,
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
        assert not hasattr(app, "tracer"), (
            "Application.tracer should be removed — Traceloop handles all tracing"
        )


class TestMindEngineNoTracerParam:
    """MindEngine.__init__ no longer accepts tracer parameter."""

    def test_engine_init_rejects_tracer_kwarg(self):
        """Passing tracer= to MindEngine should raise TypeError."""
        from unittest.mock import AsyncMock, MagicMock

        from lingya.mind import MindEngine

        with pytest.raises(TypeError, match="tracer"):
            MindEngine(
                config=MagicMock(),
                memory_store=MagicMock(),
                llm_call=AsyncMock(),
                tracer="should-reject",
            )


class TestBackgroundRunnerNoTracerParam:
    """BackgroundRunner.__init__ no longer accepts tracer parameter."""

    def test_background_runner_rejects_tracer_kwarg(self):
        """Passing tracer= to BackgroundRunner should raise TypeError."""
        from unittest.mock import MagicMock

        from lingya.gateway.background import BackgroundRunner

        with pytest.raises(TypeError, match="tracer"):
            BackgroundRunner(
                engine=MagicMock(),
                model=MagicMock(),
                data_dir="/tmp",
                tracer="should-reject",
            )
