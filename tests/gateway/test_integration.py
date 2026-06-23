"""Integration tests for the Gateway daemon lifecycle.

These tests exercise the real daemon.start() path with a real SQLite
database and a real HTTP connection.  Only the LLM is mocked.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── daemon lifecycle (real DB, real WS, mocked LLM) ──────────────────


@pytest.mark.integration
@pytest.mark.asyncio
class TestDaemonLifecycle:
    """Start daemon with real infrastructure, mocked LLM, connect via WS."""

    async def test_startup_ping_shutdown(self, tmp_path, monkeypatch):
        """Full lifecycle: start → ping/pong → shutdown → PID file removed.

        This would have caught the checkpointer.setup() AttributeError
        because _init_agent runs for real against a real SQLite database.
        """
        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )
        from lingya.gateway.client import GatewayClient

        TEST_PORT = 18765
        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        # Fake API key so _init_model doesn't raise KeyError
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="You are a test assistant.",
                core_belief="Test core belief.",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        # Mock ChatOpenAI so _init_model succeeds without an API key.
        # Also mock create_deep_agent + middleware — the agent isn't exercised
        # by this test (we only ping), but _init_agent still runs the real
        # checkpointer setup (AsyncSqliteSaver.from_conn_string), which is
        # what caught the setup() AttributeError.
        mock_model = MagicMock()
        mock_model.profile = {}
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))

        from lingya.gateway.daemon import GatewayDaemon

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        with patch(
            "lingya.app.ChatOpenAI", return_value=mock_model
        ), \
             patch(
            "lingya.app.create_deep_agent", return_value=MagicMock()
        ), \
             patch(
            "lingya.app.create_summarization_tool_middleware",
            return_value=MagicMock(),
        ):
            # Start daemon in background — it blocks on shutdown_event
            daemon_task = asyncio.create_task(daemon.start())

            try:
                # Wait for PID file to appear (daemon writes it AFTER server is ready)
                for _ in range(100):
                    if os.path.exists(pid_file):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Daemon did not write PID file within 5 seconds")

                # Connect via HTTP client — PID file signals server readiness
                client = GatewayClient(port=TEST_PORT)
                await client.connect()
                assert client.is_connected

                # Ping → /health returns {"status": "ok"}
                response = await client.send({"type": "ping", "payload": {}})
                assert response == {"status": "ok"}

                await client.close()
                assert not client.is_connected

            finally:
                # Trigger graceful shutdown — simulates daemon_main() lifecycle
                daemon._shutdown_event.set()
                await asyncio.wait_for(daemon_task, timeout=5.0)
                await daemon.shutdown()

        # PID file must be cleaned up
        assert not os.path.exists(pid_file), "PID file should be removed on shutdown"

    async def test_pid_file_is_readiness_signal(self, tmp_path, monkeypatch):
        """PID file appears only AFTER the WebSocket server is listening.

        This is the integration-level assertion of the ordering bug fix.
        """
        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )
        from lingya.gateway.client import GatewayClient

        TEST_PORT = 18766
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

        config = Config(
            db_path=str(tmp_path / "lingya.db"),
            memory_path=str(tmp_path / "memory"),
            data_dir=str(tmp_path / "data"),
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="You are a test assistant.",
                core_belief="Test core belief.",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        mock_model = MagicMock()
        mock_model.profile = {}
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))

        from lingya.gateway.daemon import GatewayDaemon

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        with patch(
            "lingya.app.ChatOpenAI", return_value=mock_model
        ), \
             patch(
            "lingya.app.create_deep_agent", return_value=MagicMock()
        ), \
             patch(
            "lingya.app.create_summarization_tool_middleware",
            return_value=MagicMock(),
        ):
            daemon_task = asyncio.create_task(daemon.start())

            try:
                # Wait for PID file
                for _ in range(100):
                    if os.path.exists(pid_file):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Daemon did not write PID file")

                # If PID file is the readiness signal, the server must be
                # reachable immediately — no additional wait needed
                client = GatewayClient(port=TEST_PORT)
                await client.connect()
                assert client.is_connected

                response = await client.send({"type": "ping", "payload": {}})
                assert response == {"status": "ok"}

                await client.close()

            finally:
                daemon._shutdown_event.set()
                await asyncio.wait_for(daemon_task, timeout=5.0)
                await daemon.shutdown()


