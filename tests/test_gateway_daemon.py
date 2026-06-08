from __future__ import annotations

import asyncio
import os
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
def config():
    return Config()


@pytest.fixture
def mind_config():
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


@pytest.fixture
def daemon(config, mind_config, tmp_path):
    """Create a GatewayDaemon with a temporary PID file path."""
    from lingya.gateway.daemon import GatewayDaemon

    pid_file = str(tmp_path / "lingya.pid")
    return GatewayDaemon(config=config, mind_config=mind_config, pid_file=pid_file)


# ── Initialization tests ──────────────────────────────────────────────


class TestGatewayDaemonInit:
    def test_init_stores_configs(self, config, mind_config, tmp_path):
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = str(tmp_path / "lingya.pid")
        d = GatewayDaemon(config=config, mind_config=mind_config, pid_file=pid_file)

        assert d.config is config
        assert d.mind_config is mind_config
        assert d.pid_file == pid_file
        assert d._engine is None
        assert d._db is None
        assert d._model is None
        assert d._memory is None
        assert d._agent is None
        assert d._checkpointer is None

    def test_init_default_pid_file(self, config, mind_config):
        from lingya.gateway.daemon import GatewayDaemon

        d = GatewayDaemon(config=config, mind_config=mind_config)
        assert d.pid_file == "/tmp/lingya.pid"


# ── is_running tests ──────────────────────────────────────────────────


class TestIsRunning:
    def test_returns_false_when_no_pid_file(self, tmp_path):
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = str(tmp_path / "nonexistent.pid")
        assert GatewayDaemon.is_running(pid_file) is False

    def test_returns_false_when_stale_pid(self, tmp_path):
        """PID file exists but process is dead."""
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = tmp_path / "lingya.pid"
        # Use a PID that almost certainly doesn't exist.
        # On most systems, PIDs are positive; 99999 is safe for a test.
        pid_file.write_text("99999")
        assert GatewayDaemon.is_running(str(pid_file)) is False

    def test_returns_true_when_process_alive(self, tmp_path):
        """PID file exists and process is alive (current process)."""
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text(str(os.getpid()))
        assert GatewayDaemon.is_running(str(pid_file)) is True

    def test_returns_false_when_pid_file_has_invalid_content(self, tmp_path):
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text("not-a-pid")
        assert GatewayDaemon.is_running(str(pid_file)) is False

    def test_returns_false_when_pid_file_empty(self, tmp_path):
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text("")
        assert GatewayDaemon.is_running(str(pid_file)) is False


# ── PID file lifecycle tests ──────────────────────────────────────────


@pytest.mark.asyncio
class TestPidFileLifecycle:
    async def test_pid_file_written_after_server_ready(self, daemon, tmp_path):
        """PID file is written only after WebSocket server is listening."""
        mock_ws_server = AsyncMock()
        mock_ws_server.start = AsyncMock()

        # Track whether server.start() was called before _write_pid_file
        events: list[str] = []

        original_write_pid = daemon._write_pid_file
        def tracking_write_pid():
            events.append("pid")
            original_write_pid()

        async def tracking_server_start():
            events.append("server_start")
            # Also trigger shutdown so start() unblocks quickly
            daemon._shutdown_event.set()

        mock_ws_server.start = tracking_server_start
        daemon._write_pid_file = tracking_write_pid

        with patch.object(daemon, "_init_database", AsyncMock()), \
             patch.object(daemon, "_init_model", MagicMock()), \
             patch.object(daemon, "_init_memory", MagicMock()), \
             patch.object(daemon, "_init_engine", AsyncMock()), \
             patch.object(daemon, "_init_agent", AsyncMock()), \
             patch.object(daemon, "_register_signal_handlers", MagicMock()), \
             patch("lingya.gateway.daemon.GatewayServer", return_value=mock_ws_server), \
             patch("builtins.print"):

            await daemon.start()

            # Server started BEFORE PID file was written
            assert events == ["server_start", "pid"], \
                f"Expected ['server_start', 'pid'] but got {events}"

    async def test_pid_file_removed_on_shutdown(self, daemon, tmp_path):
        """shutdown() removes the PID file."""
        # Create PID file to be cleaned up
        Path(daemon.pid_file).write_text(str(os.getpid()))
        assert os.path.exists(daemon.pid_file)

        # Set up mock engine and db for shutdown
        daemon._engine = AsyncMock()
        daemon._db = AsyncMock()

        with patch("builtins.print"):
            await daemon.shutdown()

        assert not os.path.exists(daemon.pid_file)

    async def test_shutdown_saves_mind_state_to_db(self, daemon, tmp_path):
        """shutdown() persists mind state via engine.save_state(db)."""
        mock_engine = AsyncMock()
        mock_db = AsyncMock()
        daemon._engine = mock_engine
        daemon._db = mock_db

        # Create PID file so removal can be verified
        Path(daemon.pid_file).write_text(str(os.getpid()))

        with patch("builtins.print"):
            await daemon.shutdown()

        mock_engine.save_state.assert_called_once_with(mock_db)
        mock_db.close.assert_called_once()

    async def test_shutdown_handles_missing_pid_file_gracefully(self, daemon):
        """shutdown() should not crash if PID file is already gone."""
        daemon._engine = AsyncMock()
        daemon._db = AsyncMock()

        # No PID file exists
        assert not os.path.exists(daemon.pid_file)

        with patch("builtins.print"):
            await daemon.shutdown()

        # Should complete without error
        daemon._engine.save_state.assert_called_once()

    async def test_shutdown_handles_no_engine_gracefully(self, daemon):
        """shutdown() should not crash if engine was never initialized."""
        # engine and db are None (initial state)
        assert daemon._engine is None

        Path(daemon.pid_file).write_text(str(os.getpid()))

        with patch("builtins.print"):
            await daemon.shutdown()

        # Should complete without error, PID file removed
        assert not os.path.exists(daemon.pid_file)

    async def test_start_full_sequence_order(self, daemon):
        """Verify start() calls init methods and server in correct order."""
        call_order: list[str] = []

        async def mock_init_db():
            call_order.append("db")

        def mock_init_model():
            call_order.append("model")

        def mock_init_memory():
            call_order.append("memory")

        async def mock_init_engine():
            call_order.append("engine")

        async def mock_init_agent():
            call_order.append("agent")

        def mock_register_signals():
            call_order.append("signals")

        mock_ws_server = AsyncMock()
        mock_ws_server.start = AsyncMock()

        # Track server start and PID write
        original_write_pid = daemon._write_pid_file
        def tracking_write_pid():
            call_order.append("pid")
            original_write_pid()

        daemon._write_pid_file = tracking_write_pid

        async def tracking_server_start():
            call_order.append("server_start")
            daemon._shutdown_event.set()

        mock_ws_server.start = tracking_server_start

        with patch.object(daemon, "_init_database", mock_init_db), \
             patch.object(daemon, "_init_model", mock_init_model), \
             patch.object(daemon, "_init_memory", mock_init_memory), \
             patch.object(daemon, "_init_engine", mock_init_engine), \
             patch.object(daemon, "_init_agent", mock_init_agent), \
             patch.object(daemon, "_register_signal_handlers", mock_register_signals), \
             patch("lingya.gateway.daemon.GatewayServer", return_value=mock_ws_server), \
             patch("builtins.print"):

            await daemon.start()

        # Init sequence: db -> model -> memory -> engine -> agent -> signals -> server_start -> pid
        assert call_order == [
            "db", "model", "memory", "engine", "agent", "signals",
            "server_start", "pid",
        ]
