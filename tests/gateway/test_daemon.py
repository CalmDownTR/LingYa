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
    from lingya.gateway.daemon import GatewayDaemon

    pid_file = str(tmp_path / "lingya.pid")
    return GatewayDaemon(config=config, mind_config=mind_config, pid_file=pid_file)


def _make_mock_app():
    """Create a mock Application with engine, db, model, memory, and teardown."""
    app = MagicMock()
    app.engine = MagicMock()
    app.engine.save_state = AsyncMock()
    app.db = MagicMock()
    app.db.close = AsyncMock()
    app.model = MagicMock()
    app.memory = MagicMock()
    app.teardown = AsyncMock()
    return app


# ── Initialization tests ──────────────────────────────────────────────


class TestGatewayDaemonInit:
    def test_init_stores_configs(self, config, mind_config, tmp_path):
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = str(tmp_path / "lingya.pid")
        d = GatewayDaemon(config=config, mind_config=mind_config, pid_file=pid_file)

        assert d.config is config
        assert d.mind_config is mind_config
        assert d.pid_file == pid_file
        assert d._app is None

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
        from lingya.gateway.daemon import GatewayDaemon

        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text("99999")
        assert GatewayDaemon.is_running(str(pid_file)) is False

    def test_returns_true_when_process_alive(self, tmp_path):
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
        """PID file is written only after HTTP server is listening."""
        mock_app = _make_mock_app()
        mock_uvicorn_server = MagicMock()
        mock_uvicorn_server.should_exit = False
        mock_uvicorn_server.serve = AsyncMock()

        events: list[str] = []

        original_write_pid = daemon._write_pid_file
        def tracking_write_pid():
            events.append("pid")
            original_write_pid()

        async def tracking_server_serve():
            # Record that server started, then block until should_exit (like real uvicorn)
            events.append("server_start")
            while not mock_uvicorn_server.should_exit:
                await asyncio.sleep(0.01)

        mock_uvicorn_server.serve = tracking_server_serve
        daemon._write_pid_file = tracking_write_pid

        with patch("lingya.app.ApplicationBuilder") as MockBuilder, \
             patch("lingya.gateway.router.MessageRouter"), \
             patch("lingya.gateway.server.create_app"), \
             patch("uvicorn.Server", return_value=mock_uvicorn_server), \
             patch("uvicorn.Config"), \
             patch("builtins.print"):
            builder = MockBuilder.return_value
            builder.with_database.return_value = builder
            builder.with_model.return_value = builder
            builder.with_memory.return_value = builder
            builder.with_event_bus.return_value = builder
            builder.with_engine.return_value = builder
            builder.with_agent.return_value = builder
            builder.build = AsyncMock(return_value=mock_app)

            # Run daemon.start() in background, wait for it to pass startup,
            # then trigger shutdown
            daemon_task = asyncio.create_task(daemon.start())

            # Give it time to complete startup (PID file written)
            await asyncio.sleep(0.5)
            assert events == ["server_start", "pid"], \
                f"Expected ['server_start', 'pid'] but got {events}"

            # Trigger shutdown
            daemon._shutdown_event.set()
            await asyncio.wait_for(daemon_task, timeout=2.0)

    async def test_pid_file_removed_on_shutdown(self, daemon, tmp_path):
        """shutdown() removes the PID file."""
        Path(daemon.pid_file).write_text(str(os.getpid()))
        assert os.path.exists(daemon.pid_file)

        mock_app = _make_mock_app()
        daemon._app = mock_app

        with patch("builtins.print"):
            await daemon.shutdown()

        assert not os.path.exists(daemon.pid_file)

    async def test_shutdown_saves_mind_state_to_db(self, daemon, tmp_path):
        """shutdown() persists mind state via engine.save_state(db)."""
        mock_app = _make_mock_app()
        daemon._app = mock_app

        Path(daemon.pid_file).write_text(str(os.getpid()))

        with patch("builtins.print"):
            await daemon.shutdown()

        mock_app.engine.save_state.assert_called_once_with(mock_app.db)
        mock_app.teardown.assert_called_once()

    async def test_shutdown_handles_missing_pid_file_gracefully(self, daemon):
        """shutdown() should not crash if PID file is already gone."""
        mock_app = _make_mock_app()
        daemon._app = mock_app

        assert not os.path.exists(daemon.pid_file)

        with patch("builtins.print"):
            await daemon.shutdown()

        mock_app.engine.save_state.assert_called_once()

    async def test_shutdown_handles_no_app_gracefully(self, daemon):
        """shutdown() should not crash if app was never built."""
        assert daemon._app is None

        Path(daemon.pid_file).write_text(str(os.getpid()))

        with patch("builtins.print"):
            await daemon.shutdown()

        assert not os.path.exists(daemon.pid_file)

    async def test_start_full_sequence_uses_builder(self, daemon):
        """Verify start() uses ApplicationBuilder and starts uvicorn server."""
        mock_app = _make_mock_app()
        mock_uvicorn_server = MagicMock()
        mock_uvicorn_server.should_exit = False
        mock_uvicorn_server.serve = AsyncMock()

        async def tracking_server_serve():
            while not mock_uvicorn_server.should_exit:
                await asyncio.sleep(0.01)

        mock_uvicorn_server.serve = tracking_server_serve

        with patch("lingya.app.ApplicationBuilder") as MockBuilder, \
             patch("lingya.gateway.router.MessageRouter"), \
             patch("lingya.gateway.server.create_app"), \
             patch("uvicorn.Server", return_value=mock_uvicorn_server), \
             patch("uvicorn.Config"), \
             patch("builtins.print"):
            builder = MockBuilder.return_value
            builder.with_database.return_value = builder
            builder.with_model.return_value = builder
            builder.with_memory.return_value = builder
            builder.with_event_bus.return_value = builder
            builder.with_engine.return_value = builder
            builder.with_agent.return_value = builder
            builder.build = AsyncMock(return_value=mock_app)

            daemon_task = asyncio.create_task(daemon.start())
            await asyncio.sleep(0.5)

            # Builder was used
            builder.with_database.assert_called_once()
            builder.with_model.assert_called_once()
            builder.with_memory.assert_called_once()
            builder.with_event_bus.assert_called_once()
            builder.with_engine.assert_called_once()
            builder.with_agent.assert_called_once()
            builder.build.assert_awaited_once()

            # Trigger shutdown
            daemon._shutdown_event.set()
            await asyncio.wait_for(daemon_task, timeout=2.0)

    async def test_start_passes_shutdown_callback_to_create_app(self, daemon):
        """start() passes daemon._shutdown_event.set as shutdown_callback to create_app."""
        mock_app = _make_mock_app()
        mock_uvicorn_server = MagicMock()
        mock_uvicorn_server.should_exit = False
        mock_uvicorn_server.serve = AsyncMock()

        async def tracking_server_serve():
            while not mock_uvicorn_server.should_exit:
                await asyncio.sleep(0.01)

        mock_uvicorn_server.serve = tracking_server_serve
        mock_create_app = MagicMock()

        with patch("lingya.app.ApplicationBuilder") as MockBuilder, \
             patch("lingya.gateway.router.MessageRouter"), \
             patch("lingya.gateway.server.create_app", mock_create_app), \
             patch("uvicorn.Server", return_value=mock_uvicorn_server), \
             patch("uvicorn.Config"), \
             patch("builtins.print"):
            builder = MockBuilder.return_value
            builder.with_database.return_value = builder
            builder.with_model.return_value = builder
            builder.with_memory.return_value = builder
            builder.with_event_bus.return_value = builder
            builder.with_engine.return_value = builder
            builder.with_agent.return_value = builder
            builder.build = AsyncMock(return_value=mock_app)

            daemon_task = asyncio.create_task(daemon.start())
            await asyncio.sleep(0.5)

            # Verify create_app was called with shutdown_callback
            mock_create_app.assert_called_once()
            call_kwargs = mock_create_app.call_args[1]
            assert "shutdown_callback" in call_kwargs
            assert call_kwargs["shutdown_callback"] is not None

            daemon._shutdown_event.set()
            await asyncio.wait_for(daemon_task, timeout=2.0)


# ── Signal handler tests ─────────────────────────────────────────────


class TestSignalHandlers:
    def test_sighup_in_signal_registration(self, daemon):
        """Verify signal.SIGHUP is registered alongside SIGTERM and SIGINT."""
        import signal

        mock_loop = MagicMock()
        registered_signals: list[int] = []

        def capture_add_signal_handler(sig, callback):
            registered_signals.append(sig)

        mock_loop.add_signal_handler = capture_add_signal_handler

        mock_app = _make_mock_app()
        # Set shutdown event before running so start() doesn't block forever
        daemon._shutdown_event.set()

        mock_uvicorn = MagicMock()
        mock_uvicorn.should_exit = False

        async def tracking_server_serve():
            """Block until should_exit, just like real uvicorn."""
            while not mock_uvicorn.should_exit:
                await asyncio.sleep(0.01)

        mock_uvicorn.serve = tracking_server_serve

        mock_bg_runner = MagicMock()
        mock_bg_runner.start = AsyncMock()
        mock_bg_runner.stop = AsyncMock()

        with patch("asyncio.get_event_loop", return_value=mock_loop), \
             patch("lingya.app.ApplicationBuilder") as MockBuilder, \
             patch("lingya.gateway.router.MessageRouter"), \
             patch("lingya.gateway.server.create_app"), \
             patch("lingya.gateway.daemon.uvicorn.Server", return_value=mock_uvicorn), \
             patch("lingya.gateway.daemon.uvicorn.Config"), \
             patch("lingya.gateway.daemon.BackgroundRunner", return_value=mock_bg_runner), \
             patch("builtins.print"):
            builder = MockBuilder.return_value
            builder.with_database.return_value = builder
            builder.with_model.return_value = builder
            builder.with_memory.return_value = builder
            builder.with_event_bus.return_value = builder
            builder.with_engine.return_value = builder
            builder.with_agent.return_value = builder
            builder.build = AsyncMock(return_value=mock_app)

            async def _run():
                await daemon.start()

            asyncio.run(_run())

        assert signal.SIGHUP in registered_signals, (
            f"Expected SIGHUP ({signal.SIGHUP}) in registered signals, "
            f"got {registered_signals}"
        )
        assert signal.SIGTERM in registered_signals
        assert signal.SIGINT in registered_signals
