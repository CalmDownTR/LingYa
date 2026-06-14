"""Test lingya start — daemon launch + WebSocket client attach logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import start_main so we can test its logic paths
from main import start_main


@pytest.mark.asyncio
class TestStartMain:
    async def test_daemon_already_running_skips_launch(self):
        """When daemon is already running, skip subprocess launch."""
        mock_client = AsyncMock()
        mock_client_ctor = MagicMock(return_value=mock_client)

        mock_cli = AsyncMock()
        mock_cli_class = MagicMock(return_value=mock_cli)

        with patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running", return_value=True
        ) as mock_is_running, \
             patch(
            "main.LingYaCLI", mock_cli_class
        ):

            await start_main()

            # Daemon was detected as running — no subprocess spawn
            mock_is_running.assert_called_once()

            # Client was created and connected
            mock_client_ctor.assert_called_once_with(port=8765)
            mock_client.connect.assert_called_once()

            # CLI run was called
            mock_cli.run.assert_called_once()

            # Client was closed after
            mock_client.close.assert_called_once()

    async def test_daemon_not_running_spawns_subprocess(self):
        """When daemon is not running, spawn it as subprocess."""
        mock_client = AsyncMock()
        mock_client_ctor = MagicMock(return_value=mock_client)

        # is_running returns False first, then True after "polling"
        is_running_values = [False, True]

        def is_running_side_effect(*args, **kwargs):
            return is_running_values.pop(0) if is_running_values else True

        mock_popen = MagicMock()
        mock_popen.poll.return_value = None  # Process still running
        mock_cli = AsyncMock()
        mock_cli_class = MagicMock(return_value=mock_cli)

        with patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running",
            side_effect=is_running_side_effect,
        ) as mock_is_running, \
             patch(
            "lingya.gateway.server._find_port_owner", return_value=None
        ), \
             patch(
            "subprocess.Popen", return_value=mock_popen
        ) as mock_popen_class, \
             patch(
            "main.LingYaCLI", mock_cli_class
        ):

            await start_main()

            # Daemon was not running — spawn subprocess
            assert mock_is_running.call_count >= 2

            # Subprocess was spawned (Popen called once for daemon, not for lsof)
            mock_popen_class.assert_called_once()
            call_args = mock_popen_class.call_args[0][0]
            assert "--daemon" in call_args

            # Client connected and CLI ran
            mock_client.connect.assert_called_once()
            mock_cli.run.assert_called_once()
            mock_client.close.assert_called_once()

    async def test_daemon_fails_to_start_times_out(self):
        """When daemon never starts, timeout and return without client connect."""
        mock_client_ctor = MagicMock()
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None  # Process still running, never exits

        with patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running", return_value=False
        ) as mock_is_running, \
             patch(
            "lingya.gateway.server._find_port_owner", return_value=None
        ), \
             patch(
            "subprocess.Popen", return_value=mock_popen
        ) as mock_popen_class, \
             patch(
            "main.LingYaCLI"
        ) as mock_cli_class:

            # Override the sleep to be instant so the test doesn't take 10 seconds
            async def instant_sleep(duration):
                pass

            with patch("asyncio.sleep", instant_sleep):
                await start_main()

            # is_running was checked at least 101 times (1 initial + 100 loop)
            assert mock_is_running.call_count >= 101

            # Subprocess was spawned
            mock_popen_class.assert_called_once()

            # Subprocess was killed on timeout
            mock_popen.kill.assert_called_once()

            # Client was NOT created (start failed)
            mock_client_ctor.assert_not_called()

            # CLI was NOT created
            mock_cli_class.assert_not_called()

    async def test_detects_crashed_subprocess(self):
        """When subprocess dies during startup, read stderr and return early."""
        mock_client_ctor = MagicMock()
        mock_popen = MagicMock()
        mock_popen.poll.return_value = 1  # Process exited
        mock_popen.returncode = 1
        mock_popen.stderr = MagicMock()
        mock_popen.stderr.read.return_value = b"Fake startup error"

        with patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running", return_value=False
        ), \
             patch(
            "lingya.gateway.server._find_port_owner", return_value=None
        ), \
             patch(
            "subprocess.Popen", return_value=mock_popen
        ) as mock_popen_class, \
             patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "main.LingYaCLI"
        ), \
             patch("builtins.print"):

            with patch("asyncio.sleep", AsyncMock()):
                await start_main()

            # Subprocess was spawned
            mock_popen_class.assert_called_once()
            # Client was NOT created (crashed before connection)
            mock_client_ctor.assert_not_called()

    async def test_client_connect_failure_handled(self):
        """When client.connect() fails, print error and return."""
        mock_client = AsyncMock()
        mock_client.connect.side_effect = ConnectionError("Connection refused")
        mock_client_ctor = MagicMock(return_value=mock_client)

        with patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running", return_value=True
        ), \
             patch(
            "main.LingYaCLI"
        ) as mock_cli_class:

            await start_main()

            # Client was created
            mock_client_ctor.assert_called_once()
            # connect was attempted
            mock_client.connect.assert_called_once()
            # CLI was NOT created (connect failed)
            mock_cli_class.assert_not_called()
            # close was still called in finally
            mock_client.close.assert_called_once()

    async def test_cli_run_cleanup_on_exception(self):
        """client.close() is called even if run() raises."""
        mock_client = AsyncMock()
        mock_client_ctor = MagicMock(return_value=mock_client)

        mock_cli = AsyncMock()
        mock_cli.run.side_effect = EOFError()

        with patch(
            "main.LingYaCLI", return_value=mock_cli
        ), \
             patch(
            "lingya.gateway.client.GatewayClient", mock_client_ctor
        ), \
             patch(
            "lingya.gateway.daemon.GatewayDaemon.is_running", return_value=True
        ):

            await start_main()

            # Despite the exception, client.close() was called
            mock_client.close.assert_called_once()
