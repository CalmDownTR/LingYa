"""Tests for main.py — daemon lifecycle CLI functions.

Covers: start_detached, read_log_tail, and the double-start / stale-PID
guards added in v0.9.6.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ── read_log_tail tests ───────────────────────────────────────────────────


class TestReadLogTail:
    def test_returns_last_n_lines(self, tmp_path):
        """read_log_tail reads last n lines from a log file."""
        from main import read_log_tail

        log_path = tmp_path / "lingya.log"
        log_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = read_log_tail(log_path, n=3)
        assert result == "line 3\nline 4\nline 5\n"

    def test_returns_fewer_when_file_shorter_than_n(self, tmp_path):
        """Returns all lines when file has fewer lines than requested."""
        from main import read_log_tail

        log_path = tmp_path / "lingya.log"
        log_path.write_text("line 1\nline 2\n")

        result = read_log_tail(log_path, n=5)
        assert result == "line 1\nline 2\n"

    def test_returns_empty_string_when_file_missing(self, tmp_path):
        """Returns '' when log file does not exist."""
        from main import read_log_tail

        result = read_log_tail(tmp_path / "nonexistent.log", n=20)
        assert result == ""

    def test_returns_empty_string_when_file_empty(self, tmp_path):
        """Returns '' when log file exists but is empty."""
        from main import read_log_tail

        log_path = tmp_path / "lingya.log"
        log_path.write_text("")

        result = read_log_tail(log_path, n=20)
        assert result == ""


# ── start_detached tests ──────────────────────────────────────────────────

# GatewayDaemon is lazily imported inside start_detached(), so the patch
# target is its definition site: lingya.gateway.daemon.GatewayDaemon
# GatewayDaemon and load_config are lazily imported inside start_detached() —
# patch at their definition sites, not main module.
GW_IS_RUNNING = "lingya.gateway.daemon.GatewayDaemon.is_running"
LOAD_CONFIG = "lingya.config.load_config"


class TestStartDetachedAlreadyRunning:
    def test_exits_when_daemon_already_running(self, tmp_path):
        """When is_running() returns True, prints message and exits 0."""
        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text(str(os.getpid()))

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=True), \
             patch("main._get_pid_from_file", return_value=os.getpid()), \
             patch("builtins.print") as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            from main import start_detached
            start_detached()

        assert exc_info.value.code == 0
        printed = " ".join(str(a[0]) for a in mock_print.call_args_list)
        assert "already running" in printed.lower()


class TestStartDetachedStalePid:
    def test_cleans_stale_pid_file_before_starting(self, tmp_path):
        """When PID file exists but process is dead, cleans it then starts."""
        pid_file = tmp_path / "lingya.pid"
        pid_file.write_text("99999")  # stale PID

        mock_child = MagicMock()
        mock_child.poll.return_value = None  # child still running

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child) as mock_popen, \
             patch("main._get_pid_from_file", return_value=os.getpid()), \
             patch("builtins.print"):
            from main import start_detached
            start_detached()

        # Stale file should be cleaned (is_running→False → unlink before Popen)
        mock_popen.assert_called_once()
        assert not pid_file.exists()


class TestStartDetachedSuccess:
    def test_prints_success_when_pid_file_appears(self, tmp_path):
        """When PID file appears during polling, prints success message."""
        from main import DEFAULT_PORT

        pid_file = tmp_path / "lingya.pid"

        mock_child = MagicMock()
        mock_child.poll.return_value = None  # child still running

        get_pid_calls = [None, 12345]

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child), \
             patch("main._get_pid_from_file", side_effect=get_pid_calls), \
             patch("main.time.sleep", return_value=None), \
             patch("builtins.print") as mock_print:
            from main import start_detached
            start_detached()

        printed = " ".join(str(a[0]) for a in mock_print.call_args_list)
        assert f"PID: {12345}" in printed
        assert f"port: {DEFAULT_PORT}" in printed
        assert "Web UI" in printed


class TestStartDetachedTimeout:
    def test_exits_with_error_when_pid_file_never_appears(self, tmp_path):
        """After 10s polling with no PID file, prints failure and exits 1."""
        pid_file = tmp_path / "lingya.pid"

        mock_child = MagicMock()
        mock_child.poll.return_value = None  # child still running, but no PID

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child), \
             patch("main._get_pid_from_file", return_value=None), \
             patch("main.time.sleep", return_value=None), \
             patch("builtins.print") as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            from main import start_detached
            start_detached()

        assert exc_info.value.code == 1
        printed = " ".join(str(a[0]) for a in mock_print.call_args_list)
        assert "Check logs" in printed


class TestStartDetachedChildCrash:
    def test_reads_log_tail_when_child_dies(self, tmp_path):
        """When child process dies during readiness wait, prints log tail."""
        pid_file = tmp_path / "lingya.pid"
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_path = log_dir / "lingya.log"
        log_path.write_text("ERROR: port already in use\nTraceback...\n")

        mock_child = MagicMock()
        mock_child.poll.return_value = 1  # child died

        mock_cfg = MagicMock()
        mock_cfg.data_dir = str(tmp_path)

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG, return_value=mock_cfg), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child), \
             patch("main._get_pid_from_file", return_value=None), \
             patch("main.time.sleep", return_value=None), \
             patch("builtins.print") as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            from main import start_detached
            start_detached()

        assert exc_info.value.code == 1
        printed = " ".join(str(a[0]) for a in mock_print.call_args_list)
        assert "Failed to start" in printed
        assert "Check logs" in printed


class TestStartDetachedLogDir:
    def test_popen_uses_python_unbuffered_env(self, tmp_path):
        """Popen call sets PYTHONUNBUFFERED=1 in child env."""
        pid_file = tmp_path / "lingya.pid"

        mock_child = MagicMock()
        mock_child.poll.return_value = None

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child) as mock_popen, \
             patch("main._get_pid_from_file", return_value=os.getpid()), \
             patch("builtins.print"):
            from main import start_detached
            start_detached()

        call_kwargs = mock_popen.call_args[1]
        assert "env" in call_kwargs
        assert call_kwargs["env"]["PYTHONUNBUFFERED"] == "1"

    def test_popen_uses_start_new_session(self, tmp_path):
        """Popen call uses start_new_session=True to detach from terminal."""
        pid_file = tmp_path / "lingya.pid"

        mock_child = MagicMock()
        mock_child.poll.return_value = None

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child) as mock_popen, \
             patch("main._get_pid_from_file", return_value=os.getpid()), \
             patch("builtins.print"):
            from main import start_detached
            start_detached()

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["start_new_session"] is True

    def test_popen_stderr_is_stdout(self, tmp_path):
        """Popen redirects stderr to STDOUT (not PIPE) to avoid SIGPIPE."""
        pid_file = tmp_path / "lingya.pid"

        mock_child = MagicMock()
        mock_child.poll.return_value = None

        with patch("main.DEFAULT_PID_FILE", str(pid_file)), \
             patch(GW_IS_RUNNING, return_value=False), \
             patch(LOAD_CONFIG), \
             patch("builtins.open", mock_open()), \
             patch("subprocess.Popen", return_value=mock_child) as mock_popen, \
             patch("main._get_pid_from_file", return_value=os.getpid()), \
             patch("builtins.print"):
            from main import start_detached
            start_detached()

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["stderr"] == subprocess.STDOUT
