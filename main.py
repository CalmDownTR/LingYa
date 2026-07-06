#!/usr/bin/env python3
"""LingYa — an autonomous AI agent with memory and evolving personality.

Usage:
  python main.py              Start daemon in foreground (Ctrl+C to stop)
  python main.py --stop       Gracefully stop a running daemon
  python main.py --status     Show daemon status (running / not running / ...)
  python main.py --diary      Show the latest diary entry
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_PORT = 8765
DEFAULT_PID_FILE = "/tmp/lingya.pid"


# ── Daemon ──────────────────────────────────────────────────────────────


async def daemon_main() -> None:
    """Run LingYa in daemon mode — long-running process with GatewayDaemon."""
    from lingya.config import load_config
    from lingya.mind import load_mind_config

    config = load_config()
    mind_config = load_mind_config(config.persona_config_path)

    from lingya.gateway import GatewayDaemon

    daemon = GatewayDaemon(config=config, mind_config=mind_config)

    try:
        await daemon.start()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await daemon.shutdown()


# ── Stop ────────────────────────────────────────────────────────────────


def stop_daemon(pid_file: str = DEFAULT_PID_FILE) -> None:
    """Send SIGTERM to a running daemon and wait for it to exit.

    Handles three cases:
    - PID file exists + process alive → SIGTERM, poll for exit (max 10s)
    - PID file exists + process dead → clean stale file, notify
    - No PID file → notify nothing to stop
    """
    pid_path = Path(pid_file)

    if not pid_path.exists():
        print("LingYa is not running.")
        return

    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        print("LingYa is not running.")
        pid_path.unlink(missing_ok=True)
        return

    # Check if process is alive
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        # Process is already dead — stale PID file
        pid_path.unlink(missing_ok=True)
        print("No running daemon, cleaned stale PID file.")
        return

    # Process is alive — send SIGTERM
    os.kill(pid, signal.SIGTERM)

    # Poll for exit (max 10s)
    for _ in range(50):  # 50 * 0.2s = 10s
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            print("LingYa daemon stopped.")
            return

    print("Daemon did not stop within 10 seconds.")


# ── Status ──────────────────────────────────────────────────────────────


def _get_pid_from_file(pid_file: str) -> int | None:
    """Read PID from file. Returns None if missing or unreadable."""
    pid_path = Path(pid_file)
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _process_alive(pid: int) -> bool:
    """Check whether a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _find_port_owner(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _pid_file_mtime(pid_file: str) -> float | None:
    """Return mtime of PID file, or None if missing."""
    try:
        return Path(pid_file).stat().st_mtime
    except OSError:
        return None


def status(pid_file: str = DEFAULT_PID_FILE, port: int = DEFAULT_PORT) -> None:
    """Display daemon status — pure local check, no HTTP call.

    Five states:

    ============ ======== ========== ===========
    State        PID file Process   Port owner
    ============ ======== ========== ===========
    Running      ✅       ✅        LingYa
    Not running  ❌       —         —
    Stale PID    ✅       ❌        —
    Port conflict ✅      ✅        Other
    Orphan port  ❌       —         Other
    ============ ======== ========== ===========
    """
    pid = _get_pid_from_file(pid_file)
    port_owner = _find_port_owner(port)

    if pid is not None and _process_alive(pid):
        # PID file exists + process is alive
        if port_owner is not None and port_owner != pid:
            # Port is held by another process
            print(f"⚠ Port {port} is in use by another process (PID {port_owner}).")
            print(f"  LingYa PID {pid} is alive but may not be serving requests.")
            print(f"  Release the port: kill {port_owner}")
            return

        # All good — running normally
        mtime = _pid_file_mtime(pid_file)
        uptime_str = ""
        if mtime is not None:
            uptime_seconds = time.time() - mtime
            hours, remainder = divmod(int(uptime_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = f"{minutes}m {seconds}s"

        # Try to read data dir from the process (fallback: show PID file path)
        data_dir = "~/.lingya"
        try:
            import subprocess

            result = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                capture_output=True, text=True, timeout=3,
            )
            if result.stdout.strip():
                data_dir = "(see config.yaml)"
        except Exception:
            pass

        print(f"LingYa is running")
        print(f"  PID:     {pid}")
        print(f"  Port:    {port}")
        print(f"  Uptime:  {uptime_str}")
        print(f"  Web UI:  http://localhost:{port}")
        print(f"  Data:    {data_dir}")
        return

    if pid is not None:
        # PID file exists but process is dead — stale PID file
        print(f"LingYa is not running (stale PID file: {pid}).")
        print(f"  Clean up with: python main.py --stop")
        return

    # No PID file
    if port_owner is not None:
        print(f"⚠ Port {port} is in use by another process (PID {port_owner}).")
        print(f"  LingYa is not running but the port is occupied.")
        print(f"  Release the port: kill {port_owner}")
        return

    print("LingYa is not running.")
    print(f"  Start with: python main.py")
    print(f"  Web UI will be at: http://localhost:{port}")


# ── Diary ───────────────────────────────────────────────────────────────


def diary(port: int = DEFAULT_PORT) -> None:
    """Fetch and display the latest diary entry from the daemon."""
    import json

    import httpx

    api_key = os.environ.get("LINGYA_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(
            f"http://localhost:{port}/diary",
            params={"action": "read", "index": 0},
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("type") == "diary_response":
            payload = data.get("payload", {})
            entries = payload.get("entries", [])
            if not entries:
                print("No diary entries yet.")
                return
            entry = entries[0]
            print(f"Diary — {entry.get('date', 'unknown date')}")
            print("-" * 40)
            print(entry.get("content", "(empty)"))
        else:
            print(f"Unexpected response: {json.dumps(data, indent=2)}")
    except httpx.ConnectError:
        print("Cannot connect to LingYa daemon. Is it running?")
        print(f"  Check: python main.py --status")
    except httpx.HTTPStatusError as e:
        print(f"Error fetching diary: HTTP {e.response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


# ── CLI entry point ─────────────────────────────────────────────────────


if __name__ == "__main__":
    if "--stop" in sys.argv:
        stop_daemon()
    elif "--status" in sys.argv:
        status()
    elif "--diary" in sys.argv:
        diary()
    else:
        asyncio.run(daemon_main())
