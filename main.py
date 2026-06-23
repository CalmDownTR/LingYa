#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from lingya.config import load_config
from lingya.cli import LingYaCLI


async def daemon_main() -> None:
    """Run LingYa in daemon mode — long-running process with GatewayDaemon."""
    config = load_config()

    from lingya.mind import load_mind_config

    mind_config = load_mind_config(config.persona_config_path)

    from lingya.gateway import GatewayDaemon

    daemon = GatewayDaemon(config=config, mind_config=mind_config)

    try:
        await daemon.start()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await daemon.shutdown()


async def start_main() -> None:
    """Auto-launch daemon + attach CLI as WebSocket client.

    1. Check if daemon is already running (via PID file)
    2. If not, launch daemon as a subprocess and wait for WS server ready
    3. Connect CLI via WebSocket to the Gateway
    """
    from lingya.gateway.daemon import GatewayDaemon, _find_port_owner
    from lingya.gateway.client import GatewayClient

    PORT = 8765

    # 1. Check if daemon is already running
    if not GatewayDaemon.is_running():
        # If port is held by a stale process (PID file missing), kill it
        stale_pid = _find_port_owner(PORT)
        if stale_pid is not None:
            import signal
            import os as _os
            _os.kill(stale_pid, signal.SIGTERM)

        import subprocess

        # Launch daemon as a subprocess using the same Python + main.py
        daemon_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for daemon to be ready (poll the PID file).
        # PID file is written AFTER server starts, so is_running() == server ready.
        for _ in range(100):  # 10 second timeout (100 * 0.1s)
            if GatewayDaemon.is_running():
                break
            # If daemon died, read stderr for diagnostics
            if daemon_proc.poll() is not None:
                stderr = daemon_proc.stderr.read().decode()
                print(f"Daemon exited with code {daemon_proc.returncode}:", file=sys.stderr)
                print(stderr, file=sys.stderr)
                return
            await asyncio.sleep(0.1)
        else:
            print("Daemon did not become ready within 10 seconds.", file=sys.stderr)
            daemon_proc.kill()
            return

    # 2. Connect CLI via WebSocket
    client = GatewayClient(port=PORT)
    try:
        await client.connect()

        cli = LingYaCLI(client=client)
        try:
            await cli.run()
        except (KeyboardInterrupt, EOFError):
            pass
    except ConnectionError as e:
        print(f"Failed to connect to Gateway: {e}", file=sys.stderr)
    finally:
        await client.close()


def stop_daemon(pid_file: str = "/tmp/lingya.pid") -> None:
    """Send SIGTERM to a running daemon and wait for it to exit.

    Handles three cases:
    - PID file exists + process alive → SIGTERM, poll for exit (max 10s)
    - PID file exists + process dead → clean stale file, notify
    - No PID file → notify nothing to stop
    """
    import os as _os
    import signal
    import time
    from pathlib import Path

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
        _os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        # Process is already dead — stale PID file
        pid_path.unlink(missing_ok=True)
        print("No running daemon, cleaned stale PID file.")
        return

    # Process is alive — send SIGTERM
    _os.kill(pid, signal.SIGTERM)

    # Poll for exit (max 10s)
    for _ in range(50):  # 50 * 0.2s = 10s
        time.sleep(0.2)
        try:
            _os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            print("LingYa daemon stopped.")
            return

    print("Daemon did not stop within 10 seconds.")


if __name__ == "__main__":
    if "--stop" in sys.argv:
        stop_daemon()
    elif "--daemon" in sys.argv:
        asyncio.run(daemon_main())
    else:
        asyncio.run(start_main())
