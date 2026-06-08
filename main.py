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
    from lingya.gateway.daemon import GatewayDaemon
    from lingya.gateway.client import GatewayClient

    PORT = 8765

    # 1. Check if daemon is already running
    if not GatewayDaemon.is_running():
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

        cli = LingYaCLI(ws_client=client)
        try:
            await cli.run()
        except (KeyboardInterrupt, EOFError):
            pass
    except ConnectionError as e:
        print(f"Failed to connect to Gateway: {e}", file=sys.stderr)
    finally:
        await client.close()


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        asyncio.run(daemon_main())
    else:
        asyncio.run(start_main())
