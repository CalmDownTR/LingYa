"""Entry point for ``[project.scripts]`` console script.

Usage::

    lingya start          Start daemon in foreground
    lingya start -d       Start daemon in background (detach from terminal)
    lingya stop           Gracefully stop a running daemon
    lingya status         Show daemon status
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main() -> None:
    # Ensure repo root is on sys.path so we can import main.py
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        prog="lingya",
        description="LingYa — an autonomous AI agent with memory and personality",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    start_parser = sub.add_parser("start", help="Start daemon in foreground (Ctrl+C to stop)")
    start_parser.add_argument(
        "-d", "--detach",
        action="store_true",
        help="Start daemon in background (detach from terminal)",
    )
    sub.add_parser("stop", help="Gracefully stop a running daemon")
    sub.add_parser("status", help="Show daemon running status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        if args.detach:
            from main import start_detached

            start_detached()
        else:
            from main import daemon_main

            asyncio.run(daemon_main())

    elif args.command == "stop":
        from main import stop_daemon

        stop_daemon()

    elif args.command == "status":
        from main import status

        status()


if __name__ == "__main__":
    main()
