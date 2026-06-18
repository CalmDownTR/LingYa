"""GatewayServer — pure asyncio WebSocket server, zero third-party deps.

Implements a minimal RFC 6455 WebSocket server:
- HTTP upgrade handshake
- Text frame parsing (opcode 0x1)
- Ping/pong handling (opcode 0x9 -> 0xA)
- Close frame handling (opcode 0x8)
- Client-to-server message masking

Uses only stdlib asyncio. No FastAPI, no websockets lib, no aiohttp.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from typing import Any

from lingya.gateway.protocol import (
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    _generate_accept_key,
    _read_frame,
    _read_http_request,
    _send_frame,
    _send_handshake_response,
)


# ── Server ──────────────────────────────────────────────────────────


class GatewayServer:
    """asyncio WebSocket server for LingYa Gateway.

    Binds to a host:port, accepts WebSocket connections,
    delegates message handling to MessageRouter.
    """

    def __init__(self, host: str, port: int, router: Any) -> None:
        self._host = host
        self._port = port
        self._router = router
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Start the asyncio WebSocket server.

        If the port is already in use by a previous LingYa process (stale PID
        file), the old process is killed and the server retries binding.
        """
        try:
            self._server = await asyncio.start_server(
                self._handle_connection, self._host, self._port
            )
        except OSError as e:
            if e.errno != 48:  # Not "address already in use"
                raise
            old_pid = _find_port_owner(self._port)
            if old_pid is None or old_pid == os.getpid():
                raise
            os.kill(old_pid, signal.SIGTERM)
            await asyncio.sleep(0.5)
            self._server = await asyncio.start_server(
                self._handle_connection, self._host, self._port
            )

    async def stop(self) -> None:
        """Stop the server gracefully."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def is_running(self) -> bool:
        """Whether the server is currently accepting connections."""
        return self._server is not None and self._server.is_serving()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single WebSocket connection lifecycle."""
        try:
            # 1. Handshake
            request = await _read_http_request(reader)
            ws_key = request.get("headers", {}).get("sec-websocket-key", "")
            if not ws_key:
                writer.close()
                return

            accept_key = _generate_accept_key(ws_key)
            await _send_handshake_response(writer, accept_key)

            # 2. Read loop
            while True:
                opcode, payload = await _read_frame(reader)

                if opcode == OP_CLOSE:
                    await _send_frame(writer, OP_CLOSE, b"")
                    break
                elif opcode == OP_PING:
                    await _send_frame(writer, OP_PONG, payload)
                elif opcode == OP_TEXT:
                    t_ws = time.monotonic()
                    try:
                        message = json.loads(payload.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Malformed message — ignore, don't crash the connection
                        continue

                    msg_type = message.get("type", "")

                    if msg_type == "chat":
                        # Streaming path: push event frames as they arrive
                        async def _emit(event_dict: dict) -> None:
                            event_bytes = json.dumps(
                                event_dict, ensure_ascii=False, default=str
                            ).encode("utf-8")
                            await _send_frame(writer, OP_TEXT, event_bytes)

                        response = await self._router._handle_chat(
                            message.get("payload", {}), emit=_emit
                        )
                    else:
                        response = await self._router.route(message)

                    # Inject ws_hop_ms into chat_response meta
                    if response.get("type") == "chat_response":
                        ws_hop_ms = round((time.monotonic() - t_ws) * 1000, 1)
                        response.setdefault("payload", {})["meta"] = {
                            **(response["payload"].get("meta", {})),
                            "ws_hop_ms": ws_hop_ms,
                        }
                    response_bytes = json.dumps(
                        response, ensure_ascii=False, default=str
                    ).encode("utf-8")
                    await _send_frame(writer, OP_TEXT, response_bytes)
                # Ignore other opcodes silently

        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # Client disconnected
        except Exception:
            pass  # Don't crash the server on any single connection error
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


def _find_port_owner(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
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
