"""GatewayClient — pure asyncio WebSocket client, zero third-party deps.

Connects to the LingYa Gateway via WebSocket and sends/receives JSON
messages. Implements a minimal RFC 6455 client:
- HTTP upgrade request
- Masked text frame sending
- Unmasked text frame receiving
- Ping/pong/close handling
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import Awaitable, Callable

from lingya.gateway.protocol import (
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    _generate_accept_key,
    _read_frame,
    _send_masked_frame,
)


class GatewayClient:
    """Async WebSocket client for connecting to LingYa Gateway."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is currently open."""
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        """Connect to the Gateway via WebSocket.

        1. Open TCP connection
        2. Send HTTP upgrade request
        3. Read handshake response (expect 101)
        4. Store reader/writer
        """
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port
        )

        try:
            # Generate a random WebSocket key and send upgrade request
            ws_key = base64.b64encode(os.urandom(16)).decode()
            request = _build_upgrade_request(self._host, self._port, ws_key)
            self._writer.write(request.encode())
            await self._writer.drain()

            # Read and validate the handshake response
            await _parse_handshake_response(self._reader, ws_key)
        except Exception:
            # Clean up on handshake failure so is_connected reports False
            try:
                self._writer.close()
            except Exception:
                pass
            self._reader = None
            self._writer = None
            raise

    async def send(self, message: dict) -> dict:
        """Send a JSON message and wait for the response.

        Args:
            message: Dict to serialize and send (e.g., {"type": "ping", "payload": {}}).

        Returns:
            The parsed JSON response dict.

        Raises:
            ConnectionError: If not connected or connection is lost.
        """
        if not self.is_connected or self._writer is None or self._reader is None:
            raise ConnectionError("Not connected to Gateway")

        # Send the message as a masked text frame
        payload = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8")
        await _send_masked_frame(self._writer, OP_TEXT, payload)

        # Read response frames until we get a text frame
        # (Skip ping/pong/close — the server may send these between messages)
        while True:
            opcode, data = await _read_frame(self._reader)

            if opcode == OP_TEXT:
                return json.loads(data.decode("utf-8"))
            elif opcode == OP_CLOSE:
                raise ConnectionError("Gateway closed the connection")
            elif opcode == OP_PING:
                # Respond with pong
                await _send_masked_frame(self._writer, OP_PONG, data)
            # PONG and other opcodes are silently ignored

    async def send_stream(
        self,
        message: dict,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """Send a message and receive streaming responses.

        When *on_event* is provided, each ``{"type": "event", ...}`` frame
        received before the final response is passed to *on_event* immediately.
        The final ``{"type": "chat_response", ...}`` (or error) frame is
        returned.

        When *on_event* is None, behaves like ``send()`` — waits for the
        first non-event text frame and returns it.

        Raises:
            ConnectionError: If not connected or connection is lost.
        """
        if not self.is_connected or self._writer is None or self._reader is None:
            raise ConnectionError("Not connected to Gateway")

        # Send the message as a masked text frame
        payload = json.dumps(message, ensure_ascii=False, default=str).encode("utf-8")
        await _send_masked_frame(self._writer, OP_TEXT, payload)

        # Read response frames — event frames go to callback, final frame returned
        while True:
            opcode, data = await _read_frame(self._reader)

            if opcode == OP_TEXT:
                msg = json.loads(data.decode("utf-8"))
                if msg.get("type") == "event":
                    if on_event is not None:
                        await on_event(msg)
                    continue
                return msg
            elif opcode == OP_CLOSE:
                raise ConnectionError("Gateway closed the connection")
            elif opcode == OP_PING:
                # Respond with pong
                await _send_masked_frame(self._writer, OP_PONG, data)
            # PONG and other opcodes are silently ignored

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        if self._writer is not None and not self._writer.is_closing():
            try:
                await _send_masked_frame(self._writer, OP_CLOSE, b"")
            except Exception:
                pass
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None


# ── Client-specific HTTP helpers ────────────────────────────────────


def _build_upgrade_request(host: str, port: int, ws_key: str) -> str:
    """Build the HTTP upgrade request for WebSocket handshake."""
    return (
        f"GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {ws_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )


async def _parse_handshake_response(
    reader: asyncio.StreamReader, expected_key: str
) -> None:
    """Read and validate the HTTP 101 handshake response.

    Raises:
        ConnectionError: If the response is not 101 or the accept key doesn't match.
    """
    # Read status line
    status_line = await reader.readline()
    status_line = status_line.decode("utf-8", errors="replace").strip()

    if "101" not in status_line:
        raise ConnectionError(
            f"Expected 101 Switching Protocols, got: {status_line}"
        )

    # Read headers
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    # Verify the accept key
    expected_accept = _generate_accept_key(expected_key)
    actual_accept = headers.get("sec-websocket-accept", "")
    if actual_accept != expected_accept:
        raise ConnectionError(
            f"Sec-WebSocket-Accept mismatch: "
            f"expected {expected_accept}, got {actual_accept}"
        )
