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
import base64
import hashlib
import json
import struct
from typing import Any

# ── RFC 6455 constants ──────────────────────────────────────────────

WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_TEXT = 0x1
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

# ── Protocol helpers ────────────────────────────────────────────────


def _generate_accept_key(key: str) -> str:
    """Compute Sec-WebSocket-Accept from Sec-WebSocket-Key per RFC 6455."""
    sha1 = hashlib.sha1((key + WS_MAGIC_STRING).encode()).digest()
    return base64.b64encode(sha1).decode()


async def _read_http_request(reader: asyncio.StreamReader) -> dict:
    """Parse an HTTP upgrade request into a dict with method, path, headers."""
    # Request line
    request_line = await reader.readline()
    request_line = request_line.decode("utf-8", errors="replace").strip()
    if not request_line:
        return {"method": "", "path": "", "headers": {}}

    parts = request_line.split(" ")
    method = parts[0] if len(parts) > 0 else ""
    path = parts[1] if len(parts) > 1 else ""

    # Headers
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    return {"method": method, "path": path, "headers": headers}


async def _send_handshake_response(
    writer: asyncio.StreamWriter, accept_key: str
) -> None:
    """Send HTTP 101 Switching Protocols response."""
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key}\r\n"
        "\r\n"
    )
    writer.write(response.encode())
    await writer.drain()


def _encode_frame(opcode: int, payload: bytes) -> bytes:
    """Encode a WebSocket frame (server->client, unmasked).

    Returns the complete frame as bytes.
    """
    frame = bytearray()
    frame.append(0x80 | opcode)  # FIN=1 + opcode

    length = len(payload)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))

    frame.extend(payload)
    return bytes(frame)


async def _send_frame(
    writer: asyncio.StreamWriter, opcode: int, payload: bytes
) -> None:
    """Send a single WebSocket frame."""
    frame = _encode_frame(opcode, payload)
    writer.write(frame)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read a single WebSocket frame. Returns (opcode, payload).

    Handles extended payload lengths and client masking.
    """
    header = await reader.readexactly(2)
    byte0, byte1 = header[0], header[1]

    opcode = byte0 & 0x0F
    masked = (byte1 & 0x80) != 0
    payload_len = byte1 & 0x7F

    # Extended payload length (16 or 64 bit)
    if payload_len == 126:
        ext = await reader.readexactly(2)
        payload_len = struct.unpack(">H", ext)[0]
    elif payload_len == 127:
        ext = await reader.readexactly(8)
        payload_len = struct.unpack(">Q", ext)[0]

    # Masking key (4 bytes, present for client->server frames)
    mask = await reader.readexactly(4) if masked else b""

    # Payload
    payload = await reader.readexactly(payload_len)

    # Unmask
    if masked:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    return opcode, payload


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
        """Start the asyncio WebSocket server."""
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
                    try:
                        message = json.loads(payload.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Malformed message — ignore, don't crash the connection
                        continue
                    response = await self._router.route(message)
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
