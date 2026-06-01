"""Shared WebSocket protocol helpers — RFC 6455, zero third-party deps.

Used by both server.py and client.py to avoid duplication.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import struct

# ── RFC 6455 constants ──────────────────────────────────────────────

WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_TEXT = 0x1
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

# ── Key generation ──────────────────────────────────────────────────


def _generate_accept_key(key: str) -> str:
    """Compute Sec-WebSocket-Accept from Sec-WebSocket-Key per RFC 6455."""
    sha1 = hashlib.sha1((key + WS_MAGIC_STRING).encode()).digest()
    return base64.b64encode(sha1).decode()


# ── Frame encoding (server→client, unmasked) ────────────────────────


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


# ── Frame encoding (client→server, masked) ──────────────────────────


def _encode_masked_frame(
    opcode: int, payload: bytes, mask: bytes | None = None
) -> bytes:
    """Encode a client-style masked WebSocket frame.

    If mask is not provided, a random 4-byte mask is generated.
    """
    if mask is None:
        mask = os.urandom(4)

    frame = bytearray()
    frame.append(0x80 | opcode)  # FIN + opcode

    length = len(payload)
    if length < 126:
        frame.append(0x80 | length)  # MASK=1 + length
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack(">Q", length))

    frame.extend(mask)
    # Mask the payload
    masked_payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    frame.extend(masked_payload)

    return bytes(frame)


# ── Frame sending ───────────────────────────────────────────────────


async def _send_frame(
    writer: asyncio.StreamWriter, opcode: int, payload: bytes
) -> None:
    """Send a single WebSocket frame (unmasked, server→client)."""
    frame = _encode_frame(opcode, payload)
    writer.write(frame)
    await writer.drain()


async def _send_masked_frame(
    writer: asyncio.StreamWriter, opcode: int, payload: bytes
) -> None:
    """Send a single WebSocket frame (masked, client→server)."""
    frame = _encode_masked_frame(opcode, payload)
    writer.write(frame)
    await writer.drain()


# ── Frame reading (handles both masked and unmasked) ────────────────


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


# ── HTTP ────────────────────────────────────────────────────────────


async def _read_http_request(reader: asyncio.StreamReader) -> dict:
    """Parse an HTTP request into a dict with method, path, headers."""
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
