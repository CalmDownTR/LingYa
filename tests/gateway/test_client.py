"""Test GatewayClient — unit tests with mocked TCP layer."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lingya.gateway.client import (
    GatewayClient,
    _build_upgrade_request,
    _parse_handshake_response,
)
from lingya.gateway.protocol import OP_CLOSE, OP_TEXT, _encode_frame


# ── Helpers ─────────────────────────────────────────────────────────


def _make_reader(data: bytes) -> asyncio.StreamReader:
    """Feed bytes into a StreamReader for testing."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def _build_101_response(accept_key: str) -> bytes:
    """Build a valid HTTP 101 response."""
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key}\r\n"
        "\r\n"
    ).encode()


def _make_mock_writer():
    """Create a mock StreamWriter with both sync and async methods.

    StreamWriter has a mix of sync methods (write, close, is_closing) and
    async methods (drain, wait_closed). We use AsyncMock as the base but
    override sync methods with regular MagicMock to avoid "can't be used
    in 'await' expression" errors in strict asyncio mode.
    """
    w = AsyncMock()
    # Sync methods — override with regular MagicMock so they're not coroutines
    w.write = MagicMock()
    w.is_closing = MagicMock(return_value=False)
    w.close = MagicMock()
    # drain and wait_closed stay as AsyncMock (they're called with await)
    return w


# ── _build_upgrade_request tests ────────────────────────────────────


class TestBuildUpgradeRequest:
    def test_produces_valid_http_request(self):
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        request = _build_upgrade_request("localhost", 8765, key)

        lines = request.split("\r\n")

        assert lines[0] == "GET / HTTP/1.1"
        assert "Host: localhost:8765" in lines
        assert "Upgrade: websocket" in lines
        assert "Connection: Upgrade" in lines
        assert f"Sec-WebSocket-Key: {key}" in lines
        assert "Sec-WebSocket-Version: 13" in lines
        assert request.endswith("\r\n\r\n")

    def test_different_host_and_port(self):
        key = "test-key-123"
        request = _build_upgrade_request("192.168.1.1", 9000, key)

        assert "Host: 192.168.1.1:9000" in request
        assert f"Sec-WebSocket-Key: {key}" in request


# ── _parse_handshake_response tests ─────────────────────────────────


@pytest.mark.asyncio
class TestParseHandshakeResponse:
    async def test_valid_101_response(self):
        """Valid 101 response with correct accept key passes."""
        from lingya.gateway.protocol import _generate_accept_key

        ws_key = "dGhlIHNhbXBsZSBub25jZQ=="
        accept = _generate_accept_key(ws_key)
        response = _build_101_response(accept)
        reader = _make_reader(response)

        # Should not raise
        await _parse_handshake_response(reader, ws_key)

    async def test_non_101_response_raises(self):
        """Response without 101 status raises ConnectionError."""
        response = b"HTTP/1.1 400 Bad Request\r\n\r\n"
        reader = _make_reader(response)

        with pytest.raises(ConnectionError, match="101"):
            await _parse_handshake_response(reader, "any-key")

    async def test_wrong_accept_key_raises(self):
        """Mismatched Sec-WebSocket-Accept raises ConnectionError."""
        wrong_accept = "wrong-accept-key-value=="
        response = _build_101_response(wrong_accept)
        reader = _make_reader(response)

        with pytest.raises(ConnectionError, match="Sec-WebSocket-Accept"):
            await _parse_handshake_response(reader, "some-key")

    async def test_500_response_raises(self):
        """Server error response raises ConnectionError."""
        response = b"HTTP/1.1 500 Internal Server Error\r\n\r\n"
        reader = _make_reader(response)

        with pytest.raises(ConnectionError, match="101"):
            await _parse_handshake_response(reader, "any-key")


# ── GatewayClient tests ─────────────────────────────────────────────


class TestGatewayClientInit:
    def test_default_host_and_port(self):
        client = GatewayClient()
        assert client._host == "localhost"
        assert client._port == 8765
        assert not client.is_connected

    def test_custom_host_and_port(self):
        client = GatewayClient(host="192.168.1.1", port=9000)
        assert client._host == "192.168.1.1"
        assert client._port == 9000

    def test_is_connected_initially_false(self):
        client = GatewayClient()
        assert client.is_connected is False


@pytest.mark.asyncio
class TestGatewayClientConnect:
    async def test_connect_handshake_success(self):
        """connect() completes handshake and sets reader/writer."""
        client = GatewayClient()

        mock_reader = asyncio.StreamReader()
        mock_writer = _make_mock_writer()

        # We need to:
        # 1. Mock open_connection to return our reader/writer
        # 2. Mock _parse_handshake_response to succeed (since connect()
        #    generates a random ws_key that we can't predict)
        with patch(
            "lingya.gateway.client.asyncio.open_connection",
            AsyncMock(return_value=(mock_reader, mock_writer)),
        ), patch(
            "lingya.gateway.client._parse_handshake_response",
            AsyncMock(),
        ) as mock_parse:
            await client.connect()

            mock_parse.assert_called_once()
            assert client.is_connected
            assert client._reader is mock_reader
            assert client._writer is mock_writer

    async def test_connect_bad_response_raises(self):
        """connect() raises if handshake fails."""
        client = GatewayClient()

        mock_reader = asyncio.StreamReader()
        mock_writer = _make_mock_writer()

        with patch(
            "lingya.gateway.client.asyncio.open_connection",
            AsyncMock(return_value=(mock_reader, mock_writer)),
        ), patch(
            "lingya.gateway.client._parse_handshake_response",
            AsyncMock(side_effect=ConnectionError("Bad handshake")),
        ):
            with pytest.raises(ConnectionError, match="Bad handshake"):
                await client.connect()

            assert not client.is_connected


@pytest.mark.asyncio
class TestGatewayClientSend:
    async def test_send_without_connect_raises(self):
        """send() raises if not connected."""
        client = GatewayClient()

        with pytest.raises(ConnectionError, match="Not connected"):
            await client.send({"type": "ping", "payload": {}})

    async def test_send_sends_masked_frame_and_reads_response(self):
        """send() sends a masked JSON frame and returns the response."""
        client = GatewayClient()

        mock_reader = asyncio.StreamReader()
        mock_writer = _make_mock_writer()

        client._reader = mock_reader
        client._writer = mock_writer

        # Feed a response frame
        response_msg = {"type": "pong", "payload": {"timestamp": "2025-01-01T00:00:00Z"}}
        response_json = json.dumps(response_msg, ensure_ascii=False).encode("utf-8")
        response_frame = _encode_frame(OP_TEXT, response_json)
        mock_reader.feed_data(response_frame)
        mock_reader.feed_eof()

        result = await client.send({"type": "ping", "payload": {}})

        assert result == response_msg
        # Verify writer was used to send data
        mock_writer.write.assert_called()
        mock_writer.drain.assert_called()

    async def test_send_handles_close_frame_from_server(self):
        """send() raises ConnectionError if server sends close frame."""
        client = GatewayClient()

        mock_reader = asyncio.StreamReader()
        mock_writer = _make_mock_writer()

        client._reader = mock_reader
        client._writer = mock_writer

        # Feed a close frame
        close_frame = _encode_frame(OP_CLOSE, b"")
        mock_reader.feed_data(close_frame)
        mock_reader.feed_eof()

        with pytest.raises(ConnectionError, match="closed"):
            await client.send({"type": "ping", "payload": {}})


@pytest.mark.asyncio
class TestGatewayClientClose:
    async def test_close_sends_close_frame_and_cleans_up(self):
        """close() sends a close frame, closes writer, and clears state."""
        client = GatewayClient()

        mock_writer = _make_mock_writer()

        client._reader = asyncio.StreamReader()
        client._writer = mock_writer

        await client.close()

        # Verify a frame was written (the close frame)
        mock_writer.write.assert_called()
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()

        assert client._reader is None
        assert client._writer is None

    async def test_close_when_not_connected_is_noop(self):
        """close() is a no-op when not connected."""
        client = GatewayClient()

        # Should not raise
        await client.close()

        assert client._reader is None
        assert client._writer is None

    async def test_close_when_already_closing_is_noop(self):
        """close() is a no-op when already closing."""
        client = GatewayClient()

        mock_writer = _make_mock_writer()
        mock_writer.is_closing.return_value = True  # Already closing

        client._writer = mock_writer

        await client.close()

        # Should not attempt to send or close again
        mock_writer.write.assert_not_called()
        mock_writer.close.assert_not_called()
