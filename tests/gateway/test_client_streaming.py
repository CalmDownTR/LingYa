"""Test GatewayClient.send_stream() — streaming responses with event frames."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lingya.gateway.client import GatewayClient
from lingya.gateway.protocol import OP_CLOSE, OP_PING, OP_TEXT, _encode_frame


# ── Helpers ─────────────────────────────────────────────────────────


def _make_mock_writer():
    """Create a mock StreamWriter."""
    w = AsyncMock()
    w.write = MagicMock()
    w.is_closing = MagicMock(return_value=False)
    w.close = MagicMock()
    return w


def _encode_json_frame(data: dict) -> bytes:
    """Encode a JSON dict as a WebSocket text frame."""
    payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    return _encode_frame(OP_TEXT, payload)


@pytest.fixture
def connected_client():
    """Return a GatewayClient with mock reader/writer, marked connected."""
    client = GatewayClient()
    mock_reader = asyncio.StreamReader()
    mock_writer = _make_mock_writer()
    client._reader = mock_reader
    client._writer = mock_writer
    return client


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGatewayClientSendStream:
    async def test_send_stream_yields_events_to_callback(self, connected_client):
        """on_event callback receives all event frames before final response."""
        events = [
            _encode_json_frame({"type": "event", "event": "process.phase", "payload": {"phase": "thinking"}}),
            _encode_json_frame({"type": "event", "event": "chat.delta", "payload": {"content": "Hello"}}),
            _encode_json_frame({"type": "event", "event": "chat.delta", "payload": {"content": " world"}}),
            _encode_json_frame({"type": "chat_response", "payload": {"text": "Hello world"}}),
        ]
        for frame in events:
            connected_client._reader.feed_data(frame)
        connected_client._reader.feed_eof()

        received_events = []

        async def on_event(event):
            received_events.append(event)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=on_event,
        )

        assert len(received_events) == 3
        assert received_events[0]["event"] == "process.phase"
        assert received_events[1]["event"] == "chat.delta"
        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Hello world"

    async def test_send_stream_no_on_event_skips_events(self, connected_client):
        """With on_event=None, event frames are skipped and only final response returned."""
        frames = [
            _encode_json_frame({"type": "event", "event": "chat.delta", "payload": {"content": "skip"}}),
            _encode_json_frame({"type": "chat_response", "payload": {"text": "Final"}}),
        ]
        for frame in frames:
            connected_client._reader.feed_data(frame)
        connected_client._reader.feed_eof()

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=None,
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Final"

    async def test_send_stream_handles_error_response(self, connected_client):
        """Error response is returned directly."""
        frames = [
            _encode_json_frame({"type": "error", "payload": {"message": "Something wrong"}}),
        ]
        for frame in frames:
            connected_client._reader.feed_data(frame)
        connected_client._reader.feed_eof()

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=AsyncMock(),
        )

        assert result["type"] == "error"
        assert "Something wrong" in result["payload"]["message"]

    async def test_send_stream_handles_close_frame(self, connected_client):
        """Close frame mid-stream raises ConnectionError."""
        close_frame = _encode_frame(OP_CLOSE, b"")
        connected_client._reader.feed_data(close_frame)
        connected_client._reader.feed_eof()

        with pytest.raises(ConnectionError, match="closed"):
            await connected_client.send_stream(
                {"type": "chat", "payload": {"text": "Hi"}},
                on_event=AsyncMock(),
            )

    async def test_send_stream_handles_ping_interleaved(self, connected_client):
        """Ping frame interleaved with event frames is handled (pong sent back)."""
        frames = [
            _encode_frame(OP_PING, b"ping-data"),
            _encode_json_frame({"type": "event", "event": "chat.delta", "payload": {"content": "A"}}),
            _encode_json_frame({"type": "chat_response", "payload": {"text": "A"}}),
        ]
        for frame in frames:
            connected_client._reader.feed_data(frame)
        connected_client._reader.feed_eof()

        received_events = []

        async def on_event(event):
            received_events.append(event)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=on_event,
        )

        assert len(received_events) == 1
        assert result["type"] == "chat_response"
        # Verify pong was sent back
        assert connected_client._writer.write.called

    async def test_send_stream_not_connected_raises(self):
        """send_stream() raises if not connected."""
        client = GatewayClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.send_stream({"type": "chat", "payload": {}})
