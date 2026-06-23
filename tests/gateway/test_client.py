"""Test GatewayClient — HTTP + SSE client.

Tests the httpx-based client with mocked HTTP transport.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lingya.gateway.client import GatewayClient


# ── Helpers ─────────────────────────────────────────────────────────


def _make_mock_response(json_data, status_code=200):
    """Create a mock httpx Response for non-streaming requests."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_client():
    """Create a mocked httpx.AsyncClient."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.is_closed = False
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def connected_client():
    """Return a GatewayClient with mocked httpx client."""
    client = GatewayClient()
    client._client = _make_mock_client()
    return client


# ── Init tests ──────────────────────────────────────────────────────


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


# ── Connect tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGatewayClientConnect:
    async def test_connect_success(self, connected_client):
        """connect() verifies server via /health."""
        mock_http = _make_mock_client()
        mock_http.get.return_value = _make_mock_response({"status": "ok"})

        with patch("lingya.gateway.client.httpx.AsyncClient", return_value=mock_http):
            await connected_client.connect()

        assert connected_client.is_connected
        mock_http.get.assert_called_with("/health", headers={})

    async def test_connect_failure(self, connected_client):
        """connect() raises ConnectionError if server unreachable."""
        mock_http = _make_mock_client()
        mock_http.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("lingya.gateway.client.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await connected_client.connect()
        assert not connected_client.is_connected
        mock_http.aclose.assert_called_once()


# ── Send tests (non-chat) ───────────────────────────────────────────


@pytest.mark.asyncio
class TestGatewayClientSend:
    async def test_send_without_connect_raises(self):
        """send() raises if not connected."""
        client = GatewayClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.send({"type": "ping", "payload": {}})

    async def test_send_ping(self, connected_client):
        """ping maps to GET /health."""
        connected_client._client.get.return_value = _make_mock_response(
            {"status": "ok"}
        )
        result = await connected_client.send({"type": "ping", "payload": {}})
        assert result == {"status": "ok"}

    async def test_send_mind(self, connected_client):
        """mind maps to GET /mind."""
        connected_client._client.get.return_value = _make_mock_response(
            {"type": "mind_state", "payload": {"pad": {"pleasure": 0.5}}}
        )
        result = await connected_client.send({"type": "mind", "payload": {"query": "state"}})
        assert result["type"] == "mind_state"

    async def test_send_memory_search(self, connected_client):
        """memory search maps to GET /memory."""
        connected_client._client.get.return_value = _make_mock_response(
            {"type": "memory_response", "payload": {"action": "search", "results": []}}
        )
        result = await connected_client.send({
            "type": "memory",
            "payload": {"action": "search", "query": "coffee"},
        })
        assert result["payload"]["action"] == "search"

    async def test_send_memory_list(self, connected_client):
        """memory list maps to GET /memory?action=list."""
        connected_client._client.get.return_value = _make_mock_response(
            {"type": "memory_response", "payload": {"action": "list", "memories": []}}
        )
        result = await connected_client.send({
            "type": "memory", "payload": {"action": "list"},
        })
        assert result["payload"]["action"] == "list"

    async def test_send_diary(self, connected_client):
        """diary maps to GET /diary."""
        connected_client._client.get.return_value = _make_mock_response(
            {"type": "diary_response", "payload": {"action": "list", "diaries": []}}
        )
        result = await connected_client.send({
            "type": "diary", "payload": {"action": "list"},
        })
        assert result["type"] == "diary_response"

    async def test_send_session(self, connected_client):
        """session maps to POST /session."""
        connected_client._client.post.return_value = _make_mock_response(
            {"type": "session_response", "payload": {"action": "new", "thread_id": "ws-test"}}
        )
        result = await connected_client.send({
            "type": "session", "payload": {"action": "new"},
        })
        assert result["type"] == "session_response"

    async def test_send_stats(self, connected_client):
        """stats maps to GET /stats."""
        connected_client._client.get.return_value = _make_mock_response(
            {"type": "stats_response", "payload": {"deprecated": True}}
        )
        result = await connected_client.send({"type": "stats", "payload": {}})
        assert result["type"] == "stats_response"

    async def test_send_unknown_type(self, connected_client):
        """Unknown message type returns error."""
        result = await connected_client.send({"type": "unknown", "payload": {}})
        assert result["type"] == "error"


# ── Send stream tests (chat) ────────────────────────────────────────


@pytest.mark.asyncio
class TestGatewayClientSendStream:
    async def test_send_stream_with_events(self, connected_client):
        """send_stream() receives SSE events and returns final response."""
        from .test_client_streaming import _make_mock_response as _sse_resp

        mock_resp = _sse_resp(lines=[
            'data: {"type":"event","event":"chat.delta","payload":{"content":"Hello"}}',
            'data: {"type":"chat_response","payload":{"text":"Hello","tone":{},"meta":{}}}',
        ])
        connected_client._client.stream.return_value.__aenter__.return_value = mock_resp

        received = []
        async def on_event(e):
            received.append(e)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=on_event,
        )

        assert len(received) == 1
        assert received[0]["event"] == "chat.delta"
        assert result["type"] == "chat_response"

    async def test_send_stream_no_callback(self, connected_client):
        """send_stream() with on_event=None returns only final response."""
        from .test_client_streaming import _make_mock_response as _sse_resp

        mock_resp = _sse_resp(lines=[
            'data: {"type":"chat_response","payload":{"text":"Final","tone":{},"meta":{}}}',
        ])
        connected_client._client.stream.return_value.__aenter__.return_value = mock_resp

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=None,
        )
        assert result["type"] == "chat_response"

    async def test_send_stream_not_connected_raises(self):
        """send_stream() raises if not connected."""
        client = GatewayClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.send_stream({"type": "chat", "payload": {}})


# ── Close tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGatewayClientClose:
    async def test_close_cleans_up(self, connected_client):
        """close() resets state."""
        assert connected_client.is_connected
        mock = connected_client._client  # save ref before close() sets to None
        await connected_client.close()
        assert not connected_client.is_connected
        mock.aclose.assert_called_once()

    async def test_close_when_not_connected_is_noop(self):
        """close() is a no-op when not connected."""
        client = GatewayClient()
        await client.close()  # Should not raise
        assert not client.is_connected
