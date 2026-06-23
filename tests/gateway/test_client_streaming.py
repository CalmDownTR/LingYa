"""Test GatewayClient.send_stream() — SSE event streaming edge cases."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lingya.gateway.client import GatewayClient


# ── Helpers ─────────────────────────────────────────────────────────


def _make_mock_response(status_code=200, headers=None, lines=None):
    """Create a mock httpx Response for streaming.

    Returns a properly configured AsyncMock that works with
    ``async with client.stream(...) as response:`` and
    ``async for line in response.aiter_lines():``.
    """
    if headers is None:
        headers = {"content-type": "text/event-stream"}
    if lines is None:
        lines = []

    async def aiter_lines():
        for line in lines:
            yield line

    # Use AsyncMock so async context manager and async iteration work
    resp = AsyncMock()
    resp.status_code = status_code
    resp.headers = headers
    resp.aiter_lines = aiter_lines
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"type": "error", "payload": {"message": "Bad request"}})
    return resp


def _make_mock_client():
    """Create a mocked httpx.AsyncClient."""
    mock = AsyncMock()
    mock.is_closed = False
    mock.aclose = AsyncMock()
    # stream() returns an async context manager
    mock.stream = MagicMock()
    return mock


@pytest.fixture
def connected_client():
    """Return a GatewayClient with mocked httpx client.

    The mock is configured so ``async with client._client.stream(...) as resp``
    works correctly.
    """
    client = GatewayClient()
    client._client = _make_mock_client()
    return client


# ── Tests ────────────────────────────────────────────────────────────


class _MockStreamCtx:
    """Async context manager that returns the mock response."""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
class TestSendStreamSSE:
    async def test_receives_multiple_event_types(self, connected_client):
        """All LingYa event types are received and dispatched."""
        mock_resp = _make_mock_response(lines=[
            'data: {"type":"event","event":"process.phase","payload":{"phase":"thinking"}}',
            'data: {"type":"event","event":"memory.recall","payload":{"count":2}}',
            'data: {"type":"event","event":"chat.delta","payload":{"content":"Hello"}}',
            'data: {"type":"event","event":"mind.transition","payload":{"pad":{"pleasure":0.5}}}',
            'data: {"type":"chat_response","payload":{"text":"Hello","tone":{},"meta":{}}}',
        ])
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        received = []
        async def on_event(e):
            received.append(e)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=on_event,
        )

        assert len(received) == 4
        assert received[0]["event"] == "process.phase"
        assert received[1]["event"] == "memory.recall"
        assert received[2]["event"] == "chat.delta"
        assert received[3]["event"] == "mind.transition"
        assert result["type"] == "chat_response"

    async def test_skips_events_when_no_callback(self, connected_client):
        """With on_event=None, event frames are consumed silently."""
        mock_resp = _make_mock_response(lines=[
            'data: {"type":"event","event":"chat.delta","payload":{"content":"skip"}}',
            'data: {"type":"chat_response","payload":{"text":"Final","tone":{},"meta":{}}}',
        ])
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
            on_event=None,
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Final"

    async def test_handles_error_response(self, connected_client):
        """Error from server is returned as final."""
        mock_resp = _make_mock_response(lines=[
            'data: {"type":"error","payload":{"message":"Something wrong"}}',
        ])
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
        )

        assert result["type"] == "error"
        assert "Something wrong" in result["payload"]["message"]

    async def test_handles_non_sse_response(self, connected_client):
        """If server returns JSON instead of SSE, return it directly."""
        mock_resp = _make_mock_response(
            headers={"content-type": "application/json"},
            lines=[],
        )
        mock_resp.json = MagicMock(return_value={"type": "error", "payload": {"message": "Bad request"}})
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": ""}},
        )
        assert result["type"] == "error"

    async def test_handles_empty_sse_stream(self, connected_client):
        """Stream with no final frame returns error."""
        mock_resp = _make_mock_response(lines=[
            'data: {"type":"event","event":"chat.delta","payload":{"content":"x"}}',
            # No final response frame
        ])
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
        )
        assert result["type"] == "error"
        assert "without final response" in result["payload"]["message"]

    async def test_handles_malformed_sse_line(self, connected_client):
        """Malformed JSON in SSE line is skipped."""
        mock_resp = _make_mock_response(lines=[
            'data: not-valid-json',
            'data: {"type":"chat_response","payload":{"text":"OK","tone":{},"meta":{}}}',
        ])
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
        )
        assert result["type"] == "chat_response"

    async def test_handles_http_error(self, connected_client):
        """HTTP error status is caught and returned as error."""
        mock_resp = _make_mock_response(status_code=503)
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=MagicMock(status_code=503)
        )
        connected_client._client.stream = lambda *a, **kw: _MockStreamCtx(mock_resp)

        result = await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
        )
        assert result["type"] == "error"
        assert "503" in result["payload"]["message"]

    async def test_auth_headers_included(self, connected_client):
        """When LINGYA_API_KEY is set, Authorization header is sent."""
        connected_client._api_key = "test-key-123"
        mock_resp = _make_mock_response(lines=[
            'data: {"type":"chat_response","payload":{"text":"OK","tone":{},"meta":{}}}',
        ])
        # Use a MagicMock for stream so we can inspect call_args
        mock_stream = MagicMock(return_value=_MockStreamCtx(mock_resp))
        connected_client._client.stream = mock_stream

        await connected_client.send_stream(
            {"type": "chat", "payload": {"text": "Hi"}},
        )

        # Check that Authorization header was passed
        call_kwargs = mock_stream.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key-123"
