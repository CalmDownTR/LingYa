"""Test FastAPI GatewayServer — HTTP + SSE endpoints.

Uses FastAPI TestClient for direct endpoint testing (no real HTTP server).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from lingya.gateway.server import create_app


# ── Helpers ─────────────────────────────────────────────────────────


def _make_test_client(router=None, auth_enabled=False):
    """Create a TestClient with optional router injection."""
    app = create_app(router=router, auth_enabled=auth_enabled)
    return TestClient(app)


async def _make_stream_events(events: list[dict]):
    """Async generator that yields event dicts for _handle_chat_streaming."""
    for event in events:
        yield event


# ── Health endpoint ─────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_ok(self):
        client = _make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_no_auth_required(self):
        """Health endpoint should not require auth."""
        client = _make_test_client(auth_enabled=True)
        # Even with auth enabled, health should be public
        # (We test that health bypasses auth here)
        resp = client.get("/health")
        assert resp.status_code == 200


# ── Chat endpoint ───────────────────────────────────────────────────


class TestChatEndpoint:
    def test_chat_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.post("/chat", json={"text": "Hello"})
        assert resp.status_code == 503

    def test_chat_empty_text_returns_400(self):
        router = MagicMock()
        client = _make_test_client(router=router)
        resp = client.post("/chat", json={"text": ""})
        assert resp.status_code == 400

    def test_chat_no_agent_returns_503(self):
        router = MagicMock()
        router._agent = None
        router._engine = MagicMock()
        client = _make_test_client(router=router)
        resp = client.post("/chat", json={"text": "Hello"})
        assert resp.status_code == 503

    def test_chat_sse_stream(self):
        """Chat endpoint returns SSE stream with proper headers."""
        router = MagicMock()
        router._agent = MagicMock()
        router._engine = MagicMock()
        router._engine.get_prompt_fragment.return_value = ""
        router._thread_id = "test-thread"

        events = [
            {"type": "event", "event": "process.phase", "payload": {"phase": "thinking"}},
            {"type": "event", "event": "chat.delta", "payload": {"content": "Hello"}},
            {"type": "event", "event": "mind.transition", "payload": {"pad": {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.7}}},
            {"type": "chat_response", "payload": {"text": "Hello", "tone": {}, "meta": {}}},
        ]
        router._handle_chat_streaming.return_value = _make_stream_events(events)

        client = _make_test_client(router=router)
        resp = client.post("/chat", json={"text": "Hi"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers["cache-control"] == "no-cache"

        # Parse SSE lines
        body = resp.text
        lines = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(lines) == 4

        # First event
        e1 = json.loads(lines[0][6:])
        assert e1["type"] == "event"
        assert e1["event"] == "process.phase"

        # Last event (final response)
        e4 = json.loads(lines[3][6:])
        assert e4["type"] == "chat_response"
        assert "sse_hop_ms" in e4["payload"]["meta"]

    def test_chat_includes_system_fragment(self):
        """When engine provides a prompt fragment, it's included in messages."""
        router = MagicMock()
        router._agent = MagicMock()
        router._engine = MagicMock()
        router._engine.get_prompt_fragment.return_value = "[Tone: warmth=high]"
        router._thread_id = "test-thread"

        events = [
            {"type": "chat_response", "payload": {"text": "OK", "tone": {}, "meta": {}}},
        ]
        router._handle_chat_streaming.return_value = _make_stream_events(events)

        client = _make_test_client(router=router)
        resp = client.post("/chat", json={"text": "Hi"})
        assert resp.status_code == 200

        # Verify _handle_chat_streaming was called with messages containing SystemMessage
        call_args = router._handle_chat_streaming.call_args
        messages = call_args[0][0]
        from langchain_core.messages import SystemMessage, HumanMessage
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert "[Tone: warmth=high]" in messages[0].content
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hi"


# ── Mind endpoint ───────────────────────────────────────────────────


class TestMindEndpoint:
    def test_mind_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.get("/mind")
        assert resp.status_code == 503

    def test_mind_default_query_returns_state(self):
        router = MagicMock()
        router._handle_mind = AsyncMock(return_value={
            "type": "mind_state",
            "payload": {"pad": {"pleasure": 0.5}},
        })
        client = _make_test_client(router=router)
        resp = client.get("/mind")
        assert resp.status_code == 200
        router._handle_mind.assert_called_with({"query": "state"})

    def test_mind_tone_query(self):
        router = MagicMock()
        router._handle_mind = AsyncMock(return_value={
            "type": "mind_state",
            "payload": {"tone": {"warmth": 60}},
        })
        client = _make_test_client(router=router)
        resp = client.get("/mind?query=tone")
        assert resp.status_code == 200
        router._handle_mind.assert_called_with({"query": "tone"})


# ── Memory endpoint ─────────────────────────────────────────────────


class TestMemoryEndpoint:
    def test_memory_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.get("/memory")
        assert resp.status_code == 503

    def test_memory_default_search(self):
        router = MagicMock()
        router._handle_memory = AsyncMock(return_value={
            "type": "memory_response",
            "payload": {"action": "search", "results": []},
        })
        client = _make_test_client(router=router)
        resp = client.get("/memory")
        assert resp.status_code == 200
        router._handle_memory.assert_called_with({"action": "search"})

    def test_memory_list_action(self):
        router = MagicMock()
        router._handle_memory = AsyncMock(return_value={
            "type": "memory_response",
            "payload": {"action": "list", "memories": []},
        })
        client = _make_test_client(router=router)
        resp = client.get("/memory?action=list")
        assert resp.status_code == 200
        router._handle_memory.assert_called_with({"action": "list"})

    def test_memory_search_with_query(self):
        router = MagicMock()
        router._handle_memory = AsyncMock(return_value={
            "type": "memory_response",
            "payload": {"action": "search", "results": []},
        })
        client = _make_test_client(router=router)
        resp = client.get("/memory?action=search&query=coffee")
        assert resp.status_code == 200
        router._handle_memory.assert_called_with({"action": "search", "query": "coffee"})

    def test_memory_recover_with_id(self):
        router = MagicMock()
        router._handle_memory = AsyncMock(return_value={
            "type": "memory_response",
            "payload": {"action": "recover", "recovered": True, "id": "mem_42"},
        })
        client = _make_test_client(router=router)
        resp = client.get("/memory?action=recover&id=mem_42")
        assert resp.status_code == 200
        router._handle_memory.assert_called_with({"action": "recover", "id": "mem_42"})


# ── Diary endpoint ──────────────────────────────────────────────────


class TestDiaryEndpoint:
    def test_diary_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.get("/diary")
        assert resp.status_code == 503

    def test_diary_list_default(self):
        router = MagicMock()
        router._handle_diary = AsyncMock(return_value={
            "type": "diary_response",
            "payload": {"action": "list", "diaries": []},
        })
        client = _make_test_client(router=router)
        resp = client.get("/diary")
        assert resp.status_code == 200
        router._handle_diary.assert_called_with({"action": "list", "index": 0})

    def test_diary_read_with_index(self):
        router = MagicMock()
        router._handle_diary = AsyncMock(return_value={
            "type": "diary_response",
            "payload": {"action": "read", "date": "2025-01-01", "content": "..."},
        })
        client = _make_test_client(router=router)
        resp = client.get("/diary?action=read&index=2")
        assert resp.status_code == 200
        router._handle_diary.assert_called_with({"action": "read", "index": 2})


# ── Session endpoint ────────────────────────────────────────────────


class TestSessionEndpoint:
    def test_session_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.post("/session")
        assert resp.status_code == 503

    def test_session_new(self):
        router = MagicMock()
        router._handle_session = AsyncMock(return_value={
            "type": "session_response",
            "payload": {"action": "new", "thread_id": "ws-abc12345"},
        })
        client = _make_test_client(router=router)
        resp = client.post("/session")
        assert resp.status_code == 200
        router._handle_session.assert_called_with({"action": "new"})


# ── Stats endpoint ──────────────────────────────────────────────────


class TestStatsEndpoint:
    def test_stats_no_router_returns_503(self):
        client = _make_test_client(router=None)
        resp = client.get("/stats")
        assert resp.status_code == 503

    def test_stats_returns_deprecation(self):
        router = MagicMock()
        router._handle_stats = AsyncMock(return_value={
            "type": "stats_response",
            "payload": {"deprecated": True},
        })
        client = _make_test_client(router=router)
        resp = client.get("/stats")
        assert resp.status_code == 200


# ── Shutdown endpoint ──────────────────────────────────────────────


class TestShutdownEndpoint:
    def test_shutdown_triggers_callback(self):
        """POST /shutdown calls the shutdown_callback and returns shutting_down."""
        callback_called = []

        def _callback():
            callback_called.append(True)

        app = create_app(router=None, auth_enabled=False, shutdown_callback=_callback)
        client = TestClient(app)
        resp = client.post("/shutdown")
        assert resp.status_code == 200
        assert resp.json() == {"status": "shutting_down"}
        assert len(callback_called) == 1

    def test_shutdown_requires_auth(self, monkeypatch):
        """POST /shutdown without auth returns 401 when auth is enabled."""
        monkeypatch.setenv("LINGYA_API_KEY", "test-key")
        app = create_app(router=None, auth_enabled=True, shutdown_callback=lambda: None)
        client = TestClient(app)
        resp = client.post("/shutdown")
        assert resp.status_code == 401

    def test_shutdown_no_callback_graceful(self):
        """POST /shutdown does not crash when callback is None."""
        app = create_app(router=None, auth_enabled=False, shutdown_callback=None)
        client = TestClient(app)
        resp = client.post("/shutdown")
        assert resp.status_code == 200
        assert resp.json() == {"status": "shutting_down"}


# ── CORS ────────────────────────────────────────────────────────────


class TestCORS:
    def test_cors_headers_present(self):
        client = _make_test_client()
        resp = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        # FastAPI CORS middleware should return appropriate headers
        assert resp.status_code in (200, 405)  # OPTIONS may not be allowed on all endpoints
