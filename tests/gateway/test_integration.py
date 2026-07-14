"""Integration tests for the Gateway daemon lifecycle.

These tests exercise the real daemon.start() path with a real SQLite
database and a real HTTP connection.  Only the LLM is mocked.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── daemon lifecycle (real DB, real WS, mocked LLM) ──────────────────


@pytest.mark.integration
@pytest.mark.asyncio
class TestDaemonLifecycle:
    """Start daemon with real infrastructure, mocked LLM, connect via WS."""

    async def test_startup_ping_shutdown(self, tmp_path, monkeypatch):
        """Full lifecycle: start → ping/pong → shutdown → PID file removed.

        This would have caught the checkpointer.setup() AttributeError
        because _init_agent runs for real against a real SQLite database.
        """
        import httpx

        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        TEST_PORT = 18765
        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        # Fake API key so _init_model doesn't raise KeyError
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="You are a test assistant.",
                core_belief="Test core belief.",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        # Mock ChatOpenAI so _init_model succeeds without an API key.
        # Also mock create_agent + SummarizationMiddleware — the agent isn't exercised
        # by this test (we only ping), but _init_agent still runs the real
        # checkpointer setup (AsyncSqliteSaver.from_conn_string), which is
        # what caught the setup() AttributeError.
        mock_model = MagicMock()
        mock_model.profile = {}
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))

        from lingya.gateway.daemon import GatewayDaemon

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        with patch(
            "lingya.app.LiteLLMModel", return_value=mock_model
        ), \
             patch(
            "lingya.app.create_agent", return_value=MagicMock()
        ), \
             patch(
            "lingya.app.SummarizationMiddleware",
            return_value=MagicMock(),
        ):
            # Start daemon in background — it blocks on shutdown_event
            daemon_task = asyncio.create_task(daemon.start())

            try:
                # Wait for PID file to appear (daemon writes it AFTER server is ready)
                for _ in range(100):
                    if os.path.exists(pid_file):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Daemon did not write PID file within 5 seconds")

                # Connect via HTTP client — PID file signals server readiness
                async with httpx.AsyncClient(base_url=f"http://localhost:{TEST_PORT}") as http:
                    # Ping → /health returns {"status": "ok"}
                    resp = await http.get("/health")
                    assert resp.status_code == 200
                    assert resp.json() == {"status": "ok"}

            finally:
                # Trigger graceful shutdown — simulates daemon_main() lifecycle
                daemon._shutdown_event.set()
                await asyncio.wait_for(daemon_task, timeout=5.0)
                await daemon.shutdown()

        # PID file must be cleaned up
        assert not os.path.exists(pid_file), "PID file should be removed on shutdown"

    async def test_pid_file_is_readiness_signal(self, tmp_path, monkeypatch):
        """PID file appears only AFTER the WebSocket server is listening.

        This is the integration-level assertion of the ordering bug fix.
        """
        import httpx

        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        TEST_PORT = 18766
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

        config = Config(
            db_path=str(tmp_path / "lingya.db"),
            memory_path=str(tmp_path / "memory"),
            data_dir=str(tmp_path / "data"),
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="You are a test assistant.",
                core_belief="Test core belief.",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        mock_model = MagicMock()
        mock_model.profile = {}
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="mock response"))

        from lingya.gateway.daemon import GatewayDaemon

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        with patch(
            "lingya.app.LiteLLMModel", return_value=mock_model
        ), \
             patch(
            "lingya.app.create_agent", return_value=MagicMock()
        ), \
             patch(
            "lingya.app.SummarizationMiddleware",
            return_value=MagicMock(),
        ):
            daemon_task = asyncio.create_task(daemon.start())

            try:
                # Wait for PID file
                for _ in range(100):
                    if os.path.exists(pid_file):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Daemon did not write PID file")

                # If PID file is the readiness signal, the server must be
                # reachable immediately — no additional wait needed
                async with httpx.AsyncClient(base_url=f"http://localhost:{TEST_PORT}") as http:
                    resp = await http.get("/health")
                    assert resp.status_code == 200
                    assert resp.json() == {"status": "ok"}

            finally:
                daemon._shutdown_event.set()
                await asyncio.wait_for(daemon_task, timeout=5.0)
                await daemon.shutdown()


# ── Session + Chat lifecycle ────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
class TestSessionLifecycle:
    """End-to-end session CRUD with real HTTP and real SQLite, mocked LLM."""

    async def test_session_lifecycle(self, tmp_path, monkeypatch):
        """Create sessions → chat → verify endpoints work correctly."""
        import httpx

        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        TEST_PORT = 18767
        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        pid_file = str(tmp_path / "lingya.pid")
        (tmp_path / "data").mkdir(exist_ok=True)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-agent", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="You are a test assistant.",
                core_belief="Test core belief.",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["Be honest."],
        )

        mock_model = MagicMock()
        mock_model.profile = {}
        mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="Hello from LingYa"))

        # Mock agent: astream_events and aget_state for history
        mock_agent = MagicMock()

        # Storage for fake checkpoints (keyed by thread_id)
        _fake_state: dict[str, list] = {}

        async def _mock_astream_events(*args, **kwargs):
            from langchain_core.messages import AIMessage

            messages = args[0].get("messages", [])
            thread_id = args[1].get("configurable", {}).get("thread_id", "unknown")
            accumulated = "Hello from LingYa"

            # Store messages in fake state so aget_state can retrieve them
            saved = list(messages) + [AIMessage(content=accumulated)]
            _fake_state[thread_id] = saved

            async def _stream():
                yield {
                    "method": "messages",
                    "params": {
                        "data": (
                            {"event": "content-block-delta", "delta": {"type": "text-delta", "text": accumulated}},
                            {},
                        ),
                    },
                }
            return _stream()

        async def _mock_aget_state(config):
            thread_id = config.get("configurable", {}).get("thread_id", "")
            msgs = _fake_state.get(thread_id, [])
            state = MagicMock()
            state.values = {"messages": msgs}
            return state

        mock_agent.astream_events = _mock_astream_events
        mock_agent.aget_state = _mock_aget_state

        from lingya.gateway.daemon import GatewayDaemon

        daemon = GatewayDaemon(
            config=config,
            mind_config=mind_config,
            pid_file=pid_file,
            port=TEST_PORT,
        )

        with patch(
            "lingya.app.LiteLLMModel", return_value=mock_model
        ), \
             patch(
            "lingya.app.create_agent", return_value=mock_agent
        ), \
             patch(
            "lingya.app.SummarizationMiddleware",
            return_value=MagicMock(),
        ):
            daemon_task = asyncio.create_task(daemon.start())

            try:
                # Wait for readiness
                for _ in range(100):
                    if os.path.exists(pid_file):
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Daemon did not write PID file within 5 seconds")

                base_url = f"http://localhost:{TEST_PORT}"

                async with httpx.AsyncClient(base_url=base_url) as http:
                    # ── 1. Create session 1 ─────────────────────────
                    resp = await http.post("/session", json={"action": "new"})
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["type"] == "session_response"
                    thread_1 = data["payload"]["thread_id"]
                    assert thread_1.startswith("ws-")

                    # ── 2. Create session 2 ─────────────────────────
                    resp = await http.post("/session", json={"action": "new"})
                    assert resp.status_code == 200
                    thread_2 = resp.json()["payload"]["thread_id"]
                    assert thread_2 != thread_1

                    # ── 3. Chat in session 2 (current) ──────────────
                    resp = await http.post("/chat", json={"text": "Hello"})
                    assert resp.status_code == 200
                    body = resp.text
                    assert "chat_response" in body
                    assert "Hello from LingYa" in body

                    # ── 4. Verify history for session 2 ─────────────
                    resp = await http.get(
                        "/session/history", params={"thread_id": thread_2}
                    )
                    assert resp.status_code == 200
                    history = resp.json()
                    assert history["type"] == "session_response"
                    assert history["payload"]["action"] == "history"
                    msgs = history["payload"]["messages"]
                    assert len(msgs) >= 2  # user + her
                    roles = [m["role"] for m in msgs]
                    assert "user" in roles
                    assert "her" in roles

                    # ── 5. Seed checkpoints so switch/delete work ────
                    # With a mocked agent, checkpoints aren't auto-created.
                    import aiosqlite
                    async with aiosqlite.connect(db_path) as conn:
                        # LangGraph checkpoints schema: (thread_id, checkpoint_ns, checkpoint_id, checkpoint, ...)
                        await conn.execute(
                            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
                            "VALUES (?, '', ?, ?)",
                            (thread_1, f"cp-{thread_1}", b"{}"),
                        )
                        await conn.execute(
                            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
                            "VALUES (?, '', ?, ?)",
                            (thread_2, f"cp-{thread_2}", b"{}"),
                        )
                        await conn.commit()

                    # ── 6. Switch to session 1 ──────────────────────
                    resp = await http.post(
                        "/session",
                        json={"action": "switch", "thread_id": thread_1},
                    )
                    assert resp.status_code == 200
                    switch_data = resp.json()
                    assert switch_data["payload"]["thread_id"] == thread_1

                    # ── 7. Chat in session 1 ────────────────────────
                    resp = await http.post("/chat", json={"text": "Hi again"})
                    assert resp.status_code == 200
                    assert "chat_response" in resp.text

                    # ── 8. Verify isolation: session 1 has separate history ──
                    resp = await http.get(
                        "/session/history", params={"thread_id": thread_1}
                    )
                    msgs_1 = resp.json()["payload"]["messages"]
                    assert len(msgs_1) >= 2  # Has its own messages

                    # ── 9. Current session endpoint ─────────────────
                    resp = await http.get("/session/current")
                    assert resp.status_code == 200
                    current = resp.json()["payload"]["session"]
                    assert current["thread_id"] == thread_1
                    assert current["is_current"] is True

                    # ── 10. List sessions (structure check) ──────────
                    resp = await http.get("/session/list")
                    assert resp.status_code == 200
                    list_data = resp.json()
                    assert list_data["type"] == "session_response"
                    assert list_data["payload"]["action"] == "list"
                    sessions = list_data["payload"]["sessions"]
                    assert isinstance(sessions, list)
                    assert len(sessions) >= 2

                    # ── 11. Delete non-current session (session 2) ──
                    resp = await http.post(
                        "/session",
                        json={"action": "delete", "thread_id": thread_2},
                    )
                    assert resp.status_code == 200
                    assert resp.json()["payload"]["deleted"] is True

                    # ── 12. Cannot delete current session ───────────
                    resp = await http.post(
                        "/session",
                        json={"action": "delete", "thread_id": thread_1},
                    )
                    assert resp.status_code == 200
                    assert "Cannot delete current session" in str(resp.json())

                    # ── 13. Switch to non-existent session → error ──
                    resp = await http.post(
                        "/session",
                        json={"action": "switch", "thread_id": "ws-nonexistent"},
                    )
                    assert resp.status_code == 200
                    assert "not found" in str(resp.json()).lower()

            finally:
                daemon._shutdown_event.set()
                await asyncio.wait_for(daemon_task, timeout=5.0)
                await daemon.shutdown()

        assert not os.path.exists(pid_file)


