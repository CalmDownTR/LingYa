"""Test MessageRouter in isolation — no WebSocket, no network."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lingya.gateway.router import MessageRouter


# Fixtures (mock_engine, mock_memory, mock_db, mock_agent, router, router_no_agent)
# are shared via tests/gateway/conftest.py
# mock_agent fixture from conftest includes both ainvoke and astream_events


# ── Ping tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPing:
    async def test_ping_returns_pong_with_timestamp(self, router):
        result = await router.route({"type": "ping", "payload": {}})

        assert result["type"] == "pong"
        assert "timestamp" in result["payload"]
        # Should be a valid ISO 8601 timestamp
        ts = result["payload"]["timestamp"]
        assert "T" in ts
        datetime.fromisoformat(ts)  # Should not raise

    async def test_ping_ignores_payload(self, router):
        result = await router.route(
            {"type": "ping", "payload": {"extra": "ignored"}}
        )

        assert result["type"] == "pong"
        assert "timestamp" in result["payload"]


# ── Mind state tests ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestMindState:
    async def test_mind_state_returns_full_state(self, router, mock_engine):
        result = await router.route(
            {"type": "mind", "payload": {"query": "state"}}
        )

        assert result["type"] == "mind_state"
        p = result["payload"]

        # PAD
        assert p["pad"]["pleasure"] == 0.5
        assert p["pad"]["arousal"] == 0.3
        assert p["pad"]["dominance"] == 0.7

        # Emotion (most recent)
        assert p["emotion"] == "surprise"
        assert p["emotion_intensity"] == 0.4

        # IPC
        assert p["ipc_state"] == "agency_high"
        assert p["ipc_agency"] == 0.8
        assert p["ipc_communion"] == 0.5

        # Tone
        assert p["tone"]["warmth"] == 60
        assert p["tone"]["formality"] == 50
        assert p["tone"]["humor"] == 0.3

        # OCEAN
        assert p["ocean"]["openness"] == 0.6
        assert p["ocean"]["conscientiousness"] == 0.7
        assert p["ocean"]["extraversion"] == 0.5
        assert p["ocean"]["agreeableness"] == 0.8
        assert p["ocean"]["neuroticism"] == 0.4

        # Turn counter
        assert p["turn_counter"] == 42

    async def test_mind_state_default_query_is_state(self, router):
        """No query specified defaults to full state."""
        result = await router.route({"type": "mind", "payload": {}})

        assert result["type"] == "mind_state"
        assert "pad" in result["payload"]
        assert "tone" in result["payload"]

    async def test_mind_tone_returns_only_tone_params(self, router):
        result = await router.route(
            {"type": "mind", "payload": {"query": "tone"}}
        )

        assert result["type"] == "mind_state"
        p = result["payload"]
        assert "tone" in p
        assert p["tone"]["warmth"] == 60
        assert p["tone"]["formality"] == 50
        assert p["tone"]["humor"] == 0.3
        # Should NOT include full state
        assert "pad" not in p
        assert "ocean" not in p

    async def test_mind_state_with_no_emotions_returns_neutral(self, router, mock_engine):
        """Empty recent_emotions should return neutral fallback."""
        mock_engine.state.recent_emotions = []

        result = await router.route(
            {"type": "mind", "payload": {"query": "state"}}
        )

        assert result["payload"]["emotion"] == "neutral"
        assert result["payload"]["emotion_intensity"] == 0.0


# ── Diary tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDiary:
    async def test_diary_list_returns_diaries(self, router, tmp_path):
        """List diaries returns sorted entries with preview."""
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        (diary_dir / "2025-06-01.md").write_text(
            "# 2025年06月01日\n\nToday was a good day.\n",
            encoding="utf-8",
        )
        (diary_dir / "2025-05-30.md").write_text(
            "# 2025年05月30日\n\nYesterday was quiet.\n",
            encoding="utf-8",
        )

        result = await router.route(
            {"type": "diary", "payload": {"action": "list"}}
        )

        assert result["type"] == "diary_response"
        p = result["payload"]
        assert p["action"] == "list"
        assert len(p["diaries"]) == 2
        # Newest first
        assert p["diaries"][0]["date"] == "2025-06-01"
        assert "Today was a good day" in p["diaries"][0]["preview"]
        assert p["diaries"][1]["date"] == "2025-05-30"

    async def test_diary_list_empty_directory(self, router):
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)

        result = await router.route(
            {"type": "diary", "payload": {"action": "list"}}
        )

        assert result["type"] == "diary_response"
        assert result["payload"]["diaries"] == []

    async def test_diary_list_default_action(self, router, tmp_path):
        """Default action should be 'list'."""
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)

        result = await router.route({"type": "diary", "payload": {}})

        assert result["type"] == "diary_response"
        assert result["payload"]["action"] == "list"

    async def test_diary_read_returns_content(self, router, tmp_path):
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        (diary_dir / "2025-06-01.md").write_text(
            "# 2025年06月01日\n\nToday was a good day.\n",
            encoding="utf-8",
        )
        (diary_dir / "2025-05-30.md").write_text(
            "# 2025年05月30日\n\nYesterday was quiet.\n",
            encoding="utf-8",
        )

        # Read latest (index 0)
        result = await router.route(
            {"type": "diary", "payload": {"action": "read", "index": 0}}
        )

        assert result["type"] == "diary_response"
        assert result["payload"]["action"] == "read"
        assert result["payload"]["date"] == "2025-06-01"
        assert "Today was a good day" in result["payload"]["content"]

    async def test_diary_read_second_entry(self, router, tmp_path):
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        (diary_dir / "2025-06-01.md").write_text(
            "# 2025年06月01日\n\nFirst diary.\n",
            encoding="utf-8",
        )
        (diary_dir / "2025-05-30.md").write_text(
            "# 2025年05月30日\n\nSecond diary.\n",
            encoding="utf-8",
        )

        result = await router.route(
            {"type": "diary", "payload": {"action": "read", "index": 1}}
        )

        assert result["payload"]["date"] == "2025-05-30"
        assert "Second diary" in result["payload"]["content"]

    async def test_diary_read_out_of_range(self, router, tmp_path):
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        (diary_dir / "2025-06-01.md").write_text(
            "# 2025年06月01日\n\nContent.\n",
            encoding="utf-8",
        )

        result = await router.route(
            {"type": "diary", "payload": {"action": "read", "index": 99}}
        )

        assert result["type"] == "error"
        assert "No diary at index 99" in result["payload"]["message"]

    async def test_diary_read_default_index_is_zero(self, router, tmp_path):
        diary_dir = Path(router._data_dir) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        (diary_dir / "2025-06-01.md").write_text(
            "# 2025年06月01日\n\nContent.\n",
            encoding="utf-8",
        )

        result = await router.route(
            {"type": "diary", "payload": {"action": "read"}}
        )

        assert result["payload"]["date"] == "2025-06-01"


# ── Memory tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestMemory:
    async def test_memory_search_returns_results(self, router):
        result = await router.route(
            {"type": "memory", "payload": {"action": "search", "query": "coffee"}}
        )

        assert result["type"] == "memory_response"
        assert result["payload"]["action"] == "search"
        assert len(result["payload"]["results"]) == 2
        assert result["payload"]["results"][0]["text"] == "User likes coffee"

    async def test_memory_search_default_action(self, router):
        result = await router.route(
            {"type": "memory", "payload": {"query": "test"}}
        )

        assert result["type"] == "memory_response"
        assert result["payload"]["action"] == "search"

    async def test_memory_list_returns_all(self, router):
        result = await router.route(
            {"type": "memory", "payload": {"action": "list"}}
        )

        assert result["type"] == "memory_response"
        assert result["payload"]["action"] == "list"
        assert len(result["payload"]["memories"]) == 2

    async def test_memory_search_with_empty_query(self, router):
        """Empty query should still work (passed through to memory.search)."""
        result = await router.route(
            {"type": "memory", "payload": {"action": "search", "query": ""}}
        )

        assert result["type"] == "memory_response"
        assert result["payload"]["action"] == "search"

    async def test_memory_recover_valid_id_returns_success(self, router, mock_memory):
        """Recover with valid id returns success."""
        mock_memory.recover = MagicMock(return_value=True)
        result = await router.route(
            {"type": "memory", "payload": {"action": "recover", "id": "mem_42"}}
        )
        assert result["type"] == "memory_response"
        assert result["payload"]["action"] == "recover"
        assert result["payload"]["recovered"] is True
        assert result["payload"]["id"] == "mem_42"
        mock_memory.recover.assert_called_once_with("mem_42")

    async def test_memory_recover_missing_id_returns_error(self, router):
        """Recover with missing id returns error."""
        result = await router.route(
            {"type": "memory", "payload": {"action": "recover"}}
        )
        assert result["type"] == "error"
        assert "Missing memory id" in result["payload"]["message"]

    async def test_memory_recover_nonexistent_id_returns_false(self, router, mock_memory):
        """Recover with non-existent id returns False but doesn't error."""
        mock_memory.recover = MagicMock(return_value=False)
        result = await router.route(
            {"type": "memory", "payload": {"action": "recover", "id": "mem_bogus"}}
        )
        assert result["type"] == "memory_response"
        assert result["payload"]["recovered"] is False
        assert result["payload"]["id"] == "mem_bogus"


# ── Chat tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestChat:
    async def test_chat_empty_text_returns_error(self, router):
        """Empty message text returns an error."""
        result = await router.route(
            {"type": "chat", "payload": {"text": ""}}
        )

        assert result["type"] == "error"
        assert "Empty message" in result["payload"]["message"]

    async def test_chat_no_text_returns_error(self, router):
        """Missing text field returns an error."""
        result = await router.route(
            {"type": "chat", "payload": {}}
        )

        assert result["type"] == "error"
        assert "Empty message" in result["payload"]["message"]

    async def test_chat_no_agent_returns_error(self, router_no_agent):
        """Router without agent returns an error."""
        result = await router_no_agent.route(
            {"type": "chat", "payload": {"text": "Hello"}}
        )

        assert result["type"] == "error"
        assert "Agent not initialized" in result["payload"]["message"]

    async def test_chat_returns_response_with_tone(self, router, mock_agent, mock_engine):
        """Chat returns response text + tone params and processes through engine."""
        from langchain_core.messages import AIMessage, HumanMessage

        # Simulate agent response
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Hello! How can I help?")],
        }

        result = await router.route(
            {"type": "chat", "payload": {"text": "Hi there"}}
        )

        # Verify response type
        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Hello! How can I help?"
        assert result["payload"]["tone"] == {
            "warmth": 60,
            "formality": 50,
            "humor": 0.3,
        }

        # Verify agent was called with correct messages
        mock_agent.ainvoke.assert_called_once()
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        thread_config = call_args[0][1]
        assert len(messages) >= 1
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "Hi there"
        assert thread_config["configurable"]["thread_id"] == "ws-default"

        # Verify MindEngine callbacks
        mock_engine.process_event.assert_called_once()
        mock_engine.check_response_alignment.assert_called_once_with(
            "Hello! How can I help?"
        )

    async def test_chat_without_fragment(self, router, mock_agent, mock_engine):
        """Chat works when get_prompt_fragment returns empty string."""
        from langchain_core.messages import AIMessage

        mock_engine.get_prompt_fragment.return_value = ""
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Hey!")],
        }

        result = await router.route(
            {"type": "chat", "payload": {"text": "Hello"}}
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Hey!"

    async def test_chat_agent_error_returned_as_error(self, router, mock_agent):
        """Agent exception is caught and returned as error response."""
        mock_agent.ainvoke.side_effect = RuntimeError("LLM timeout")

        result = await router.route(
            {"type": "chat", "payload": {"text": "Hello"}}
        )

        assert result["type"] == "error"
        assert "LLM timeout" in result["payload"]["message"]

    async def test_chat_with_extra_payload_fields(self, router, mock_agent):
        """Extra payload fields are ignored, only text is used."""
        from langchain_core.messages import AIMessage

        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Response")],
        }

        result = await router.route(
            {"type": "chat", "payload": {"text": "anything", "extra": 123}}
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Response"


# ── Error handling tests ────────────────────────────────────────────


@pytest.mark.asyncio
class TestErrorHandling:
    async def test_unknown_message_type_returns_error(self, router):
        result = await router.route(
            {"type": "nonexistent", "payload": {}}
        )

        assert result["type"] == "error"
        assert "Unknown message type" in result["payload"]["message"]

    async def test_empty_message_returns_error(self, router):
        result = await router.route({})

        assert result["type"] == "error"
        assert "Unknown message type" in result["payload"]["message"]

    async def test_handler_exception_caught_as_error(self, router, mock_memory):
        """When a handler raises, the error is caught and returned."""
        mock_memory.search.side_effect = RuntimeError("DB connection lost")

        result = await router.route(
            {"type": "memory", "payload": {"query": "test"}}
        )

        assert result["type"] == "error"
        assert "DB connection lost" in result["payload"]["message"]


# ── Session tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSession:
    async def test_session_new_returns_new_thread_id(self, router):
        """Starting a new session returns a fresh thread_id."""
        old_thread_id = router._thread_id

        result = await router.route(
            {"type": "session", "payload": {"action": "new"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "new"
        assert result["payload"]["thread_id"] != old_thread_id
        assert result["payload"]["thread_id"].startswith("ws-")
        # Router's internal state should be updated
        assert router._thread_id == result["payload"]["thread_id"]

    async def test_session_new_default_action(self, router):
        """Default action for session is 'new'."""
        old_thread_id = router._thread_id

        result = await router.route(
            {"type": "session", "payload": {}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "new"
        assert router._thread_id != old_thread_id

    async def test_session_new_affects_chat_thread_id(self, router, mock_agent):
        """After /new, subsequent chat uses the new thread_id."""
        from langchain_core.messages import AIMessage

        # Start a new session
        session_result = await router.route(
            {"type": "session", "payload": {"action": "new"}}
        )
        new_thread_id = session_result["payload"]["thread_id"]

        # Chat after new session
        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Fresh start!")],
        }
        await router.route({"type": "chat", "payload": {"text": "Hello"}})

        # Verify agent was called with the new thread_id
        call_args = mock_agent.ainvoke.call_args
        assert call_args[0][1]["configurable"]["thread_id"] == new_thread_id

    async def test_session_unknown_action_returns_error(self, router):
        """Unknown session action returns an error."""
        result = await router.route(
            {"type": "session", "payload": {"action": "bogus"}}
        )

        assert result["type"] == "error"
        assert "Unknown session action" in result["payload"]["message"]

    async def test_session_switch_valid_thread_id(self, router, mock_db):
        """Switch to an existing thread_id updates internal state."""
        # Mock: the thread exists
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_db.conn.execute = AsyncMock(return_value=mock_cursor)

        result = await router.route(
            {"type": "session", "payload": {"action": "switch", "thread_id": "ws-existing"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "switch"
        assert result["payload"]["thread_id"] == "ws-existing"
        assert router._thread_id == "ws-existing"

    async def test_session_switch_missing_thread_id_returns_error(self, router):
        """Switch without thread_id returns error."""
        result = await router.route(
            {"type": "session", "payload": {"action": "switch"}}
        )

        assert result["type"] == "error"
        assert "Missing thread_id" in result["payload"]["message"]

    async def test_session_switch_nonexistent_thread_returns_error(self, router, mock_db):
        """Switch to a thread_id that doesn't exist returns error."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(0,))
        mock_db.conn.execute = AsyncMock(return_value=mock_cursor)

        result = await router.route(
            {"type": "session", "payload": {"action": "switch", "thread_id": "ws-nonexistent"}}
        )

        assert result["type"] == "error"
        assert "not found" in result["payload"]["message"]

    async def test_session_delete_returns_error_when_current(self, router, mock_db):
        """Cannot delete the currently active session."""
        current_tid = router._thread_id
        # Mock: the thread exists
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_db.conn.execute = AsyncMock(return_value=mock_cursor)

        result = await router.route(
            {"type": "session", "payload": {"action": "delete", "thread_id": current_tid}}
        )

        assert result["type"] == "error"
        assert "Cannot delete current session" in result["payload"]["message"]

    async def test_session_delete_missing_thread_id_returns_error(self, router):
        """Delete without thread_id returns error."""
        result = await router.route(
            {"type": "session", "payload": {"action": "delete"}}
        )

        assert result["type"] == "error"
        assert "Missing thread_id" in result["payload"]["message"]

    async def test_session_list_returns_sessions(self, router, mock_db):
        """List returns all distinct thread_ids from checkpoints table,
        ordered by last activity (MAX(checkpoint_id) DESC)."""
        mock_cursor_list = MagicMock()
        # Now returns 2 columns: thread_id, MAX(checkpoint_id) as last_cp
        mock_cursor_list.fetchall = AsyncMock(return_value=[
            ("ws-abc123", "cp-latest-abc"),
            ("ws-def456", "cp-latest-def"),
        ])
        mock_cursor_count = MagicMock()
        mock_cursor_count.fetchone = AsyncMock(return_value=(3,))

        # First call: SELECT thread_id, MAX(checkpoint_id) → mock_cursor_list
        # Subsequent calls: SELECT COUNT(*) → mock_cursor_count
        call_count = 0

        async def _execute(sql, params=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return mock_cursor_list
            else:
                return mock_cursor_count

        mock_db.conn.execute = _execute

        result = await router.route(
            {"type": "session", "payload": {"action": "list"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "list"
        assert len(result["payload"]["sessions"]) == 2
        sessions = result["payload"]["sessions"]
        assert sessions[0]["thread_id"] == "ws-abc123"
        assert sessions[1]["thread_id"] == "ws-def456"
        assert sessions[0]["message_count"] == 2
        assert sessions[1]["message_count"] == 2
        # last_activity is exposed for future UI use (sorting, display)
        assert sessions[0]["last_activity"] == "cp-latest-abc"
        assert sessions[1]["last_activity"] == "cp-latest-def"

    async def test_session_current_returns_info(self, router, mock_db):
        """Current returns info for the active thread_id."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(3,))
        mock_db.conn.execute = AsyncMock(return_value=mock_cursor)

        result = await router.route(
            {"type": "session", "payload": {"action": "current"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "current"
        assert result["payload"]["session"]["thread_id"] == router._thread_id
        assert result["payload"]["session"]["is_current"] is True

    async def test_session_history_returns_messages(self, router, mock_agent):
        """History action loads messages from agent state."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="你好呀~"),
                HumanMessage(content="今天怎么样"),
                AIMessage(content="还不错呢"),
            ]
        }
        mock_agent.aget_state = AsyncMock(return_value=mock_state)

        result = await router.route(
            {"type": "session", "payload": {"action": "history"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["action"] == "history"
        msgs = result["payload"]["messages"]
        assert len(msgs) == 4
        assert msgs[0] == {"role": "user", "content": "你好"}
        assert msgs[1] == {"role": "her", "content": "你好呀~"}
        assert msgs[2] == {"role": "user", "content": "今天怎么样"}
        assert msgs[3] == {"role": "her", "content": "还不错呢"}

    async def test_session_history_filters_system_and_tool_messages(self, router, mock_agent):
        """History skips SystemMessage and ToolMessage, returns only user + her."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                SystemMessage(content="system prompt"),
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
                ToolMessage(content="tool result", tool_call_id="tc1"),
                HumanMessage(content="thanks"),
                AIMessage(content="np"),
            ]
        }
        mock_agent.aget_state = AsyncMock(return_value=mock_state)

        result = await router.route(
            {"type": "session", "payload": {"action": "history"}}
        )

        msgs = result["payload"]["messages"]
        assert len(msgs) == 4
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "her", "user", "her"]
        assert msgs[0]["content"] == "hi"
        assert msgs[-1]["content"] == "np"

    async def test_session_history_empty_state(self, router, mock_agent):
        """History returns empty list when agent has no state."""
        mock_agent.aget_state = AsyncMock(return_value=None)

        result = await router.route(
            {"type": "session", "payload": {"action": "history"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["messages"] == []

    async def test_session_history_no_agent_returns_empty(self, router_no_agent):
        """History returns empty list when agent is None."""
        result = await router_no_agent.route(
            {"type": "session", "payload": {"action": "history"}}
        )

        assert result["type"] == "session_response"
        assert result["payload"]["messages"] == []

    async def test_session_history_with_explicit_thread_id(self, router, mock_agent):
        """History uses explicit thread_id when provided."""
        from langchain_core.messages import HumanMessage

        mock_state = MagicMock()
        mock_state.values = {"messages": [HumanMessage(content="test")]}
        mock_agent.aget_state = AsyncMock(return_value=mock_state)

        result = await router.route(
            {"type": "session", "payload": {"action": "history", "thread_id": "ws-special"}}
        )

        assert result["payload"]["thread_id"] == "ws-special"
        mock_agent.aget_state.assert_called_once_with(
            {"configurable": {"thread_id": "ws-special"}}
        )

    # ── Persistence tests ─────────────────────────────────────────

    async def test_session_new_persists_thread_id(self, router, session_service, tmp_path):
        """new action writes the thread_id to current_session.txt."""
        result = await router.route(
            {"type": "session", "payload": {"action": "new"}}
        )
        new_tid = result["payload"]["thread_id"]

        persisted = session_service._current_session_file.read_text(encoding="utf-8")
        assert persisted == new_tid

    async def test_session_switch_persists_thread_id(
        self, router, mock_db, session_service, tmp_path
    ):
        """switch action writes the new thread_id to current_session.txt."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_db.conn.execute = AsyncMock(return_value=mock_cursor)

        await router.route(
            {"type": "session", "payload": {"action": "switch", "thread_id": "ws-persist"}}
        )

        persisted = session_service._current_session_file.read_text(encoding="utf-8")
        assert persisted == "ws-persist"


# ── Init / persistence (sync tests — no asyncio mark) ─────────────


class TestRouterInit:
    def test_router_loads_persisted_thread_id_on_init(
        self, mock_engine, mock_memory, mock_db, mock_agent, tmp_path
    ):
        """SessionService restores the persisted thread_id if present."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "current_session.txt").write_text("ws-restored", encoding="utf-8")

        from lingya.gateway.session_service import SessionService
        s = SessionService(db=mock_db, data_dir=str(data_dir))

        assert s.thread_id == "ws-restored"

    def test_router_falls_back_to_default_when_no_persisted_session(
        self, mock_engine, mock_memory, mock_db, mock_agent, tmp_path
    ):
        """SessionService uses the constructor default if no persisted file."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        from lingya.gateway.session_service import SessionService
        s = SessionService(db=mock_db, data_dir=str(data_dir), thread_id="ws-fallback")

        assert s.thread_id == "ws-fallback"
