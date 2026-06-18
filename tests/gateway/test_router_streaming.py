"""Test MessageRouter streaming — _handle_chat with emit callback and astream_events."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def _make_text_stream_event(text: str, seq: int = 1) -> dict:
    """Create a ProtocolEvent for a text-delta in messages stream."""
    return {
        "type": "event",
        "eventId": f"evt-{seq}",
        "seq": seq,
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": 1000 + seq,
            "data": (
                {
                    "event": "content-block-delta",
                    "delta": {"type": "text-delta", "text": text},
                },
                {},
            ),
            "interrupts": (),
        },
    }


def _make_lingya_inner_event(inner_type: str, payload: dict, seq: int = 1) -> dict:
    """Create a ProtocolEvent for a lingya_inner (transformer) event."""
    return {
        "type": "event",
        "eventId": f"evt-{seq}",
        "seq": seq,
        "method": "lingya_inner",
        "params": {
            "namespace": [],
            "timestamp": 1000 + seq,
            "data": {"type": inner_type, "payload": payload},
            "interrupts": (),
        },
    }


async def _collect(emit_calls: list) -> list[dict]:
    """Convert mock emit calls to a flat list of event dicts."""
    result = []
    for c in emit_calls:
        # Each call is (positional_args, keyword_args)
        # emit was called with a single positional arg (the event dict)
        result.append(c[0][0])
    return result


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestHandleChatStreaming:
    """Tests for _handle_chat when emit callback is provided."""

    @staticmethod
    async def _make_async_iterable(events: list[dict]):
        """Create an async generator from a list of events."""
        for event in events:
            yield event

    async def test_emits_chat_delta_events(self, router, mock_agent, mock_engine):
        """Text deltas from messages stream should emit chat.delta events."""
        events = [
            _make_text_stream_event("Hello", seq=1),
            _make_text_stream_event(" world", seq=2),
        ]
        mock_agent.astream_events.return_value = self._make_async_iterable(events)

        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        result = await router._handle_chat(
            {"text": "Hi"}, emit=emit
        )

        # Check chat.delta events
        deltas = [e for e in emitted if e.get("event") == "chat.delta"]
        assert len(deltas) == 2
        assert deltas[0]["payload"]["content"] == "Hello"
        assert deltas[1]["payload"]["content"] == " world"

        # Check final response
        assert result["type"] == "chat_response"
        assert "Hello world" in result["payload"]["text"]

    async def test_emits_lingya_inner_events(self, router, mock_agent, mock_engine):
        """Lingya inner events from transformer should be forwarded as-is."""
        events = [
            _make_lingya_inner_event("process.phase", {"phase": "thinking"}, seq=1),
            _make_lingya_inner_event("process.phase", {"phase": "recalling"}, seq=2),
            _make_lingya_inner_event("memory.recall", {"count": 2, "top_match": "coffee"}, seq=3),
            _make_text_stream_event("I remember", seq=4),
        ]
        mock_agent.astream_events.return_value = self._make_async_iterable(events)

        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        await router._handle_chat(
            {"text": "What do you remember?"}, emit=emit
        )

        inner_events = [e for e in emitted if e.get("event") not in ("chat.delta", "mind.transition")]
        assert len(inner_events) == 3
        assert inner_events[0]["event"] == "process.phase"
        assert inner_events[0]["payload"]["phase"] == "thinking"
        assert inner_events[2]["event"] == "memory.recall"
        assert inner_events[2]["payload"]["count"] == 2

    async def test_emits_mind_transition_after_stream(self, router, mock_agent, mock_engine):
        """After stream completes, mind.transition should be emitted with engine state."""
        events = [_make_text_stream_event("Hello", seq=1)]
        mock_agent.astream_events.return_value = self._make_async_iterable(events)

        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        await router._handle_chat(
            {"text": "Hi"}, emit=emit
        )

        # Last emitted event should be mind.transition
        mind_events = [e for e in emitted if e.get("event") == "mind.transition"]
        assert len(mind_events) == 1
        mind_event = mind_events[0]
        assert "pad" in mind_event["payload"]
        assert "occ_emotion" in mind_event["payload"]
        assert "ipc" in mind_event["payload"]

        # Check engine was called
        mock_engine.process_event.assert_called_once()

    async def test_calls_agent_with_correct_config(self, router, mock_agent, mock_engine):
        """Streaming should pass the correct config to astream_events."""
        events = [_make_text_stream_event("OK", seq=1)]
        mock_agent.astream_events.return_value = self._make_async_iterable(events)

        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        await router._handle_chat({"text": "Hey"}, emit=emit)

        # Verify astream_events was called with correct args
        mock_agent.astream_events.assert_called_once()
        call_args = mock_agent.astream_events.call_args
        # First positional arg: input dict
        input_dict = call_args[0][0]
        assert "messages" in input_dict
        # Second positional arg: config
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == "ws-default"
        # version kwarg
        assert call_args[1].get("version") == "v3"

    async def test_final_response_includes_accumulated_text(self, router, mock_agent, mock_engine):
        """Final chat_response should contain the full accumulated text."""
        events = [
            _make_text_stream_event("你", seq=1),
            _make_text_stream_event("好", seq=2),
            _make_text_stream_event("！", seq=3),
        ]
        mock_agent.astream_events.return_value = self._make_async_iterable(events)

        result = await router._handle_chat(
            {"text": "Hello"}, emit=AsyncMock()
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "你好！"
        assert "tone" in result["payload"]
        assert "meta" in result["payload"]


@pytest.mark.asyncio
class TestHandleChatFallback:
    """Tests for _handle_chat when emit is None (backward compat)."""

    async def test_fallback_uses_ainvoke(self, router, mock_agent, mock_engine):
        """With emit=None, should use agent.ainvoke() not astream_events."""
        from langchain_core.messages import AIMessage

        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Hello!")],
        }

        result = await router._handle_chat({"text": "Hi"}, emit=None)

        mock_agent.ainvoke.assert_called_once()
        assert not mock_agent.astream_events.called
        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Hello!"

    async def test_fallback_passes_through_to_route(self, router, mock_agent, mock_engine):
        """The route() method passes emit=None, so existing tests don't break."""
        from langchain_core.messages import AIMessage

        mock_agent.ainvoke.return_value = {
            "messages": [AIMessage(content="Response")],
        }

        result = await router.route(
            {"type": "chat", "payload": {"text": "Hello"}}
        )

        assert result["type"] == "chat_response"
        assert result["payload"]["text"] == "Response"


@pytest.mark.asyncio
class TestHandleChatEdgeCases:
    """Edge case tests for _handle_chat with streaming."""

    async def test_empty_text_with_emit_returns_error(self, router):
        """Empty text with emit should return error immediately."""
        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        result = await router._handle_chat({"text": ""}, emit=emit)
        assert result["type"] == "error"
        assert emitted == []

    async def test_no_agent_with_emit_returns_error(self, router_no_agent):
        """No agent with emit should return error."""
        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        result = await router_no_agent._handle_chat({"text": "Hello"}, emit=emit)
        assert result["type"] == "error"
        assert "Agent not initialized" in result["payload"]["message"]

    async def test_streaming_error_returns_error(self, router, mock_agent):
        """Agent stream exception should be caught and returned as error."""
        mock_agent.astream_events.side_effect = RuntimeError("LLM timeout")

        emitted = []
        async def emit(event_dict):
            emitted.append(event_dict)

        result = await router._handle_chat({"text": "Hello"}, emit=emit)
        assert result["type"] == "error"
        assert "LLM timeout" in result["payload"]["message"]
