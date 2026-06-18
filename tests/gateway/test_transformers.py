"""Test LingYaInnerProcessTransformer — unit tests for custom stream transformer."""

from __future__ import annotations


from unittest.mock import MagicMock

from langgraph.stream.stream_channel import StreamChannel


# ── Helpers ───────────────────────────────────────────────────────────


def _make_transformer_with_spy(scope=()):
    """Create a transformer and replace its channel with a MagicMock for inspection."""
    from lingya.transformers import LingYaInnerProcessTransformer

    t = LingYaInnerProcessTransformer(scope=scope)
    t._channel = MagicMock()
    return t


def _make_event(method: str, namespace: list[str], data: object = None) -> dict:
    """Create a minimal ProtocolEvent for testing."""
    return {
        "type": "event",
        "eventId": "evt-001",
        "seq": 1,
        "method": method,
        "params": {
            "namespace": namespace,
            "timestamp": 1000,
            "data": data,
            "interrupts": (),
        },
    }


def _make_tasks_event(node_name: str, scope: list[str] | None = None) -> dict:
    """Create a tasks ProtocolEvent for a node starting."""
    ns = (scope or []) + [f"{node_name}:task1"]
    return _make_event("tasks", ns, data={})


def _make_content_block_start(
    block_type: str,
    name: str = "",
    block_id: str = "cb-1",
    scope: list[str] | None = None,
) -> dict:
    """Create a messages ProtocolEvent for a content-block-start."""
    msg_payload = {
        "event": "content-block-start",
        "content": {"type": block_type, "name": name, "id": block_id},
    }
    return _make_event(
        "messages",
        scope or [],
        data=(msg_payload, {"some": "metadata"}),
    )


def _make_content_block_end(
    block_type: str,
    block_id: str = "cb-1",
    scope: list[str] | None = None,
) -> dict:
    """Create a messages ProtocolEvent for a content-block-finish."""
    msg_payload = {
        "event": "content-block-finish",
        "content": {"type": block_type, "id": block_id},
    }
    return _make_event(
        "messages",
        scope or [],
        data=(msg_payload, {"some": "metadata"}),
    )


def _make_text_delta(text: str, scope: list[str] | None = None) -> dict:
    """Create a messages ProtocolEvent for a text delta."""
    msg_payload = {
        "event": "content-block-delta",
        "delta": {"type": "text-delta", "text": text},
    }
    return _make_event(
        "messages",
        scope or [],
        data=(msg_payload, {"some": "metadata"}),
    )


def _collect_channel_items(channel: StreamChannel) -> list:
    """Drain all items from a StreamChannel into a list."""
    items = []
    while True:
        try:
            items.append(channel.push_buffer.get_nowait())
        except Exception:
            break
    return items


def _pushed_items(spy_channel) -> list:
    """Extract the positional args from MagicMock push() calls."""
    return [call[0][0] for call in spy_channel.push.call_args_list]


# ── Tests ─────────────────────────────────────────────────────────────


class TestTransformerInit:
    """Verify the transformer's init() method returns the correct projection."""

    def test_init_returns_lingya_inner_channel(self):
        """init() should return a dict with 'lingya_inner' key mapping to a StreamChannel."""
        from lingya.transformers import LingYaInnerProcessTransformer

        t = LingYaInnerProcessTransformer()
        result = t.init()

        assert isinstance(result, dict)
        assert "lingya_inner" in result
        assert isinstance(result["lingya_inner"], StreamChannel)


class TestRequiredStreamModes:
    """Verify the transformer declares its required stream modes."""

    def test_required_stream_modes(self):
        from lingya.transformers import LingYaInnerProcessTransformer

        assert LingYaInnerProcessTransformer.required_stream_modes == (
            "messages",
            "tasks",
        )


class TestTransformerIsNative:
    """Verify _native flag is set so projections appear as direct attributes."""

    def test_native_is_true(self):
        from lingya.transformers import LingYaInnerProcessTransformer

        assert LingYaInnerProcessTransformer._native is True


class TestTransformerScope:
    """Verify scope filtering — only events matching the transformer's scope are processed."""

    def test_matching_scope_processed(self):
        """Events with namespace matching scope should be processed."""
        t = _make_transformer_with_spy(scope=("agent",))
        event = _make_tasks_event("agent_model", scope=["agent"])
        result = t.process(event)
        assert result is True
        # Phase should have been emitted
        items = _pushed_items(t._channel)
        assert any(i["type"] == "process.phase" and i["payload"]["phase"] == "thinking" for i in items)

    def test_non_matching_scope_passed_through(self):
        """Events with non-matching namespace should pass through unprocessed."""
        t = _make_transformer_with_spy(scope=("agent",))
        event = _make_tasks_event("agent_model", scope=["other"])
        result = t.process(event)
        assert result is True
        # No phase should have been emitted since scope doesn't match
        assert t._channel.push.call_count == 0
        assert t._current_phase is None  # stays at default (None)

    def test_root_scope_matches_all(self):
        """Empty scope () matches all namespaces."""
        t = _make_transformer_with_spy(scope=())
        event = _make_tasks_event("agent_model", scope=[])
        result = t.process(event)
        assert result is True
        assert t._channel.push.call_count >= 1


class TestProcessPhaseEvents:
    """Verify process.phase events are emitted correctly from tasks events."""

    def test_agent_model_start_emits_thinking(self):
        """When agent_model node starts, phase should be 'thinking'."""
        t = _make_transformer_with_spy()
        t.process(_make_tasks_event("agent_model"))
        items = _pushed_items(t._channel)
        assert any(
            i["type"] == "process.phase" and i["payload"]["phase"] == "thinking"
            for i in items
        )

    def test_tools_start_emits_recalling(self):
        """When tools node starts, phase should be 'recalling'."""
        t = _make_transformer_with_spy()
        t.process(_make_tasks_event("tools"))
        items = _pushed_items(t._channel)
        assert any(
            i["type"] == "process.phase" and i["payload"]["phase"] == "recalling"
            for i in items
        )

    def test_duplicate_phase_not_emitted(self):
        """Same phase should not be emitted twice."""
        t = _make_transformer_with_spy()
        t.process(_make_tasks_event("agent_model"))
        t.process(_make_tasks_event("agent_model"))
        items = _pushed_items(t._channel)
        phase_events = [i for i in items if i.get("type") == "process.phase"]
        assert len(phase_events) == 1

    def test_text_delta_transitions_to_generating(self):
        """First text_delta should transition phase to 'generating'."""
        t = _make_transformer_with_spy()
        t._current_phase = "thinking"
        t.process(_make_text_delta("Hello"))
        items = _pushed_items(t._channel)
        assert any(
            i["type"] == "process.phase" and i["payload"]["phase"] == "generating"
            for i in items
        )


class TestMemoryRecallEvents:
    """Verify memory.recall events are emitted correctly from messages events."""

    def test_memory_tool_start_increments_count(self):
        """When a memory tool call starts, count increments."""
        from lingya.transformers import LingYaInnerProcessTransformer

        t = LingYaInnerProcessTransformer()
        t.process(_make_content_block_start("tool_call", name="memory_search"))
        t.process(_make_content_block_start("tool_call", name="memory_store"))
        assert t._memory_tool_count == 2

    def test_non_memory_tool_start_ignored(self):
        """Non-memory tool calls should not affect count."""
        from lingya.transformers import LingYaInnerProcessTransformer

        t = LingYaInnerProcessTransformer()
        t.process(_make_content_block_start("tool_call", name="get_weather"))
        assert t._memory_tool_count == 0

    def test_tool_call_end_emits_memory_recall(self):
        """When a memory tool call ends, emit memory.recall event."""
        t = _make_transformer_with_spy()
        t.process(_make_content_block_start("tool_call", name="memory_search", block_id="cb-1"))
        t.process(_make_text_delta("User likes coffee"))
        t.process(_make_content_block_end("tool_call", block_id="cb-1"))
        items = _pushed_items(t._channel)
        recall_events = [i for i in items if i.get("type") == "memory.recall"]
        assert len(recall_events) >= 1
        assert recall_events[0]["payload"]["count"] >= 1

    def test_multiple_tool_calls_aggregated(self):
        """Multiple memory tool calls should be counted together."""
        t = _make_transformer_with_spy()
        t.process(_make_content_block_start("tool_call", name="memory_search", block_id="cb-1"))
        t.process(_make_content_block_end("tool_call", block_id="cb-1"))
        t.process(_make_content_block_start("tool_call", name="memory_search", block_id="cb-2"))
        t.process(_make_content_block_end("tool_call", block_id="cb-2"))
        items = _pushed_items(t._channel)
        recall_events = [i for i in items if i.get("type") == "memory.recall"]
        assert len(recall_events) == 2


class TestTransformerFinalize:
    """Verify finalize() cleans up internal state."""

    def test_finalize_clears_pending_tool_calls(self):
        from lingya.transformers import LingYaInnerProcessTransformer

        t = LingYaInnerProcessTransformer()
        t._pending_tool_calls["cb-1"] = "memory_search"
        t._pending_tool_calls["cb-2"] = "memory_store"
        t.finalize()
        assert t._pending_tool_calls == {}
