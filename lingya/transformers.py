"""LingYaInnerProcessTransformer — injects process.phase and memory.recall
events into the LangGraph v3 stream event log.

Published on stream.extensions["lingya_inner"] and auto-forwarded to the
main event log as ProtocolEvent objects with method="lingya_inner".
"""

from __future__ import annotations

from typing import Any

from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel


class LingYaInnerProcessTransformer(StreamTransformer):
    """Custom stream transformer for LingYa inner process events.

    Observes the LangGraph v3 protocol event stream and injects two
    kinds of payloads into the ``lingya_inner`` channel:

    - ``process.phase``: emitted when the agent transitions between
      recalling (tools active), thinking (agent_model node started),
      and generating (LLM is producing tokens).
    - ``memory.recall``: emitted when a memory-store tool call starts
      or completes, carrying the count of memory tools invoked and
      the top-match search text.

    All payloads use this envelope::

        {"type": "process.phase", "payload": {"phase": "recalling"}}
        {"type": "memory.recall", "payload": {"count": 3, "top_match": "..."}}

    The channel is named, so the StreamMux auto-injects every push
    into the main event log as a ProtocolEvent with
    ``method="lingya_inner"``.
    """

    _native = True
    required_stream_modes = ("messages", "tasks")

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[dict[str, Any]] = StreamChannel("lingya_inner")
        self._scope_list: list[str] = list(scope)

        # Phase tracking — starts as None so the first detected phase always emits
        self._current_phase: str | None = None
        self._phase_emitted: set[str] = set()

        # Memory recall tracking
        self._memory_tool_count: int = 0
        self._top_match: str = ""
        self._pending_tool_calls: dict[str, str] = {}

    # -- StreamTransformer contract ---------------------------------

    def init(self) -> dict[str, Any]:
        return {"lingya_inner": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        """Route each protocol event to the appropriate handler."""
        ns = event["params"]["namespace"]

        # Root scope () matches all namespaces. Non-root scope checks prefix match.
        if self._scope_list and list(ns[:len(self._scope_list)]) != self._scope_list:
            return True  # Not our scope — keep event, don't process

        method = event["method"]
        if method == "tasks":
            self._handle_tasks(event)
        elif method == "messages":
            self._handle_messages(event)

        return True

    # -- Phase tracking via tasks events -----------------------------

    def _handle_tasks(self, event: ProtocolEvent) -> None:
        """Detect node transitions from tasks events."""
        data = event["params"]["data"]
        if not isinstance(data, dict) or "result" in data:
            return  # Task result — ignore

        ns = event["params"]["namespace"]
        if not ns:
            return

        # Should have at least one segment beyond the scope prefix
        if len(ns) <= len(self._scope_list):
            return

        last_segment = ns[-1]
        # Parse "node_name:task_id" format
        node_name = last_segment.split(":", 1)[0] if ":" in last_segment else last_segment

        if node_name == "agent_model":
            self._transition_phase("thinking")
        elif node_name == "tools":
            self._transition_phase("recalling")

    def _transition_phase(self, phase: str) -> None:
        """Emit process.phase if this is a new phase."""
        if phase == self._current_phase:
            return
        self._current_phase = phase
        self._channel.push({
            "type": "process.phase",
            "payload": {"phase": phase},
        })

    # -- Memory recall tracking via messages events ------------------

    def _handle_messages(self, event: ProtocolEvent) -> None:
        """Detect memory tool calls and text deltas from messages events."""
        payload, _metadata = event["params"]["data"]

        if not isinstance(payload, dict) or "event" not in payload:
            return

        msg_event = payload["event"]

        if msg_event == "content-block-start":
            content_block = payload.get("content") or payload.get("data", {})
            if content_block.get("type") == "tool_call":
                tool_name = content_block.get("name", "")
                tool_id = content_block.get("id", "")
                if tool_name in ("memory_store", "memory_search"):
                    self._pending_tool_calls[tool_id] = tool_name
                    self._memory_tool_count += 1

        elif msg_event == "content-block-delta":
            delta = payload.get("delta", payload.get("data", {}))
            if delta.get("type") == "text-delta":
                if self._current_phase != "generating":
                    self._transition_phase("generating")
                if self._pending_tool_calls and not self._top_match:
                    text = delta.get("text", "")
                    if text:
                        self._top_match = text

        elif msg_event in ("content-block-finish", "content-block-end"):
            content_block = payload.get("content") or payload.get("data", {})
            if isinstance(content_block, dict):
                tool_id = content_block.get("id")
                if tool_id and tool_id in self._pending_tool_calls:
                    self._channel.push({
                        "type": "memory.recall",
                        "payload": {
                            "count": self._memory_tool_count,
                            "top_match": self._top_match or "",
                        },
                    })
                    del self._pending_tool_calls[tool_id]

        elif msg_event == "message-finish":
            if not self._pending_tool_calls:
                self._memory_tool_count = 0
                self._top_match = ""

    def finalize(self) -> None:
        """Cleanup."""
        self._pending_tool_calls.clear()


def create_lingya_transformer(scope: tuple[str, ...] = ()) -> LingYaInnerProcessTransformer:
    """Factory function for StreamMux. Matches the TransformerFactory protocol.

    Usage::

        agent.astream_events(
            input, config, version="v3",
            transformers=[create_lingya_transformer],
        )
    """
    return LingYaInnerProcessTransformer(scope)
