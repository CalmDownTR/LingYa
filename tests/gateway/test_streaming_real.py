"""Streaming tests with real LLM — covers the full pipeline:
agent → astream_events(v3) → stream.extensions["lingya_inner"] → event_dict.

Uses real DeepSeek API. Marked @pytest.mark.slow + @pytest.mark.e2e
for optional CI filtering.

This is intentionally separate from test_router_streaming.py (which uses
mock ProtocolEvent dicts) — those tests are fast unit tests for the
parsing layer; these are behaviour-level tests for the full stream.

Replaces the old mock-based tests for:
- LingYa inner event forwarding (now tested end-to-end with real
  LingYaInnerProcessTransformer via LingYaStreamMiddleware)
- Text delta accumulation
- mind.transition emission
- Final chat_response shape
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest


def _skip_if_no_api_key():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set — skipping real-LLM test")


@pytest.mark.slow
@pytest.mark.e2e
class TestStreamingWithRealLLM:
    """Streaming tests driven by a real LLM — no ProtocolEvent mocks.

    Each test builds the full Application via ApplicationBuilder,
    creates a ChatHandler, and drives a single-turn conversation
    through astream_events(version="v3") with the real LingYa
    middleware pipeline.
    """

    @pytest.fixture(autouse=True)
    def _check_api_key(self):
        _skip_if_no_api_key()

    @staticmethod
    async def _build_handler(tmp_path):
        """Build a real Application and ChatHandler for one test."""
        from lingya.app import ApplicationBuilder
        from lingya.config import Config
        from lingya.gateway.chat_handler import ChatHandler
        from lingya.gateway.session_service import SessionService
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        (tmp_path / "data").mkdir(exist_ok=True)

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-streaming", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity=(
                    "你是 LingYa，一个友好的 AI 助手。"
                    "请用中文回答问题，保持简洁。"
                ),
                core_belief="帮助用户是最重要的事。",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["保持诚实。"],
        )

        app = await (
            ApplicationBuilder(config, mind_config)
            .with_database()
            .with_model()
            .with_memory()
            .with_event_bus()
            .with_engine()
            .with_agent()
            .build()
        )

        session_svc = SessionService(db=app.db, data_dir=data_dir)
        session_svc.set_agent(app.agent)
        handler = ChatHandler(
            engine=app.engine,
            agent=app.agent,
            session_service=session_svc,
        )

        return app, handler

    # ── Core streaming behaviour ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_yields_all_event_types(self, tmp_path):
        """A single chat round should emit process.phase, chat.delta,
        mind.transition, and chat_response — in that relative order."""
        app, handler = await self._build_handler(tmp_path)
        try:
            events: list[dict] = []

            async def emit(event_dict):
                events.append(event_dict)

            start = time.monotonic()
            result = await handler.handle_chat(
                {"text": "用一句话介绍一下自己。"}, emit=emit,
            )
            elapsed = time.monotonic() - start

            # ─── Final response ──────────────────────────────────────────
            assert result["type"] == "chat_response", (
                f"Expected chat_response, got {result.get('type')}: "
                f"{result.get('payload', {}).get('message', '')}"
            )
            assert result["payload"]["text"], "Final text must not be empty"
            assert "tone" in result["payload"]

            # ─── Event types ─────────────────────────────────────────────
            event_kinds = [e.get("event") for e in events if e.get("type") == "event"]
            assert "chat.delta" in event_kinds, f"Missing chat.delta in {event_kinds}"
            assert "process.phase" in event_kinds, f"Missing process.phase in {event_kinds}"
            assert "mind.transition" in event_kinds, f"Missing mind.transition in {event_kinds}"

            # ─── Order: phases before deltas before transition ──
            chat_delta_indices = [
                i for i, e in enumerate(events)
                if e.get("type") == "event" and e.get("event") == "chat.delta"
            ]
            phase_indices = [
                i for i, e in enumerate(events)
                if e.get("type") == "event" and e.get("event") == "process.phase"
            ]
            mind_indices = [
                i for i, e in enumerate(events)
                if e.get("type") == "event" and e.get("event") == "mind.transition"
            ]
            if phase_indices and chat_delta_indices:
                assert phase_indices[0] < chat_delta_indices[-1] + 1, (
                    f"First process.phase ({phase_indices[0]}) should be "
                    f"before last chat.delta ({chat_delta_indices[-1]})"
                )
            if mind_indices and chat_delta_indices:
                assert mind_indices[0] > chat_delta_indices[0], (
                    f"mind.transition ({mind_indices[0]}) should come after "
                    f"first chat.delta ({chat_delta_indices[0]})"
                )

            # ─── Timing — should complete within 60s ─────────────────────
            assert elapsed < 60, f"Stream took {elapsed:.1f}s — too slow"
        finally:
            await app.teardown()

    @pytest.mark.asyncio
    async def test_text_accumulates(self, tmp_path):
        """All chat.delta chunks should concatenate to the final text."""
        app, handler = await self._build_handler(tmp_path)
        try:
            deltas: list[str] = []

            async def emit(event_dict):
                if event_dict.get("event") == "chat.delta":
                    deltas.append(event_dict["payload"]["content"])

            result = await handler.handle_chat(
                {"text": "说'你好世界'，不要加任何其他内容。"}, emit=emit,
            )

            accumulated = "".join(deltas)
            final = result["payload"]["text"]
            assert accumulated, "Should have at least one text delta"
            assert accumulated.strip() == final.strip(), (
                f"Accumulated deltas ({accumulated!r}) != final text ({final!r})"
            )
        finally:
            await app.teardown()

    @pytest.mark.asyncio
    async def test_mind_transition_has_required_fields(self, tmp_path):
        """mind.transition event must include pad, occ_emotion, ipc."""
        app, handler = await self._build_handler(tmp_path)
        try:
            transitions: list[dict] = []

            async def emit(event_dict):
                if event_dict.get("event") == "mind.transition":
                    transitions.append(event_dict["payload"])

            await handler.handle_chat(
                {"text": "你好"}, emit=emit,
            )

            assert len(transitions) == 1, f"Expected 1 mind.transition, got {len(transitions)}"
            t = transitions[0]
            assert "pad" in t
            assert "pleasure" in t["pad"]
            assert "arousal" in t["pad"]
            assert "dominance" in t["pad"]
            assert "occ_emotion" in t
            assert "ipc" in t
        finally:
            await app.teardown()

    @pytest.mark.asyncio
    async def test_chat_handler_errors(self, tmp_path):
        """Empty text and None agent should return errors (no LLM needed)."""
        app, handler = await self._build_handler(tmp_path)
        try:
            # Empty text
            result = await handler.handle_chat({"text": ""}, emit=None)
            assert result["type"] == "error"

            # No agent
            from lingya.gateway.chat_handler import ChatHandler
            no_agent_handler = ChatHandler(
                engine=handler._engine,
                agent=None,
                session_service=handler._session_service,
            )
            result = await no_agent_handler.handle_chat({"text": "hello"}, emit=None)
            assert result["type"] == "error"
            assert "Agent not initialized" in result["payload"]["message"]
        finally:
            await app.teardown()


@pytest.mark.slow
@pytest.mark.e2e
class TestStreamExtensionsWithRealLLM:
    """Directly verify stream.extensions["lingya_inner"] is iterable
    and yields LingYa domain events (process.phase, memory.recall)
    when using real LingYaStreamMiddleware + real LLM."""

    @pytest.fixture(autouse=True)
    def _check_api_key(self):
        _skip_if_no_api_key()

    @pytest.mark.asyncio
    async def test_extensions_lingya_inner_yields_events(self, tmp_path):
        """stream.extensions["lingya_inner"] must yield typed events
        from the real LingYaInnerProcessTransformer pipeline."""
        from lingya.app import ApplicationBuilder
        from lingya.config import Config
        from lingya.mind.config import (
            BigFiveTraits,
            IdentityAnchor,
            MindConfig,
            PersonaMeta,
            ToneMatrix,
        )

        db_path = str(tmp_path / "lingya.db")
        memory_path = str(tmp_path / "memory")
        data_dir = str(tmp_path / "data")
        (tmp_path / "data").mkdir(exist_ok=True)

        config = Config(
            db_path=db_path,
            memory_path=memory_path,
            data_dir=data_dir,
        )
        mind_config = MindConfig(
            version="1.0",
            meta=PersonaMeta(agent_id="test-ext", created_at="2025-01-01"),
            identity=IdentityAnchor(
                identity="你是 LingYa，请简洁回答。",
                core_belief="帮助用户。",
            ),
            ocean=BigFiveTraits(),
            tone_matrix=ToneMatrix(),
            behavior_guardrails=["保持诚实。"],
        )

        app = await (
            ApplicationBuilder(config, mind_config)
            .with_database()
            .with_model()
            .with_memory()
            .with_event_bus()
            .with_engine()
            .with_agent()
            .build()
        )

        try:
            import uuid

            config_dict = {"configurable": {"thread_id": f"test-ext-{uuid.uuid4().hex[:8]}"}}
            run = await app.agent.astream_events(
                {"messages": [
                    {"role": "user", "content": "说一个词：你好"}
                ]},
                config_dict,
                version="v3",
            )

            # Concurrent consumers: the AsyncGraphRunStream pump is driven
            # by any consumer iterating. Use asyncio.gather so both the
            # main event log (to drive the pump) and the extensions channel
            # drain concurrently.
            ext_events: list[dict] = []

            async def _consume_extensions() -> None:
                async for item in run.extensions["lingya_inner"]:
                    ext_events.append(item)

            async def _consume_main() -> None:
                async for _event in run:
                    pass  # drive the pump

            await asyncio.gather(_consume_extensions(), _consume_main())

            assert len(ext_events) > 0, (
                "stream.extensions['lingya_inner'] should yield at least one event"
            )

            event_types = {e["type"] for e in ext_events}
            assert "process.phase" in event_types, (
                f"Expected process.phase in ext events, got {event_types}"
            )

        finally:
            await app.teardown()
