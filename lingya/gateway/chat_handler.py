"""ChatHandler — streaming chat orchestration with MindEngine integration.

Extracted from MessageRouter (v0.9.5 router.py split).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from lingya.gateway.session_service import SessionService

logger = logging.getLogger(__name__)


class ChatHandler:
    """Orchestrates chat: streaming/invoke dispatch + mind.transition yield."""

    def __init__(
        self,
        engine: Any,
        agent: Any,
        session_service: SessionService,
    ) -> None:
        self._engine = engine
        self._agent = agent
        self._session_service = session_service

    async def handle_chat(
        self,
        payload: dict,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> dict:
        """Process a chat message through the agent + mind engine pipeline.

        When *emit* is provided, the agent runs via ``astream_events(version="v3")``
        and streaming events are pushed through *emit* as they happen.
        When *emit* is None (backward compat), falls back to ``agent.ainvoke()``.

        Returns the final ``chat_response`` dict.
        """
        text = payload.get("text", "")
        if not text:
            return {"type": "error", "payload": {"message": "Empty message"}}

        if self._agent is None:
            return {"type": "error", "payload": {"message": "Agent not initialized"}}

        # 1. Get dynamic tone fragment from engine
        fragment = self._engine.get_prompt_fragment()
        messages: list = [HumanMessage(content=text)]
        if fragment:
            messages.insert(0, SystemMessage(content=fragment))

        config = {"configurable": {"thread_id": self._session_service.thread_id}}

        if emit is not None:
            final = None
            async for event_dict in self._chat_streaming(messages, config, text):
                if event_dict.get("type") in ("chat_response", "error"):
                    final = event_dict
                else:
                    await emit(event_dict)
            return final
        else:
            return await self._chat_invoke(messages, config, text)

    async def _chat_streaming(
        self,
        messages: list,
        config: dict,
        user_text: str,
    ):
        """Run agent via astream_events and yield streaming events + final response."""
        from lingya.transformers import create_lingya_transformer

        accumulated_text = ""

        # Filter out _subagent_factory from compiled stream_transformers
        _saved_st = self._agent.stream_transformers
        self._agent.stream_transformers = tuple(
            t for t in _saved_st
            if not (callable(t) and getattr(t, "__name__", "") == "_subagent_factory")
        )

        engine_task: asyncio.Task | None = None

        try:
            run = await self._agent.astream_events(
                {"messages": messages},
                config,
                version="v3",
                transformers=[create_lingya_transformer],
            )

            # Start MindEngine processing concurrently — runs while LLM streams.
            # process_event does 1 LLM call (affect.py:_OCC_IPC_TIMEOUT=1.5s)
            # + DB save + event publish.
            engine_task = asyncio.create_task(
                self._engine.process_event({
                    "description": user_text,
                    "content": user_text,
                })
            )

            async for event in run:
                method = event["method"]

                if method == "messages":
                    payload_data, _metadata = event["params"]["data"]
                    if isinstance(payload_data, dict) and "event" in payload_data:
                        if payload_data["event"] == "content-block-delta":
                            delta = payload_data.get("delta", {})
                            if delta.get("type") == "text-delta":
                                chunk = delta.get("text", "")
                                accumulated_text += chunk
                                yield {
                                    "type": "event",
                                    "event": "chat.delta",
                                    "payload": {"content": chunk},
                                }

                elif method == "lingya_inner":
                    inner_event = event["params"]["data"]
                    yield {
                        "type": "event",
                        "event": inner_event["type"],
                        "payload": inner_event["payload"],
                    }

            # Wait for engine — 2.0s covers 1.5s LLM timeout + 0.5s persistence.
            try:
                await asyncio.wait_for(engine_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass

            # Fire-and-forget: response alignment check runs in background.
            if accumulated_text:
                asyncio.create_task(
                    self._engine.check_response_alignment(accumulated_text)
                )

            # v0.9.9: Record conversation turn for diary generation
            response_text = (
                accumulated_text
                if isinstance(accumulated_text, str)
                else self._session_service._extract_text_content_from_value(accumulated_text)
            )
            self._engine.record_conversation_turn(user_text, response_text)

            # Yield mind.transition
            pad = self._engine.state.current_pad
            last_emotion = (
                self._engine.state.recent_emotions[-1]
                if self._engine.state.recent_emotions
                else {"emotion": "neutral", "intensity": 0.0}
            )
            yield {
                "type": "event",
                "event": "mind.transition",
                "payload": {
                    "pad": {
                        "pleasure": pad.pleasure,
                        "arousal": pad.arousal,
                        "dominance": pad.dominance,
                    },
                    "occ_emotion": last_emotion["emotion"],
                    "ipc": f"{self._engine.state.ipc_state} (agency={self._engine.state.ipc_agency:.2f}, communion={self._engine.state.ipc_communion:.2f})",
                },
            }

            # Yield final response
            yield {
                "type": "chat_response",
                "payload": {
                    "text": (
                        accumulated_text
                        if isinstance(accumulated_text, str)
                        else self._session_service._extract_text_content_from_value(accumulated_text)
                    ),
                    "tone": self._engine.get_tone_params(),
                },
            }

        except RuntimeError as e:
            # "v2 stream finished without producing a message" — the underlying
            # LLM (e.g. DeepSeek) produced zero tokens. Usually transient: rate
            # limit, model overload, or a network hiccup mid-request.
            msg = str(e)
            logger.warning("_chat_streaming: streaming failed (%s)", msg)
            if engine_task is not None and not engine_task.done():
                engine_task.cancel()
            user_msg = (
                "模型暂时没有返回结果，请稍后重试。"
                if "without producing a message" in msg
                else f"流式响应中断：{msg}"
            )
            yield {"type": "error", "payload": {"message": user_msg}}
        except Exception as e:
            # Classify transient errors for user-friendly messages.
            # LiteLLM MidStreamFallbackError / APIConnectionError = TCP drop
            # mid-stream (Bad file descriptor, Connection reset, etc.).
            cls_name = type(e).__name__
            cls_module = type(e).__module__
            msg = str(e)
            if "litellm" in cls_module or "MidStreamFallback" in cls_name or "APIConnection" in cls_name:
                logger.warning("_chat_streaming: LiteLLM stream error (%s: %s)", cls_name, msg[:200])
                if engine_task is not None and not engine_task.done():
                    engine_task.cancel()
                yield {"type": "error", "payload": {"message": "模型连接中断，请稍后重试。"}}
            elif "Bad file descriptor" in msg or "Connection" in cls_name:
                logger.warning("_chat_streaming: connection dropped (%s)", msg[:200])
                if engine_task is not None and not engine_task.done():
                    engine_task.cancel()
                yield {"type": "error", "payload": {"message": "网络连接中断，请稍后重试。"}}
            else:
                logger.exception("_chat_streaming failed")
                if engine_task is not None and not engine_task.done():
                    engine_task.cancel()
                yield {"type": "error", "payload": {"message": str(e)}}
        finally:
            self._agent.stream_transformers = _saved_st

    async def _chat_invoke(
        self,
        messages: list,
        config: dict,
        user_text: str,
    ) -> dict:
        """Fallback path: agent.ainvoke() for backward compatibility."""
        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                config,
            )
        except Exception as e:
            return {"type": "error", "payload": {"message": str(e)}}

        msgs = result.get("messages", [])
        ais = [m for m in msgs if isinstance(m, AIMessage)]
        response_text = (
            self._session_service._extract_text_content(ais[-1]) if ais else ""
        )

        # Process through MindEngine
        try:
            await asyncio.wait_for(
                self._engine.process_event({
                    "description": user_text,
                    "content": user_text,
                }),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pass
        if response_text:
            await self._engine.check_response_alignment(response_text)

        # v0.9.9: Record conversation turn for diary generation
        self._engine.record_conversation_turn(user_text, response_text)

        return {
            "type": "chat_response",
            "payload": {
                "text": response_text,
                "tone": self._engine.get_tone_params(),
            },
        }
