"""EventBus — lightweight pub/sub for in-process event decoupling.

Zero framework dependency. Handlers are async callables. Exceptions in one
handler never block others (logged via logger.exception).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Event type constants
MIND_STATE_CHANGED = "mind_state_changed"
DIARY_READY = "diary_ready"
MEMORY_DECAYED = "memory_decayed"


class EventBus:
    """In-process pub/sub event bus.

    @reserved: v0.10 Dreaming 将添加 mind_state_changed 消费者。
    当前只有 MindEngine.process_event 发布事件，无订阅方。

    Usage::

        bus = EventBus()
        bus.subscribe(MIND_STATE_CHANGED, my_handler)
        await bus.publish(MIND_STATE_CHANGED, pad=..., emotion=...)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Register an async handler for an event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event_type: str, **kwargs: Any) -> None:
        """Fire all handlers for *event_type* with **kwargs.

        Each handler is awaited individually. Exceptions are caught and
        logged — they never block other handlers.
        """
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(**kwargs)
            except Exception:
                logger.exception(
                    "EventBus handler %r failed for event %r", handler, event_type
                )
