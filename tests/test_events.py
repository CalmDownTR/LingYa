"""Tests for lingya.events — EventBus pub/sub."""

from __future__ import annotations

import pytest


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Subscribed handler receives published data."""
        from lingya.events import EventBus

        bus = EventBus()
        received = []

        async def handler(**kwargs):
            received.append(kwargs)

        bus.subscribe("test_event", handler)
        await bus.publish("test_event", key="value")

        assert len(received) == 1
        assert received[0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_publish_no_subscribers_no_error(self):
        """Publishing to an event with no subscribers should not raise."""
        from lingya.events import EventBus

        bus = EventBus()
        await bus.publish("no_one_listening", data=42)

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_block_others(self):
        """One handler raising should not block other handlers."""
        from lingya.events import EventBus

        bus = EventBus()
        received = []

        async def bad_handler(**kwargs):
            raise RuntimeError("boom")

        async def good_handler(**kwargs):
            received.append(kwargs)

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)
        await bus.publish("test", value=1)

        assert len(received) == 1
        assert received[0] == {"value": 1}

    @pytest.mark.asyncio
    async def test_multiple_events_independent(self):
        """Subscribers only receive their subscribed events."""
        from lingya.events import EventBus

        bus = EventBus()
        a_received = []
        b_received = []

        async def handler_a(**kwargs):
            a_received.append(kwargs)

        async def handler_b(**kwargs):
            b_received.append(kwargs)

        bus.subscribe("event_a", handler_a)
        bus.subscribe("event_b", handler_b)

        await bus.publish("event_a", msg="hello")
        await bus.publish("event_b", msg="world")

        assert len(a_received) == 1
        assert a_received[0] == {"msg": "hello"}
        assert len(b_received) == 1
        assert b_received[0] == {"msg": "world"}

    def test_event_constants_exist(self):
        """Predefined event type constants should be importable."""
        from lingya.events import (
            DIARY_READY,
            MEMORY_DECAYED,
            MIND_STATE_CHANGED,
        )

        assert isinstance(MIND_STATE_CHANGED, str)
        assert isinstance(DIARY_READY, str)
        assert isinstance(MEMORY_DECAYED, str)
