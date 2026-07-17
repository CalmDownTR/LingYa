"""Shared fixtures for gateway tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockStreamRun:
    """Mock ``AsyncGraphRunStream`` for tests that consume ``stream.extensions``.

    The production ``_chat_streaming`` uses two async iterables concurrently:
    the main event loop (``async for event in run``) and the LingYa extension
    channel (``async for item in run.extensions["lingya_inner"]``).  This
    class wraps two lists into a single object that satisfies both protocols.
    """

    def __init__(
        self,
        main_events: list[dict[str, Any]],
        extensions: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._main_events = main_events
        self._ext_events = extensions or {}
        self.extensions = self._Extensions(self._ext_events)

    def __aiter__(self):
        return self._make_async_gen(self._main_events)

    @staticmethod
    async def _make_async_gen(items: list[dict[str, Any]]):
        for item in items:
            yield item

    class _Extensions:
        def __init__(self, events_map: dict[str, list[dict[str, Any]]]) -> None:
            self._events_map = events_map

        def __getitem__(self, key: str):
            events = self._events_map.get(key, [])

            async def _gen():
                for e in events:
                    yield e

            return _gen()


@pytest.fixture
def mock_engine():
    """Mock MindEngine with controlled state."""
    engine = MagicMock()

    engine.state.current_pad.pleasure = 0.5
    engine.state.current_pad.arousal = 0.3
    engine.state.current_pad.dominance = 0.7

    engine.state.recent_emotions = [
        {"emotion": "joy", "intensity": 0.8},
        {"emotion": "surprise", "intensity": 0.4},
    ]

    engine.state.ipc_state = "agency_high"
    engine.state.ipc_agency = 0.8
    engine.state.ipc_communion = 0.5

    engine.get_tone_params.return_value = {
        "warmth": 60,
        "formality": 50,
        "humor": 0.3,
    }

    engine.get_prompt_fragment.return_value = (
        "[本次回复的语气指令]\nwarmth=high\nformality=mid\n"
    )

    engine.process_event = AsyncMock()
    engine.check_response_alignment = AsyncMock()

    engine.state.current_ocean.openness = 0.6
    engine.state.current_ocean.conscientiousness = 0.7
    engine.state.current_ocean.extraversion = 0.5
    engine.state.current_ocean.agreeableness = 0.8
    engine.state.current_ocean.neuroticism = 0.4

    engine.state.turn_counter = 42

    return engine


@pytest.fixture
def mock_memory():
    """Mock EnhancedMemoryStore with controlled data."""
    memory = MagicMock()
    memory.search.return_value = [
        {"id": "mem_1", "text": "User likes coffee"},
        {"id": "mem_2", "text": "User works at a startup"},
    ]
    memory.list_all.return_value = [
        {"id": "mem_1", "text": "User likes coffee"},
        {"id": "mem_2", "text": "User works at a startup"},
    ]
    return memory


@pytest.fixture
def mock_agent():
    """Mock deep agent with controlled responses."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    agent.astream_events = AsyncMock()
    return agent


@pytest.fixture
def session_service(mock_db, tmp_path):
    """SessionService with mock DB."""
    from lingya.gateway.session_service import SessionService

    data_dir = str(tmp_path / "data")
    return SessionService(db=mock_db, data_dir=data_dir)


@pytest.fixture
def settings_service(mock_engine):
    """SettingsService with mock engine."""
    from lingya.gateway.settings_service import SettingsService

    return SettingsService(engine=mock_engine)


@pytest.fixture
def chat_handler(mock_engine, mock_agent, session_service):
    """ChatHandler with mock engine, agent, and session service."""
    from lingya.gateway.chat_handler import ChatHandler

    # Wire agent into session_service for history loading
    session_service.set_agent(mock_agent)

    return ChatHandler(
        engine=mock_engine,
        agent=mock_agent,
        session_service=session_service,
    )


@pytest.fixture
def chat_handler_no_agent(mock_engine, session_service):
    """ChatHandler without agent (error case)."""
    from lingya.gateway.chat_handler import ChatHandler

    return ChatHandler(
        engine=mock_engine,
        agent=None,
        session_service=session_service,
    )


@pytest.fixture
def router(mock_engine, mock_memory, session_service, settings_service, chat_handler, tmp_path):
    """MessageRouter with mocked dependencies."""
    from lingya.gateway.router import MessageRouter

    data_dir = str(tmp_path / "data")
    return MessageRouter(
        engine=mock_engine,
        memory=mock_memory,
        data_dir=data_dir,
        session_service=session_service,
        settings_service=settings_service,
        chat_handler=chat_handler,
    )


@pytest.fixture
def router_no_agent(mock_engine, mock_memory, session_service, settings_service, chat_handler_no_agent, tmp_path):
    """MessageRouter without agent (error case)."""
    from lingya.gateway.router import MessageRouter

    data_dir = str(tmp_path / "data")
    return MessageRouter(
        engine=mock_engine,
        memory=mock_memory,
        data_dir=data_dir,
        session_service=session_service,
        settings_service=settings_service,
        chat_handler=chat_handler_no_agent,
    )
