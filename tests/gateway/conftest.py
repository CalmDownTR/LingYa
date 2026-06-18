"""Shared fixtures for gateway tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


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
def router(mock_engine, mock_memory, mock_db, mock_agent, tmp_path):
    """MessageRouter with mocked dependencies, including agent."""
    from lingya.gateway.router import MessageRouter

    data_dir = str(tmp_path / "data")
    return MessageRouter(mock_engine, mock_memory, mock_db, data_dir, agent=mock_agent)


@pytest.fixture
def router_no_agent(mock_engine, mock_memory, mock_db, tmp_path):
    """MessageRouter without agent (backward compat / error case)."""
    from lingya.gateway.router import MessageRouter

    data_dir = str(tmp_path / "data")
    return MessageRouter(mock_engine, mock_memory, mock_db, data_dir, agent=None)
