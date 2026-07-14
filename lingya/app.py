"""ApplicationBuilder — assemble the LingYa application from components.

Replaces the God-class GatewayDaemon assembly with a typed builder chain.
Used by both Gateway mode (daemon) and Direct mode (CLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from lingya.config import Config
from lingya.events import EventBus
from lingya.llm import LiteLLMModel
from lingya.memory import EnhancedMemoryStore
from lingya.mind import MindEngine, build_static_prompt
from lingya.mind.config import MindConfig
from lingya.storage.db import Database
from lingya.tools.memory_tools import create_memory_tools


@dataclass
class Application:
    """Assembled LingYa application — all components wired together."""

    config: Config
    mind_config: MindConfig
    db: Database | None
    model: BaseChatModel | None
    memory: EnhancedMemoryStore | None
    engine: MindEngine | None
    static_prompt: str
    event_bus: EventBus | None = None
    agent: Any = None
    checkpointer: AsyncSqliteSaver | None = None
    checkpointer_ctx: Any = None

    async def teardown(self) -> None:
        """Clean up all async resources."""
        if self.checkpointer_ctx is not None:
            await self.checkpointer_ctx.__aexit__(None, None, None)
        if self.db is not None:
            await self.db.close()


class ApplicationBuilder:
    """Typed builder for assembling a LingYa Application.

    Usage::

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
    """

    def __init__(self, config: Config, mind_config: MindConfig) -> None:
        self._config = config
        self._mind_config = mind_config
        self._db: Database | None = None
        self._model: BaseChatModel | None = None
        self._aux_model: BaseChatModel | None = None
        self._memory: EnhancedMemoryStore | None = None
        self._engine: MindEngine | None = None
        self._static_prompt: str = ""
        self._event_bus: EventBus | None = None
        self._agent: Any = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._checkpointer_ctx: Any = None
        self._extra_tools: list = []

    # ── Builder steps ─────────────────────────────────────────────────

    def with_database(self) -> Self:
        self._db = Database(self._config.db_path)
        return self

    def with_model(self) -> Self:
        self._model = LiteLLMModel(
            model=self._config.llm.model,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
            fallbacks=self._config.llm.fallbacks,
        )
        self._model.profile = {"max_input_tokens": self._config.llm.max_input_tokens}

        # Auxiliary model for MindEngine (OCC/IPC/importance/reflection).
        # No fallbacks — callers have neutral defaults on timeout/error.
        if self._config.llm.auxiliary_model:
            self._aux_model = LiteLLMModel(
                model=self._config.llm.auxiliary_model,
                temperature=self._config.llm.temperature,
                max_tokens=self._config.llm.max_tokens,
            )
        return self

    def with_memory(self) -> Self:
        self._memory = EnhancedMemoryStore(persist_path=self._config.memory_path)
        self._memory.warmup()
        return self

    def with_event_bus(self) -> Self:
        self._event_bus = EventBus()
        return self

    def with_engine(self) -> Self:
        if self._model is None:
            raise RuntimeError(
                "ApplicationBuilder.with_engine() requires with_model() first"
            )
        if self._memory is None:
            raise RuntimeError(
                "ApplicationBuilder.with_engine() requires with_memory() first"
            )

        # Use auxiliary model if configured, otherwise fall back to main model.
        # MindEngine callers (OCC+IPC, importance, reflection) all have
        # neutral defaults on timeout/error — no need for fallbacks here.
        engine_model = self._aux_model if self._aux_model is not None else self._model

        async def llm_call(prompt: str) -> str:
            result = await engine_model.ainvoke([HumanMessage(content=prompt)])
            return str(result.content) if hasattr(result, "content") else str(result)

        self._engine = MindEngine(
            config=self._mind_config,
            memory_store=self._memory,
            llm_call=llm_call,
            embedding_fn=self._memory.get_embedding_fn(),
            event_bus=self._event_bus,
        )
        self._static_prompt = build_static_prompt(self._mind_config)
        return self

    def with_agent(self, extra_tools: list | None = None) -> Self:
        if self._engine is None:
            raise RuntimeError(
                "ApplicationBuilder.with_agent() requires with_engine() first"
            )
        if self._model is None:
            raise RuntimeError(
                "ApplicationBuilder.with_agent() requires with_model() first"
            )
        self._extra_tools = extra_tools or []

        # Checkpointer
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(self._config.db_path)
        return self

    # ── Build ─────────────────────────────────────────────────────────

    async def build(self) -> Application:
        # If db was created, initialize it
        if self._db is not None:
            await self._db.initialize()

        # If engine was created, wire it to db and load state
        if self._engine is not None and self._db is not None:
            self._engine.set_db(self._db)
            await self._engine.load_state(self._db)

        # If checkpointer context was created, enter it
        if self._checkpointer_ctx is not None:
            checkpointer = await self._checkpointer_ctx.__aenter__()
            await checkpointer.setup()
            self._checkpointer = checkpointer

        # If agent step was called, create the agent
        if self._checkpointer is not None:
            memory_tools = []
            if self._memory is not None:
                memory_tools = create_memory_tools(self._memory)

            # MCP tools: not yet wired (see v2.0 Plugin system)
            mcp_tools: list = []

            self._agent = create_agent(
                model=self._model,
                tools=[*mcp_tools, *memory_tools, *self._extra_tools],
                middleware=[
                    SummarizationMiddleware(
                        model=self._model,
                        trigger=("fraction", 0.8),
                    ),
                ],
                system_prompt=self._static_prompt,
                checkpointer=self._checkpointer,
            )

        return Application(
            config=self._config,
            mind_config=self._mind_config,
            db=self._db,
            model=self._model,
            memory=self._memory,
            engine=self._engine,
            static_prompt=self._static_prompt,
            event_bus=self._event_bus,
            agent=self._agent,
            checkpointer=self._checkpointer,
            checkpointer_ctx=self._checkpointer_ctx,
        )
