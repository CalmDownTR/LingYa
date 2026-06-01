"""GatewayDaemon — long-running process that owns the MindEngine singleton.

Lifecycle:
  start() -> init engine -> init server -> serve_forever
                                                |
                                          SIGTERM -> save_state -> close_db -> exit
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware

from lingya.config import Config
from lingya.gateway.router import MessageRouter
from lingya.gateway.server import GatewayServer
from lingya.memory import EnhancedMemoryStore
from lingya.mind import MindEngine, build_static_prompt
from lingya.mind.config import MindConfig
from lingya.storage.db import Database

if TYPE_CHECKING:
    from lingya.gateway.background import BackgroundRunner


class GatewayDaemon:
    """Long-running process that owns the MindEngine singleton.

    Lifecycle:
      start() -> init engine -> init server -> serve_forever
                                                    |
                                              SIGTERM -> save_state -> close_db -> exit
    """

    def __init__(
        self,
        config: Config,
        mind_config: MindConfig,
        pid_file: str = "/tmp/lingya.pid",
        port: int = 8765,
    ) -> None:
        """Store configs; engine and infrastructure are created in start()."""
        self.config = config
        self.mind_config = mind_config
        self.pid_file = pid_file
        self.port = port
        self._shutdown_event = asyncio.Event()
        self._engine: MindEngine | None = None
        self._db: Database | None = None
        self._model: ChatOpenAI | None = None
        self._memory: EnhancedMemoryStore | None = None
        self._agent: object | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._ws_server: GatewayServer | None = None
        self._bg_runner: BackgroundRunner | None = None

    # ── Public API ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Full startup sequence.

        1. Write PID file
        2. Create Database, initialize
        3. Create ChatOpenAI model
        4. Create EnhancedMemoryStore, warmup
        5. Create MindEngine, set_db, load_state
        6. Register signal handlers (SIGTERM, SIGINT)
        7. Print startup message
        8. Keep running (await shutdown event)
        """
        self._write_pid_file()
        await self._init_database()
        self._init_model()
        self._init_memory()
        await self._init_engine()
        await self._init_agent()
        self._register_signal_handlers()

        # Create router and WebSocket server
        router = MessageRouter(
            engine=self._engine,
            memory=self._memory,
            db=self._db,
            data_dir=self.config.data_dir,
            agent=self._agent,
        )
        self._ws_server = GatewayServer("0.0.0.0", self.port, router)
        await self._ws_server.start()

        # Start background runner (heartbeat + diary scheduler)
        # Only start if all dependencies are initialized (they always are in
        # normal operation; guarded for test scenarios that mock init methods).
        if self._engine is not None and self._db is not None and self._model is not None:
            from lingya.gateway.background import BackgroundRunner

            self._bg_runner = BackgroundRunner(
                engine=self._engine,
                db=self._db,
                model=self._model,
                data_dir=self.config.data_dir,
            )
            await self._bg_runner.start()

        print(
            f"LingYa daemon started (PID: {os.getpid()}, port: {self.port})"
        )

        # Block until shutdown is triggered (signal or programmatic)
        await self._shutdown_event.wait()

        # Stop background runner before WebSocket server
        if self._bg_runner:
            await self._bg_runner.stop()

        # Stop the WebSocket server before cleanup
        if self._ws_server:
            await self._ws_server.stop()

    async def shutdown(self) -> None:
        """Graceful shutdown.

        1. Save mind state via engine.save_state(db)
        2. Close database
        3. Remove PID file
        4. Print shutdown message
        """
        if self._engine is not None and self._db is not None:
            await self._engine.save_state(self._db)

        if self._checkpointer is not None:
            await self._checkpointer.__aexit__(None, None, None)

        if self._db is not None:
            await self._db.close()

        self._remove_pid_file()
        print("LingYa daemon shut down.")

    @staticmethod
    def is_running(pid_file: str = "/tmp/lingya.pid") -> bool:
        """Check if a daemon is running by reading the PID file.

        Returns True only if:
        - The PID file exists
        - Contains a valid PID
        - A process with that PID is alive

        Used by ``lingya start`` to detect an already-running daemon.
        """
        if not os.path.exists(pid_file):
            return False
        try:
            pid = int(Path(pid_file).read_text().strip())
            os.kill(pid, 0)  # Signal 0 checks existence without sending a signal
            return True
        except ProcessLookupError:
            return False
        except (ValueError, OSError):
            return False

    # ── Internal helpers ────────────────────────────────────────────

    def _write_pid_file(self) -> None:
        """Write the current process PID to the configured PID file."""
        Path(self.pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.pid_file).write_text(str(os.getpid()))

    def _remove_pid_file(self) -> None:
        """Remove the PID file if it exists."""
        try:
            Path(self.pid_file).unlink(missing_ok=True)
        except Exception:
            pass  # PID file already gone — nothing to clean up

    async def _init_database(self) -> None:
        """Create and initialize the SQLite database."""
        self._db = Database(self.config.db_path)
        await self._db.initialize()

    def _init_model(self) -> None:
        """Create the ChatOpenAI model with config-driven parameters."""
        api_key = os.environ[self.config.llm.api_key_env]
        self._model = ChatOpenAI(
            model=self.config.llm.model,
            api_key=SecretStr(api_key),
            base_url=self.config.llm.api_base_url,
            temperature=self.config.llm.temperature,
        )
        # Tell deepagents the actual context window so summarization
        # triggers at 85% (on-time) instead of a fixed 170k default.
        self._model.profile = {"max_input_tokens": self.config.llm.max_input_tokens}

    def _init_memory(self) -> None:
        """Create and warm up the EnhancedMemoryStore."""
        self._memory = EnhancedMemoryStore(persist_path=self.config.memory_path)
        self._memory.warmup()

    async def _init_engine(self) -> None:
        """Create MindEngine, wire it to DB and LLM, restore saved state."""
        async def llm_call(prompt: str) -> str:
            """Simple LLM call for MindEngine's cognitive appraisal, IPC, etc."""
            result = await self._model.ainvoke([HumanMessage(content=prompt)])
            return str(result.content) if hasattr(result, "content") else str(result)

        self._engine = MindEngine(
            config=self.mind_config,
            memory_store=self._memory,
            llm_call=llm_call,
        )
        self._engine.set_db(self._db)
        await self._engine.load_state(self._db)

        # Pre-build the static prompt so it's available to clients
        self._static_prompt = build_static_prompt(self.mind_config)

    async def _init_agent(self) -> None:
        """Create the deep agent with model, tools, middleware, and checkpointer.

        Called after _init_engine() so memory and static prompt are available.
        """
        # 1. Checkpointer — manually manage the async context manager lifecycle
        checkpointer = AsyncSqliteSaver.from_conn_string(self.config.db_path)
        await checkpointer.__aenter__()
        await checkpointer.setup()
        self._checkpointer = checkpointer

        # 2. Backend
        backend = StateBackend()

        # 3. Memory tools (same as main.py)
        @tool
        def memory_store(text: str) -> str:
            """Remember important information about the user.

            Use this tool when the user shares personal preferences, identity,
            emotional states, or context useful for future interactions.
            Examples: "I like rainy days", "I'm afraid of loneliness",
            "I'm a freelancer".

            Do NOT use for transient information (e.g. "I'm running late"),
            one-time tasks, small talk, or credentials/API keys.
            """
            return self._memory.store(text)

        @tool
        def memory_search(query: str) -> str:
            """Search for prior memories about the user.

            Use this tool when the user asks "do you remember...", or when you
            need to recall context from past conversations to answer accurately.
            Returns matching memories with their text content.
            """
            results = self._memory.search(query)
            if not results:
                return "(No matching memories found)"
            lines = []
            for r in results:
                lines.append(f"[{r['id']}] {r['text']}")
            return "\n".join(lines)

        # 4. MCP tools (optional)
        mcp_tools: list = []
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            # If MCP servers are configured, connect and discover tools
            # mcp_client = MultiServerMCPClient({...})  # TODO: config-driven
            # mcp_tools = await mcp_client.get_tools()
            pass
        except Exception:
            pass

        # 5. Agent
        self._agent = create_deep_agent(
            model=self._model,
            tools=[*mcp_tools, memory_store, memory_search],
            middleware=[
                create_summarization_tool_middleware(self._model, backend=backend),
            ],
            system_prompt=self._static_prompt,
            backend=backend,
            checkpointer=checkpointer,
        )

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT handlers for graceful shutdown.

        When either signal is received, the shutdown event is set, which
        unblocks the start() event loop and allows the caller to perform
        a clean shutdown sequence.
        """
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                # Windows does not support add_signal_handler.
                # Graceful shutdown via other mechanisms (e.g., pipe).
                pass
