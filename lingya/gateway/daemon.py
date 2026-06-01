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

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from lingya.config import Config
from lingya.memory import EnhancedMemoryStore
from lingya.mind import MindEngine, build_static_prompt
from lingya.mind.config import MindConfig
from lingya.storage.db import Database


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
    ) -> None:
        """Store configs; engine and infrastructure are created in start()."""
        self.config = config
        self.mind_config = mind_config
        self.pid_file = pid_file
        self._shutdown_event = asyncio.Event()
        self._engine: MindEngine | None = None
        self._db: Database | None = None
        self._model: ChatOpenAI | None = None
        self._memory: EnhancedMemoryStore | None = None

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
        self._register_signal_handlers()

        print(
            f"LingYa daemon started (PID: {os.getpid()}, port: TODO)"
        )

        # Block until shutdown is triggered (signal or programmatic)
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """Graceful shutdown.

        1. Save mind state via engine.save_state(db)
        2. Close database
        3. Remove PID file
        4. Print shutdown message
        """
        if self._engine is not None and self._db is not None:
            await self._engine.save_state(self._db)

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
