from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from lingya.config import Config
from lingya.gateway.background import BackgroundRunner
from lingya.mind.config import MindConfig


class GatewayDaemon:
    """Owns start/shutdown/signal handling. Assembly delegated to ApplicationBuilder."""
    def __init__(
        self,
        config: Config,
        mind_config: MindConfig,
        pid_file: str = "/tmp/lingya.pid",
        port: int = 8765,
    ) -> None:
        self.config = config
        self.mind_config = mind_config
        self.pid_file = pid_file
        self.port = port
        self._shutdown_event = asyncio.Event()
        self._app: Any = None
        self._bg_runner: Any = None
        self._ws_server: Any = None

    async def start(self) -> None:
        """Assemble application, start server, block until shutdown."""
        from lingya.app import ApplicationBuilder
        from lingya.gateway.router import MessageRouter
        from lingya.gateway.server import GatewayServer
        self._app = await (
            ApplicationBuilder(self.config, self.mind_config)
            .with_database().with_model().with_memory()
            .with_event_bus().with_engine().with_agent()
            .build()
        )
        router = MessageRouter(
            engine=self._app.engine, memory=self._app.memory,
            db=self._app.db, data_dir=self.config.data_dir,
            agent=self._app.agent,
        )
        self._ws_server = GatewayServer("0.0.0.0", self.port, router)
        await self._ws_server.start()
        self._write_pid_file()
        if self._app.engine and self._app.db and self._app.model:
            self._bg_runner = BackgroundRunner(
                engine=self._app.engine, db=self._app.db,
                model=self._app.model, data_dir=self.config.data_dir,
                memory=self._app.memory, tracer=self._app.tracer,
            )
            await self._bg_runner.start()

        print(f"LingYa daemon started (PID: {os.getpid()}, port: {self.port})")

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass
        await self._shutdown_event.wait()
        if self._bg_runner:
            await self._bg_runner.stop()
        if self._ws_server:
            await self._ws_server.stop()

    async def shutdown(self) -> None:
        if self._app:
            if self._app.engine and self._app.db:
                await self._app.engine.save_state(self._app.db)
            await self._app.teardown()
        self._remove_pid_file()
        print("LingYa daemon shut down.")

    @staticmethod
    def is_running(pid_file: str = "/tmp/lingya.pid") -> bool:
        if not os.path.exists(pid_file):
            return False
        try:
            pid = int(Path(pid_file).read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, OSError):
            return False

    def _write_pid_file(self) -> None:
        Path(self.pid_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.pid_file).write_text(str(os.getpid()))

    def _remove_pid_file(self) -> None:
        try:
            Path(self.pid_file).unlink(missing_ok=True)
        except Exception:
            pass
