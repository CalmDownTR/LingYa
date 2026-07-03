from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import uvicorn

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
        self._uvicorn_server: uvicorn.Server | None = None
        self._uvicorn_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Assemble application, start HTTP+SSE server, block until shutdown."""
        from lingya.app import ApplicationBuilder
        from lingya.gateway.router import MessageRouter
        from lingya.gateway.server import create_app

        # 0. Auto-instrumentation via OpenLLMetry (no-op if otel.enabled=False).
        # One line covers LangChain, OpenAI, and ChromaDB — all producing OTel spans.
        # v1.0 export to Langfuse via OTEL_EXPORTER_OTLP_ENDPOINT env var (zero code change).
        if self.config.otel.enabled:
            from traceloop.sdk import Traceloop

            Traceloop.init(
                disable_batch=False,
                # TRACELOOP_TRACE_CONTENT=false keeps prompt/completion out of spans
            )

        # 1. Assemble application via builder
        self._app = await (
            ApplicationBuilder(self.config, self.mind_config)
            .with_database().with_model().with_memory()
            .with_event_bus().with_engine().with_agent()
            .build()
        )

        # 2. Create router
        router = MessageRouter(
            engine=self._app.engine, memory=self._app.memory,
            db=self._app.db, data_dir=self.config.data_dir,
            agent=self._app.agent,
        )

        # 3. Build FastAPI app + start uvicorn in background
        fastapi_app = create_app(
            router=router,
            auth_enabled=self.config.auth_enabled,
            shutdown_callback=self._shutdown_event.set,
        )

        uvicorn_config = uvicorn.Config(
            fastapi_app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )
        self._uvicorn_server = uvicorn.Server(uvicorn_config)

        # Start uvicorn in background — serve() blocks until should_exit is set
        async def _run_server():
            # First attempt — try the original port
            try:
                await self._uvicorn_server.serve()
            except (OSError, SystemExit):
                # uvicorn calls sys.exit(1) on bind failure, which raises SystemExit.
                # Check if port is occupied by a stale process and retry.
                old_pid = _find_port_owner(self.port)
                if old_pid is not None and old_pid != os.getpid():
                    os.kill(old_pid, signal.SIGTERM)
                    await asyncio.sleep(0.5)
                    # Re-create server since uvicorn.Server can't be restarted
                    self._uvicorn_server = uvicorn.Server(uvicorn_config)
                    await self._uvicorn_server.serve()
                else:
                    raise RuntimeError(
                        f"Port {self.port} is already in use by an unknown process. "
                        f"Kill it manually: lsof -ti :{self.port} | xargs kill"
                    )

        self._uvicorn_task = asyncio.create_task(_run_server())

        # Give uvicorn a moment to bind the port, then verify it's alive
        await asyncio.sleep(0.3)
        if self._uvicorn_task.done():
            # Task crashed — retrieve and re-raise the exception
            exc = self._uvicorn_task.exception()
            if exc:
                raise RuntimeError(f"uvicorn server failed to start: {exc}") from exc
            raise RuntimeError("uvicorn server exited unexpectedly")

        self._write_pid_file()

        # 4. Start background runner
        if self._app.engine and self._app.model:
            self._bg_runner = BackgroundRunner(
                engine=self._app.engine,
                model=self._app.model, data_dir=self.config.data_dir,
                memory=self._app.memory,
            )
            await self._bg_runner.start()

        print(f"LingYa daemon started (PID: {os.getpid()}, port: {self.port})")

        # Check for Web UI
        web_dist = Path("web/dist")
        if web_dist.exists():
            print(f"Web UI available at http://localhost:{self.port}")
        else:
            print(
                "Note: Web UI not built. "
                "To enable: cd web && npm install && npm run build"
            )

        # 5. Wait for shutdown signal
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass
        await self._shutdown_event.wait()

        # 6. Shutdown in reverse order
        if self._bg_runner:
            await self._bg_runner.stop()
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            if self._uvicorn_task:
                try:
                    await self._uvicorn_task
                except Exception:
                    pass

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


def _find_port_owner(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None