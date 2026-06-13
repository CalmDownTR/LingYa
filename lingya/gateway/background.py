"""BackgroundRunner — heartbeat, diary scheduler, and memory decay for idle daemon lifecycle.

Gives LingYa "a life beyond conversations" — PAD slowly drifts toward
baseline when idle, diary generation is scheduled periodically, and
memory decay runs once per day.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from lingya.protocols import IMemoryStore
    from lingya.mind.engine import MindEngine
    from lingya.storage.db import Database

logger = logging.getLogger(__name__)


class BackgroundRunner:
    """Background tasks that give LingYa an independent life rhythm.

    Three loops:
    - heartbeat: PAD idle drift, fires every N seconds
    - diary_scheduler: checks if diary is due, fires every hour
    - decay: memory decay, fires once per day
    """

    def __init__(
        self,
        engine: MindEngine,
        db: Database,
        model: BaseChatModel,
        data_dir: str,
        memory: IMemoryStore | None = None,
        heartbeat_interval: int = 60,
        diary_check_interval: int = 3600,
        decay_interval: int = 86400,
        tracer: Any = None,
    ) -> None:
        self._engine = engine
        self._db = db
        self._model = model
        self._data_dir = data_dir
        self._memory = memory
        self.heartbeat_interval = heartbeat_interval
        self.diary_check_interval = diary_check_interval
        self.decay_interval = decay_interval
        self._tracer = tracer

        self._running: bool = True
        self._heartbeat_task: asyncio.Task | None = None
        self._diary_task: asyncio.Task | None = None
        self._decay_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background loops as asyncio tasks.

        Safe to call multiple times — second call is a no-op if already running.
        """
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._diary_task = asyncio.create_task(self._diary_scheduler_loop())
        if self._memory is not None:
            self._decay_task = asyncio.create_task(self._decay_loop())

    async def stop(self) -> None:
        """Cancel background tasks and wait for them to finish."""
        self._running = False

        tasks = []
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            tasks.append(self._heartbeat_task)
        if self._diary_task is not None:
            self._diary_task.cancel()
            tasks.append(self._diary_task)
        if self._decay_task is not None:
            self._decay_task.cancel()
            tasks.append(self._decay_task)

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._heartbeat_task = None
        self._diary_task = None
        self._decay_task = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Loops ──────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """PAD idle drift — fire every heartbeat_interval seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break
            if self._tracer:
                with self._tracer.start_as_current_span("bg.heartbeat"):
                    await self._engine.idle_tick()
            else:
                await self._engine.idle_tick()

    async def _diary_scheduler_loop(self) -> None:
        """Check if diary is due, generate if so — fire every diary_check_interval seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.diary_check_interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            if self._tracer:
                with self._tracer.start_as_current_span("bg.diary_scheduler"):
                    await self._try_generate_diary()
            else:
                await self._try_generate_diary()

    # ── Memory Decay Loop ───────────────────────────────────────────

    async def _decay_loop(self) -> None:
        """Apply memory decay once per day (default: every 86400 seconds)."""
        while self._running:
            try:
                await asyncio.sleep(self.decay_interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            if self._memory is None:
                continue

            if self._tracer:
                with self._tracer.start_as_current_span("bg.decay"):
                    affected = self._memory.apply_decay()
                    if affected > 0:
                        logger.info("Memory decay: %d memories affected", affected)
            else:
                try:
                    affected = self._memory.apply_decay()
                    if affected > 0:
                        logger.info("Memory decay: %d memories affected", affected)
                except Exception:
                    logger.exception(
                        "Memory decay error — will retry on next tick"
                    )

    async def _try_generate_diary(self) -> None:
        """Check diary eligibility and generate if due.

        Extracted for testability — can be called directly with mocked dependencies.
        """
        from lingya.diary import (
            format_transcript,
            generate_diary,
            get_diary_dir,
            get_last_diary_date,
            has_deep_conversation,
            save_diary,
            should_generate_diary,
        )

        try:
            diary_dir = get_diary_dir(self._data_dir)
            if not should_generate_diary(diary_dir, period_days=1):
                return

            # Get turns since last diary (or all recent if no diary yet)
            last_date = get_last_diary_date(diary_dir)
            since = last_date.isoformat() if last_date else "1970-01-01"
            turns = await self._db.get_turns_since(since)

            if not has_deep_conversation(turns):
                logger.debug("Diary due but no deep conversation — skipping")
                return

            transcript = format_transcript(turns)

            # Build a MindConfig from the engine's current config
            mind_config = self._engine.config

            content = await generate_diary(self._model, mind_config, transcript)
            path = save_diary(diary_dir, date.today(), content)
            logger.info("Diary generated: %s", path)

        except Exception:
            logger.exception("Diary scheduler error — will retry on next tick")
