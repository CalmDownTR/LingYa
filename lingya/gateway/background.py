"""BackgroundRunner — heartbeat, diary scheduler, and memory decay for idle daemon lifecycle.

Gives LingYa "a life beyond conversations" — PAD slowly drifts toward
baseline when idle, diary generation is scheduled periodically, and
memory decay runs once per day.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from lingya.protocols import IMemoryStore
    from lingya.mind.engine import MindEngine

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
        model: BaseChatModel,
        data_dir: str,
        memory: IMemoryStore | None = None,
        heartbeat_interval: int = 60,
        diary_check_interval: int = 3600,
        decay_interval: int = 86400,
    ) -> None:
        self._engine = engine
        self._model = model
        self._data_dir = data_dir
        self._memory = memory
        self.heartbeat_interval = heartbeat_interval
        self.diary_check_interval = diary_check_interval
        self.decay_interval = decay_interval

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

            try:
                affected = self._memory.apply_decay()
                if affected > 0:
                    logger.info("Memory decay: %d memories affected", affected)
            except Exception:
                logger.exception(
                    "Memory decay error — will retry on next tick"
                )

    async def _try_generate_diary(self) -> None:
        """Generate a diary entry if enough time has passed since the last one.

        v0.9.9: Reads recent conversation transcript from MindEngine,
        generates a diary via LLM, and saves to data/diary/YYYY-MM-DD.md.
        """
        from datetime import date

        from lingya.diary import (
            generate_diary,
            get_diary_dir,
            save_diary,
            should_generate_diary,
        )

        diary_dir = get_diary_dir(self._data_dir)

        # Check if a diary is due (default: once per day)
        if not should_generate_diary(diary_dir, period_days=1):
            return

        # Get recent conversation transcript from engine
        transcript = self._engine.get_recent_transcript(hours=24)

        try:
            content = await generate_diary(
                model=self._model,
                mind_config=self._engine.config,
                transcript=transcript,
            )
            path = save_diary(diary_dir, date.today(), content)
            logger.info("Diary generated: %s (%d chars)", path, len(content))
        except Exception:
            logger.exception("Diary generation failed — will retry on next tick")
