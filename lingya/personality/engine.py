from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from lingya.config import PersonalityConfig

from .model import ActivePersonality, PersonalityAdapter, PersonalityGenome
from .templates import REFLECTION_SYSTEM_PROMPT

if TYPE_CHECKING:
    from lingya.llm.base import BaseLLMBackend
    from lingya.storage.db import Database


class PersonalityEngine:
    def __init__(
        self,
        config: PersonalityConfig,
        llm: BaseLLMBackend,
        db: Database,
    ) -> None:
        self.config = config
        self.llm = llm
        self.db = db
        self._genome = PersonalityGenome()
        self._turn_since_reflection = 0

    @property
    def personality(self) -> ActivePersonality:
        return PersonalityAdapter.activate(self._genome)

    def get_system_prompt(self) -> str:
        return self.personality.to_system_prompt()

    async def load(self) -> None:
        data = await self.db.get_personality()
        if data:
            self._genome = PersonalityGenome.model_validate(data)
        elif self.config.seed_personality:
            try:
                with open(self.config.seed_personality) as f:
                    seed_data = json.load(f)
                self._genome = PersonalityGenome.model_validate(seed_data)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    async def save(self) -> None:
        self._genome.last_updated = datetime.now(timezone.utc).isoformat()
        await self.db.save_personality(self._genome.model_dump())

    async def maybe_evolve(
        self, recent_summary: str, user_feedback: str | None = None
    ) -> bool:
        """Try to evolve the personality. Returns True if changes were made.
        Currently a stub — full implementation deferred to v0.2.0."""
        self._turn_since_reflection += 1
        if self._turn_since_reflection < self.config.reflection_interval_turns:
            return False

        self._turn_since_reflection = 0
        return False
