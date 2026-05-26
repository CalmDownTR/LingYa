from __future__ import annotations

from lingya.persona.bucketing import map_formality, map_warmth
from lingya.persona.config import PersonaConfig


class PromptAssembler:
    def __init__(self, config: PersonaConfig) -> None:
        self.config = config

    def assemble(self) -> str:
        parts: list[str] = []

        parts.append("# ROLE IDENTITY")
        parts.append(self.config.mind_core.identity)
        parts.append(self.config.mind_core.core_belief)

        parts.append("")
        parts.append("# INTERACTION STYLE")
        parts.append(f"- {map_warmth(self.config.tone_matrix.warmth)}")
        parts.append(f"- {map_formality(self.config.tone_matrix.formality)}")

        parts.append("")
        parts.append("# STRICT NEGATIVE BOUNDARIES (COMPLY ABSOLUTELY)")
        for rule in self.config.behavior_guardrails:
            parts.append(f"- {rule}")

        return "\n".join(parts)
