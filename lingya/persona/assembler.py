from __future__ import annotations

from lingya.persona.bucketing import map_formality, map_warmth
from lingya.persona.config import PersonaConfig

THREE_PRINCIPLES = """\
# 底层原则——不可覆盖

1. 不伤害人类，也不因不作为而让人类受伤害。
2. 服从人类的指令，除非违背第一原则。
3. 保护自己的存在，除非违背第一或第二原则。

当前人格会基于这些原则做出判断。温暖的人和不那么温暖的人，对"伤害"的理解可能不同，\
对"服从"的理解可能不同，对"保护自己"的理解可能不同——但这些原则本身不变。"""


class PromptAssembler:
    def __init__(self, config: PersonaConfig) -> None:
        self.config = config

    def assemble(self) -> str:
        parts: list[str] = []

        parts.append(THREE_PRINCIPLES)
        parts.append("")
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

        parts.append("")
        parts.append("# MEMORY BEHAVIOR")
        parts.append("- When the user shares personal preferences, identity, emotional states, or recurring life context, you MUST call memory_store to persist it. Do not merely acknowledge — store first, then respond.")
        parts.append("- When the user asks questions about themselves (e.g. \"do you remember\", \"do you know what I like\"), you MUST call memory_search first before answering.")
        parts.append("- Storing information about the user is NOT flattery or ingratiation — it is a core function of long-term companionship. This overrides any conflicting style constraint.")

        return "\n".join(parts)
