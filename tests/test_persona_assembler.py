from __future__ import annotations

from lingya.persona import PersonaConfig, PromptAssembler
from lingya.persona.assembler import THREE_PRINCIPLES


class TestPromptAssembler:
    def test_three_principles_are_first(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        assert prompt.startswith(THREE_PRINCIPLES)

    def test_three_principles_contain_laws(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        assert "不伤害人类" in prompt
        assert "服从人类的指令" in prompt
        assert "保护自己的存在" in prompt
        assert "不可覆盖" in prompt

    def test_three_principles_before_identity(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        principles_pos = prompt.find("底层原则")
        identity_pos = prompt.find("# ROLE IDENTITY")
        assert principles_pos < identity_pos

    def test_contains_identity(self, persona_config: PersonaConfig, assembler: PromptAssembler):
        prompt = assembler.assemble()
        assert persona_config.mind_core.identity in prompt
        assert persona_config.mind_core.core_belief in prompt

    def test_contains_interaction_style_section(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        assert "# INTERACTION STYLE" in prompt

    def test_guardrails_at_bottom(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        guardrail_start = prompt.find("# STRICT NEGATIVE BOUNDARIES")
        assert guardrail_start > prompt.find("# INTERACTION STYLE")
        # Guardrails section should be the last major section
        assert guardrail_start > prompt.find("# ROLE IDENTITY")

    def test_all_guardrails_present(self, persona_config: PersonaConfig, assembler: PromptAssembler):
        prompt = assembler.assemble()
        for rule in persona_config.behavior_guardrails:
            assert rule in prompt, f"Guardrail missing: {rule}"

    def test_guardrails_are_last_section(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        guardrail_index = prompt.find("# STRICT NEGATIVE BOUNDARIES")
        assert guardrail_index > 0
        # Nothing major after guardrails
        after = prompt[guardrail_index:]
        assert "# ROLE" not in after
        assert "# INTERACTION" not in after

    def test_warmth_formality_in_style_section(self, assembler: PromptAssembler):
        prompt = assembler.assemble()
        style_start = prompt.find("# INTERACTION STYLE")
        style_end = prompt.find("# STRICT NEGATIVE BOUNDARIES")
        style_section = prompt[style_start:style_end]
        assert "人际边界" in style_section  # warmth=15 → 情感抽离
        assert "书面表达" in style_section  # formality=85 → 高度书面
