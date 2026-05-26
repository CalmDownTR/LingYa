from __future__ import annotations

from lingya.persona import PersonaConfig, PromptAssembler


class TestPromptAssembler:
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
