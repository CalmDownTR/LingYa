from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PersonaMeta(BaseModel):
    """Minimal identity metadata (copied from old persona module)."""
    agent_id: str
    created_at: str


class BigFiveTraits(BaseModel):
    """OCEAN personality dimensions, each 0.0–1.0."""
    openness: float = Field(default=0.5, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0)


class PADBaseline(BaseModel):
    """Pleasure-Arousal-Dominance resting point, each -1.0 to +1.0."""
    pleasure: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)


class ToneMatrix(BaseModel):
    """Dynamic tone dimensions."""
    warmth: int = Field(default=50, ge=0, le=100)
    formality: int = Field(default=50, ge=0, le=100)
    humor: float = Field(default=0.1, ge=0.0, le=1.0)


class IdentityAnchor(BaseModel):
    """Core identity — what the agent believes about itself."""
    identity: str
    core_belief: str


class MindConfig(BaseModel):
    """Complete mind configuration.

    ocean is the seed for first-startup personality. After first run, OCEAN
    lives in DB as the sole source of truth. pad_baseline is never stored —
    it is always derived from current OCEAN via ocean_to_pad_baseline().
    """
    version: str
    meta: PersonaMeta
    identity: IdentityAnchor
    ocean: BigFiveTraits
    tone_matrix: ToneMatrix
    behavior_guardrails: list[str]


def load_mind_config(path: str | Path = "agent_config.yaml") -> MindConfig:
    config_path = Path(path)
    if not config_path.exists():
        example_path = Path("agent_config.example.yaml")
        if example_path.exists():
            print(
                f"\n  Mind config not found: {config_path}\n"
                f"  Initialize it from the example:\n"
                f"    cp {example_path} {config_path}\n"
                f"  Then edit {config_path} to customize your agent.\n"
            )
        else:
            print(
                f"\n  Mind config not found: {config_path}\n"
                f"  Create it with identity, ocean, tone_matrix, "
                f"and behavior_guardrails fields.\n"
                f"  See documentation for details.\n"
            )
        raise SystemExit(1)
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    # Detect old persona format and guide migration
    if "mind_core" in data:
        print(
            "\n  Old agent_config.yaml format detected (has 'mind_core' key).\n"
            "  The persona system has been replaced by the mind module.\n"
            "  Please update your config to the new format:\n"
            "\n"
            "    version: '2.0.0'\n"
            "    meta:\n"
            "      agent_id: <your_agent_id>\n"
            "      created_at: <date>\n"
            "    identity:\n"
            "      identity: <your identity text>\n"
            "      core_belief: <your core belief text>\n"
            "    ocean:\n"
            "      openness: 0.5\n"
            "      conscientiousness: 0.5\n"
            "      extraversion: 0.5\n"
            "      agreeableness: 0.5\n"
            "      neuroticism: 0.5\n"
            "    tone_matrix:\n"
            "      warmth: 50\n"
            "      formality: 50\n"
            "      humor: 0.1\n"
            "    behavior_guardrails:\n"
            "      - ...\n"
            "\n"
            "  See agent_config.example.yaml for a full template.\n"
        )
        raise SystemExit(1)

    return MindConfig.model_validate(data)
