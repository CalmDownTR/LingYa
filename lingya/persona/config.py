from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class MindCore(BaseModel):
    identity: str
    core_belief: str


class ToneMatrix(BaseModel):
    warmth: int = Field(ge=0, le=100)
    formality: int = Field(ge=0, le=100)


class PersonaMeta(BaseModel):
    agent_id: str
    created_at: str


class PersonaConfig(BaseModel):
    version: str
    meta: PersonaMeta
    mind_core: MindCore
    tone_matrix: ToneMatrix
    behavior_guardrails: list[str]


def load_persona_config(path: str | Path = "agent_config.yaml") -> PersonaConfig:
    config_path = Path(path)
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return PersonaConfig.model_validate(data)
