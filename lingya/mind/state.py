from __future__ import annotations

from pydantic import BaseModel, Field

from lingya.mind.config import BigFiveTraits, MindConfig


class PADPoint(BaseModel):
    """Current PAD position in 3D space."""
    pleasure: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)
    dominance: float = Field(default=0.0, ge=-1.0, le=1.0)


class MindState(BaseModel):
    """Fully serializable runtime state of the mind.

    Persisted to SQLite across CLI restarts for long-term personality evolution.
    """

    current_pad: PADPoint = Field(default_factory=PADPoint)
    pad_history: list[PADPoint] = Field(default_factory=list)

    current_ocean: BigFiveTraits = Field(default_factory=BigFiveTraits)

    recent_emotions: list[dict] = Field(default_factory=list)

    turn_counter: int = 0

    ipc_agency: float = 0.5
    ipc_communion: float = 0.5
    ipc_state: str = "neutral"

    cumulative_importance: float = 0.0
    reflection_threshold: float = 150.0
    self_notions: list[str] = Field(default_factory=list)

    reanchor_needed: bool = False
    reanchor_hint: str = ""

    @classmethod
    def from_config(cls, config: MindConfig) -> MindState:
        return cls(
            current_pad=PADPoint(
                pleasure=config.pad_baseline.pleasure,
                arousal=config.pad_baseline.arousal,
                dominance=config.pad_baseline.dominance,
            ),
            current_ocean=config.ocean.model_copy(deep=True),
        )

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> MindState:
        return cls.model_validate(d)
