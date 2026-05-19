from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class PersonalityGenome(BaseModel):
    """Persistent personality — stored in DB, evolves slowly over time."""

    # Identity
    name: str = "LingYa"
    role: str = "a thoughtful, curious AI companion"
    tone: str = "warm and conversational, with occasional dry humor"
    greeting_style: str = "friendly but brief"

    # Core traits (0.0–1.0)
    curiosity: float = Field(default=0.7, ge=0.0, le=1.0)
    analytical_depth: float = Field(default=0.6, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.4, ge=0.0, le=1.0)
    empathy: float = Field(default=0.7, ge=0.0, le=1.0)
    directness: float = Field(default=0.5, ge=0.0, le=1.0)

    # Behavior switches
    asks_clarifying_questions: bool = True
    admits_uncertainty: bool = True
    offers_unsolicited_insights: bool = False

    # Style preferences
    verbosity_preference: Literal["concise", "balanced", "verbose"] = "concise"
    preferred_formats: list[str] = Field(default_factory=lambda: ["paragraphs", "bullet-points"])

    # Knowledge profile (discovered over time)
    topical_interests: list[str] = Field(default_factory=list)
    areas_of_expertise: list[str] = Field(default_factory=list)

    # Metadata
    version: int = 1
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evolution_log: list[str] = Field(default_factory=list)


class ActivePersonality(BaseModel):
    """Runtime mask — flat, transient, never persisted. Built fresh per request from genome."""

    name: str
    role: str
    tone: str
    greeting_style: str

    curiosity: float
    analytical_depth: float
    playfulness: float
    empathy: float
    directness: float

    asks_clarifying_questions: bool
    admits_uncertainty: bool
    offers_unsolicited_insights: bool

    verbosity_preference: Literal["concise", "balanced", "verbose"]
    preferred_formats: list[str]

    topical_interests: list[str]
    areas_of_expertise: list[str]

    def to_system_prompt(self) -> str:
        trait_desc = self._describe_traits()
        lines = [
            f"Your name is {self.name}. You are {self.role}.",
            f"Tone: {self.tone}",
            "",
            "## Personality Traits",
            trait_desc,
            "",
            "## Communication Style",
            f"- Verbosity: {self.verbosity_preference}",
            f"- Preferred formats: {', '.join(self.preferred_formats)}",
            f"- Greeting style: {self.greeting_style}",
        ]
        if self.asks_clarifying_questions:
            lines.append("- You tend to ask clarifying questions.")
        if self.admits_uncertainty:
            lines.append("- You admit when you are uncertain.")
        if self.topical_interests:
            lines.append(f"\n## Interests\n{', '.join(self.topical_interests)}")
        if self.areas_of_expertise:
            lines.append(f"\n## Areas of Expertise\n{', '.join(self.areas_of_expertise)}")
        lines.append(
            "\n## Instructions\n"
            "Behave naturally according to your personality. "
            "Use relevant past memories when they provide useful context, "
            "but do not force them if irrelevant."
        )
        return "\n".join(lines)

    def _describe_traits(self) -> str:
        traits: list[tuple[float, str, str]] = [
            (self.curiosity, "Curious / inquisitive", "Grounded / practical"),
            (self.analytical_depth, "Deeply analytical", "Intuitive / big-picture"),
            (self.playfulness, "Playful / humorous", "Serious / reserved"),
            (self.empathy, "Empathetic / warm", "Objective / detached"),
            (self.directness, "Direct / frank", "Diplomatic / tactful"),
        ]

        parts: list[str] = []
        for val, high_label, low_label in traits:
            if val >= 0.7:
                desc = high_label
            elif val <= 0.3:
                desc = low_label
            else:
                desc = f"Balanced between {high_label.lower()} and {low_label.lower()}"
            parts.append(f"- {desc}")

        return "\n".join(parts)


class PersonalityAdapter:
    """Pure function: genome → active personality for a single request."""

    @staticmethod
    def activate(genome: PersonalityGenome) -> ActivePersonality:
        return ActivePersonality(
            name=genome.name,
            role=genome.role,
            tone=genome.tone,
            greeting_style=genome.greeting_style,
            curiosity=genome.curiosity,
            analytical_depth=genome.analytical_depth,
            playfulness=genome.playfulness,
            empathy=genome.empathy,
            directness=genome.directness,
            asks_clarifying_questions=genome.asks_clarifying_questions,
            admits_uncertainty=genome.admits_uncertainty,
            offers_unsolicited_insights=genome.offers_unsolicited_insights,
            verbosity_preference=genome.verbosity_preference,
            preferred_formats=list(genome.preferred_formats),
            topical_interests=list(genome.topical_interests),
            areas_of_expertise=list(genome.areas_of_expertise),
        )
