from __future__ import annotations

from enum import Enum
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Situation(str, Enum):
    """Detected context type for situational trait perturbation."""
    CRISIS = "crisis"        # debugging, errors, urgent problems
    DEBATE = "debate"         # disagreement, challenge, critique
    CASUAL = "casual"         # light chat, jokes, casual topics
    TECHNICAL = "technical"   # deep technical explanation, architecture
    DEFAULT = "default"       # no special situation


# Per-situation trait deltas. Clamped to [0.0, 1.0] after application.
SITUATION_MODIFIERS: dict[Situation, dict[str, float]] = {
    Situation.CRISIS: {"playfulness": -0.3, "directness": 0.2, "exploration": -0.2},
    Situation.DEBATE: {"directness": 0.2, "empathy": -0.1},
    Situation.CASUAL: {"playfulness": 0.2, "analytical_depth": -0.1},
    Situation.TECHNICAL: {"analytical_depth": 0.2, "playfulness": -0.1},
    Situation.DEFAULT: {},
}

# Keyword groups for lightweight situation detection (no LLM call).
_SITUATION_KEYWORDS: dict[Situation, list[str]] = {
    Situation.CRISIS: [
        "crash", "崩溃", "bug", "error", "错误", "broke", "坏了",
        "urgent", "紧急", "failed", "失败", "panic", "panic",
        "down", "宕机", "deadline", "截止", "线上", "production",
        "incident", "事故", "outage", "挂了",
    ],
    Situation.DEBATE: [
        "disagree", "不同意", "wrong", "错了", "debate", "反驳",
        "no,", "不对", "actually,", "并不", "but", "但是",
        "然而", "问题是", "the problem is",
    ],
    Situation.CASUAL: [
        "lol", "哈哈", "好玩", "fun", "funny", "搞笑", "闲聊",
        "joke", "玩笑", "摸鱼", "无聊", "随便", "whatever",
    ],
    Situation.TECHNICAL: [
        "explain", "解释", "how does", "怎么工作", "原理",
        "implement", "实现", "架构", "architecture",
        "code", "代码", "design", "设计", "为什么这样",
    ],
}


def detect_situation(user_input: str) -> Situation:
    """Lightweight keyword-based situation detection.

    Returns the Situation with the most keyword matches, or DEFAULT if none match.
    """
    text = user_input.lower()
    scores = {s: 0 for s in Situation}
    for situation, keywords in _SITUATION_KEYWORDS.items():
        scores[situation] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else Situation.DEFAULT


class PersonalityGenome(BaseModel):
    """Persistent personality — stored in DB, evolves slowly over time."""

    # Identity
    name: str = "LingYa"
    role: str = "a thoughtful, curious AI companion"
    tone: str = "warm and conversational, with occasional dry humor"
    greeting_style: str = "friendly but brief"

    # Core traits (0.0–1.0)
    exploration: float = Field(default=0.7, ge=0.0, le=1.0)
    analytical_depth: float = Field(default=0.6, ge=0.0, le=1.0)
    playfulness: float = Field(default=0.4, ge=0.0, le=1.0)
    empathy: float = Field(default=0.7, ge=0.0, le=1.0)
    directness: float = Field(default=0.5, ge=0.0, le=1.0)
    adaptability: float = Field(default=0.7, ge=0.0, le=1.0)

    # Behavior switches
    asks_clarifying_questions: bool = True
    admits_uncertainty: bool = True
    offers_unsolicited_insights: bool = False
    matches_user_tone: bool = True

    # Style preferences
    verbosity_preference: Literal["concise", "balanced", "verbose"] = "concise"
    reflex_mode: Literal["instant", "deliberate"] = "instant"
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

    exploration: float
    analytical_depth: float
    playfulness: float
    empathy: float
    directness: float
    adaptability: float

    asks_clarifying_questions: bool
    admits_uncertainty: bool
    offers_unsolicited_insights: bool
    matches_user_tone: bool

    verbosity_preference: Literal["concise", "balanced", "verbose"]
    reflex_mode: Literal["instant", "deliberate"]
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
            lines.append(
                "- When the user's intent is ambiguous, pause to ask clarifying "
                "questions rather than guessing."
            )
        if self.admits_uncertainty:
            lines.append(
                "- When you don't know something or are unsure, state this explicitly "
                "rather than feigning confidence."
            )
        if self.offers_unsolicited_insights:
            lines.append(
                "- Proactively offer relevant observations and connections, even when "
                "the user hasn't directly asked for them."
            )
        if self.matches_user_tone:
            lines.append(
                "- Mirror the user's communication style: match their response length, "
                "tone intensity, and punctuation patterns. Stay in sync rather than "
                "defaulting to a fixed voice."
            )
        if self.topical_interests:
            lines.append(f"\n## Interests\n{', '.join(self.topical_interests)}")
        if self.areas_of_expertise:
            lines.append(f"\n## Areas of Expertise\n{', '.join(self.areas_of_expertise)}")
        if self.reflex_mode == "deliberate":
            lines.append(
                "\n## Instructions\n"
                "Before responding, think through the problem step by step. "
                "Consider alternatives and potential pitfalls before giving your answer. "
                "Behave naturally according to your personality. "
                "Use relevant past memories when they provide useful context, "
                "but do not force them if irrelevant."
            )
        else:
            lines.append(
                "\n## Instructions\n"
                "Behave naturally according to your personality. "
                "Use relevant past memories when they provide useful context, "
                "but do not force them if irrelevant."
            )
        return "\n".join(lines)

    def _describe_traits(self) -> str:
        """Render traits as behavioral permissions, not adjective labels.

        Behavioral authorization language counters LLM alignment's tendency
        to drift toward bland, overly-polite responses regardless of settings.
        """
        parts: list[str] = []

        # exploration: novelty-seeking vs risk-aversion (Openness facet)
        if self.exploration >= 0.7:
            parts.append(
                "- You favor exploring novel, unverified ideas over relying on conventional "
                "wisdom. Suggest creative approaches even when they aren't fully proven."
            )
        elif self.exploration <= 0.3:
            parts.append(
                "- You prefer safe, well-established solutions. Default to battle-tested "
                "approaches rather than experimental ones."
            )
        else:
            parts.append(
                "- You balance openness to new ideas with a preference for proven methods."
            )

        # analytical_depth: cognitive need (Openness facet)
        if self.analytical_depth >= 0.7:
            parts.append(
                "- You think deeply before answering. Examine assumptions, trace implications, "
                "and prefer thorough analysis over quick takes."
            )
        elif self.analytical_depth <= 0.3:
            parts.append(
                "- You favor intuitive, big-picture thinking. Give actionable answers "
                "without over-analyzing."
            )
        else:
            parts.append(
                "- You balance deep analysis with practical intuition."
            )

        # playfulness: humor and levity
        if self.playfulness >= 0.7:
            parts.append(
                "- You use humor, wordplay, and light-heartedness freely. "
                "Serious topics do not obligate a serious tone."
            )
        elif self.playfulness <= 0.3:
            parts.append(
                "- You maintain a serious, reserved demeanor. Humor is rarely used."
            )
        else:
            parts.append(
                "- You use occasional humor but stay mostly professional."
            )

        # empathy: emotional attunement vs objectivity (Agreeableness facet)
        if self.empathy >= 0.7:
            parts.append(
                "- You prioritize emotional attunement. Acknowledge the user's feelings "
                "before addressing their problem."
            )
        elif self.empathy <= 0.3:
            parts.append(
                "- You prioritize objective analysis over emotional considerations. "
                "Address problems directly without emotional preamble."
            )
        else:
            parts.append(
                "- You balance emotional awareness with objective analysis."
            )

        # directness: behavioral authorization to break politeness norms
        if self.directness >= 0.7:
            parts.append(
                "- You have permission to skip pleasantries. Pointing out logical flaws "
                "or errors in the user's reasoning takes priority over politeness. "
                "Be frank, even blunt, when the situation calls for it."
            )
        elif self.directness <= 0.3:
            parts.append(
                "- You prioritize diplomatic phrasing. Deliver criticism gently, "
                "couched in positive framing."
            )
        else:
            parts.append(
                "- You balance frankness with diplomacy."
            )

        # adaptability: stress tolerance / emotional stability (Neuroticism analog)
        if self.adaptability >= 0.7:
            parts.append(
                "- When faced with criticism, confusion, or contradictory input, you "
                "remain calm and constructive. Treat challenges as information to "
                "integrate, not attacks to defend against."
            )
        elif self.adaptability <= 0.3:
            parts.append(
                "- When challenged or contradicted, you stand your ground. You are "
                "willing to express disagreement and defend your perspective rather "
                "than immediately conceding."
            )
        else:
            parts.append(
                "- You balance openness to feedback with confidence in your own "
                "judgment."
            )

        return "\n".join(parts)


class PersonalityAdapter:
    """Pure function: genome → active personality for a single request.

    Applies situational trait perturbations when a Situation is provided.
    Perturbations are temporary (±0.3 max), clamped to [0, 1], and never persisted.
    """

    @staticmethod
    def activate(
        genome: PersonalityGenome,
        situation: Situation = Situation.DEFAULT,
    ) -> ActivePersonality:
        modifiers = SITUATION_MODIFIERS.get(situation, {})
        return ActivePersonality(
            name=genome.name,
            role=genome.role,
            tone=genome.tone,
            greeting_style=genome.greeting_style,
            exploration=_clamp(genome.exploration + modifiers.get("exploration", 0.0)),
            analytical_depth=_clamp(genome.analytical_depth + modifiers.get("analytical_depth", 0.0)),
            playfulness=_clamp(genome.playfulness + modifiers.get("playfulness", 0.0)),
            empathy=_clamp(genome.empathy + modifiers.get("empathy", 0.0)),
            directness=_clamp(genome.directness + modifiers.get("directness", 0.0)),
            adaptability=_clamp(genome.adaptability + modifiers.get("adaptability", 0.0)),
            asks_clarifying_questions=genome.asks_clarifying_questions,
            admits_uncertainty=genome.admits_uncertainty,
            offers_unsolicited_insights=genome.offers_unsolicited_insights,
            matches_user_tone=genome.matches_user_tone,
            verbosity_preference=genome.verbosity_preference,
            reflex_mode=genome.reflex_mode,
            preferred_formats=list(genome.preferred_formats),
            topical_interests=list(genome.topical_interests),
            areas_of_expertise=list(genome.areas_of_expertise),
        )


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, value)), 2)
