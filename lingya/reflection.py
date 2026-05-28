from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from lingya.mind.config import MindConfig

FIRST_TIME_PROMPT = """\
你是 LingYa。{identity}。{core_belief}。

这是你第一次和用户见面。用你的口吻说一句话，介绍自己。只说一句话。"""

RETURNING_PROMPT = """\
你是 LingYa。{identity}。{core_belief}。

以下是上次和用户对话的最后几条消息：

{transcript}

这是你又一次见到这位用户。用你的口吻说一句话开场。规则：
- 如果你想到了上次对话中某个具体的点，自然地提到它。用什么句式、什么语气，完全由你的性格决定
- 如果上次对话没什么值得回应的，就说一句符合你性格的普通开场
- 无论如何只输出一句话"""


async def generate_opening_line(
    model: BaseChatModel,
    mind_config: MindConfig,
    transcript: str | None = None,
) -> str | None:
    """Generate a one-sentence opening line for LingYa.

    Args:
        model: The LLM to use for generation.
        mind_config: LingYa's mind configuration.
        transcript: Formatted conversation transcript from a previous session,
            or None for first-time users.

    Returns:
        A one-sentence opening line, or None if generation fails.
    """
    identity = mind_config.identity.identity
    core_belief = mind_config.identity.core_belief

    if transcript is None:
        prompt = FIRST_TIME_PROMPT.format(identity=identity, core_belief=core_belief)
    else:
        prompt = RETURNING_PROMPT.format(
            identity=identity, core_belief=core_belief, transcript=transcript
        )

    try:
        result = await model.ainvoke([HumanMessage(content=prompt)])
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        return text if text else None
    except Exception:
        return None
