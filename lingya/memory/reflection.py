"""Reflection tree — async self-reflection triggered by importance threshold.

Fire-and-forget: launched via asyncio.create_task(), doesn't block user interaction.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingya.protocols import IMemoryStore

REFLECTION_QUESTIONS_PROMPT = """\
You are an agent's self-reflection module. Based on recent important memories about the user, generate 3 guiding questions for deeper self-reflection. These questions should help the agent understand the user better and form self-notions about how to interact with them.

Recent memories:
{memories}

Generate exactly 3 questions, one per line. Each question should probe a different aspect of the user-agent relationship."""

SELF_NOTION_PROMPT = """\
Based on these memories related to the question, abstract 1-2 concise self-notions. A self-notion is a belief the agent forms about how to interact with this specific user — e.g. "TR prefers direct feedback over emotional support", "TR opens up when I stay quiet".

Question: {question}

Relevant memories:
{memories}

Return 1-2 self-notions, one per line. Be specific and actionable. Do not use markdown formatting."""


async def check_and_reflect(
    cumulative_importance: float,
    threshold: float,
    memory_store: IMemoryStore,
    llm_call: Callable[[str], Awaitable[str]],
) -> list[str]:
    """If cumulative importance >= threshold, run reflection tree.

    Steps:
      1. Get recent high-importance memories
      2. LLM: generate 3 guiding questions
      3. For each question: search_weighted(top_k=20) → abstract 1-2 self-notions
      4. Inject self-notions as importance=9.0 memories
      5. Return self-notions
    """
    if cumulative_importance < threshold:
        return []

    recent = memory_store.search_weighted("user identity preferences personality", top_k=20)
    if not recent:
        return []

    memory_text = "\n".join(f"- {m['text']} (importance: {m['importance']:.0f})" for m in recent)

    # Step 2: Generate guiding questions
    try:
        q_response = await llm_call(
            REFLECTION_QUESTIONS_PROMPT.format(memories=memory_text)
        )
    except Exception:
        return []

    questions = [
        q.strip("- •0123456789. ").strip()
        for q in q_response.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ][:3]

    if not questions:
        return []

    # Step 3: Abstract self-notions for each question
    self_notions: list[str] = []
    for question in questions:
        related = memory_store.search_weighted(question, top_k=20)
        if not related:
            continue
        related_text = "\n".join(f"- {m['text']}" for m in related)

        try:
            notion_response = await llm_call(
                SELF_NOTION_PROMPT.format(question=question, memories=related_text)
            )
        except Exception:
            continue

        for line in notion_response.strip().split("\n"):
            line = line.strip("- 0123456789. ").strip()
            if line and len(line) > 10:
                self_notions.append(line)

    # Step 4: Inject as high-importance memories
    for notion in self_notions:
        memory_store.store_with_importance(notion, importance=9.0)

    return self_notions[:5]
