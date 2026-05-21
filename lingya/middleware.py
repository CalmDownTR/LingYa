from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

from lingya.personality.engine import PersonalityEngine


class PersonalityMiddleware(AgentMiddleware):
    """Inject behavioral authorization language before each model call.

    Detects situation from the latest user message, activates the personality
    adapter, and prepends the resulting behavioral prompt to the system message.
    """

    def __init__(self, engine: PersonalityEngine) -> None:
        self.engine = engine

    async def awrap_model_call(self, request, handler):
        user_input = _extract_last_user_text(request.messages)
        personality_prompt = self.engine.get_system_prompt(user_input)

        existing = request.system_message.content if request.system_message else ""
        new_system = personality_prompt + "\n\n" + (existing if existing else "")

        modified = request.override(
            system_message=SystemMessage(content=new_system)
        )
        return await handler(modified)


def _extract_last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                return str(first)
    return ""
