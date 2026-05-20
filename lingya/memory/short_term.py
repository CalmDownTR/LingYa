from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ShortTermMemory:
    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self._messages: deque[Message] = deque()

    def add(self, message: Message) -> None:
        self._messages.append(message)
        # Hard cap safety guard — pops oldest to prevent unbounded growth
        while len(self._messages) > self.max_messages:
            self._messages.popleft()

    def prepend(self, message: Message) -> None:
        """Insert a message at the front of the deque (used for compression summaries)."""
        self._messages.appendleft(message)

    def get_messages(self, last_n: int | None = None) -> list[Message]:
        if last_n is None:
            return list(self._messages)
        return list(self._messages)[-last_n:]

    def get_conversation_text(self) -> str:
        parts: list[str] = []
        for m in self._messages:
            parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)

    def pop_compressible(self, count: int) -> list[Message]:
        """Pop the oldest `count` messages from the front for compression."""
        popped: list[Message] = []
        for _ in range(min(count, len(self._messages))):
            popped.append(self._messages.popleft())
        return popped

    def clear(self) -> None:
        self._messages.clear()

    def token_count_estimate(self) -> int:
        # Rough estimate: ~2 chars per token
        total_chars = sum(len(m.content) for m in self._messages)
        return total_chars // 2

    def __len__(self) -> int:
        return len(self._messages)
