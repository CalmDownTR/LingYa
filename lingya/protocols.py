"""Protocol interfaces for decoupling modules.

IMemoryStore: interface that EnhancedMemoryStore implements, used by
  MindEngine, MessageRouter, reflection, and memory tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IMemoryStore(Protocol):
    """Memory store interface — 12 methods covering all callers.

    Callers:
      - MindEngine: store_with_importance, score_importance, update_importance
      - MindEngine state: get_cumulative_importance
      - MessageRouter: list_all, delete, recover
      - memory tools: store, search
      - reflection: search_weighted
      - BackgroundRunner: apply_decay
      - ApplicationBuilder: warmup
    """

    def warmup(self) -> None: ...

    def store(self, text: str) -> str: ...

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str]]: ...

    def list_all(self) -> list[dict[str, str]]: ...

    def delete(self, mem_id: str) -> bool: ...

    def recover(self, mem_id: str) -> bool: ...

    def store_with_importance(self, text: str, importance: float) -> str: ...

    async def score_importance(
        self, text: str, llm_call: Callable[[str], Awaitable[str]]
    ) -> float: ...

    def update_importance(self, entry_id: str, importance: float) -> None: ...

    def search_weighted(
        self, query: str, top_k: int = 5, recency_lambda: float = 0.01
    ) -> list[dict[str, Any]]: ...

    def apply_decay(self) -> int: ...

    def get_cumulative_importance(self) -> float: ...
