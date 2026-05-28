from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone


class MemoryStore:
    """ChromaDB-backed semantic memory for user facts and preferences."""

    def __init__(self, persist_path: str) -> None:
        self._persist_path = persist_path
        self._client = None
        self._collection = None

    def warmup(self) -> None:
        """Pre-load the embedding model at startup so the first user message
        doesn't stall on model download and leak a progress bar into the chat.
        """
        import io
        import os
        from contextlib import redirect_stderr

        old = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            with redirect_stderr(io.StringIO()):
                self._ensure_client()
                try:
                    self._collection.add(documents=["warmup"], ids=["_warmup"])
                    self._collection.delete(ids=["_warmup"])
                except Exception:
                    pass
        finally:
            if old is not None:
                os.environ["TQDM_DISABLE"] = old
            else:
                os.environ.pop("TQDM_DISABLE", None)

    def _ensure_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_path)
            self._collection = self._client.get_or_create_collection(
                name="lingya_memory",
                metadata={"hnsw:space": "cosine"},
            )

    @property
    def collection(self):
        self._ensure_client()
        return self._collection

    def store(self, text: str) -> str:
        """Store a memory. Returns the entry ID."""
        col = self.collection
        entry_id = f"mem_{time.monotonic_ns()}"
        col.add(
            documents=[text],
            ids=[entry_id],
            metadatas=[{"timestamp": datetime.now(timezone.utc).isoformat()}],
        )
        return entry_id

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """Semantic search over stored memories."""
        col = self.collection
        count = col.count()
        if count == 0:
            return []
        n = min(top_k, count)
        results = col.query(query_texts=[query], n_results=n)
        items: list[dict[str, str]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        for mem_id, doc in zip(ids, docs):
            items.append({"id": mem_id, "text": doc})
        return items

    def list_all(self) -> list[dict[str, str]]:
        """List all stored memories with their IDs."""
        col = self.collection
        if col.count() == 0:
            return []
        results = col.get()
        items: list[dict[str, str]] = []
        for mem_id, doc in zip(results.get("ids", []), results.get("documents", [])):
            if doc:
                items.append({"id": mem_id, "text": doc})
        return items

    def delete(self, mem_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        col = self.collection
        col.delete(ids=[mem_id])
        return True


class EnhancedMemoryStore(MemoryStore):
    """MemoryStore with importance scoring and weighted retrieval.

    Extends MemoryStore — all existing methods unchanged.
    """

    def store_with_importance(self, text: str, importance: float = 5.0) -> str:
        """Store a memory with an importance score (1-10)."""
        col = self.collection
        entry_id = f"mem_{time.monotonic_ns()}"
        col.add(
            documents=[text],
            ids=[entry_id],
            metadatas=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance": importance,
            }],
        )
        return entry_id

    async def score_importance(
        self,
        text: str,
        llm_call: Callable[[str], Awaitable[str]],
    ) -> float:
        """Use LLM to score memory importance on a 1-10 scale."""
        prompt = (
            "Rate the importance of this information for long-term memory about a user. "
            "1 = trivial small talk, 10 = deeply personal/identity-defining.\n\n"
            f"Text: {text}\n\n"
            "Return ONLY a number 1-10."
        )
        try:
            response = await llm_call(prompt)
            score = float(response.strip())
            return max(1.0, min(10.0, score))
        except Exception:
            return 5.0

    def search_weighted(
        self,
        query: str,
        top_k: int = 5,
        recency_lambda: float = 0.01,
    ) -> list[dict]:
        """Weighted search: score = exp(-lambda * hours_since) * importance * similarity."""
        col = self.collection
        count = col.count()
        if count == 0:
            return []

        n = min(top_k * 3, count)  # Fetch more, re-rank
        results = col.query(query_texts=[query], n_results=n)
        items: list[dict] = []
        now = datetime.now(timezone.utc)
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for mem_id, doc, meta in zip(ids, docs, metas):
            importance = float(meta.get("importance", 5.0)) if meta else 5.0
            ts_str = meta.get("timestamp", "") if meta else ""
            hours_since = 0.0
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    hours_since = (now - ts).total_seconds() / 3600.0
                except ValueError:
                    pass
            recency_weight = math.exp(-recency_lambda * hours_since)
            combined = recency_weight * importance
            items.append({
                "id": mem_id,
                "text": doc,
                "importance": importance,
                "hours_since": hours_since,
                "weighted_score": combined,
            })

        items.sort(key=lambda x: x["weighted_score"], reverse=True)
        return items[:top_k]

    def get_cumulative_importance(self) -> float:
        """Sum importance of all stored memories."""
        col = self.collection
        if col.count() == 0:
            return 0.0
        results = col.get()
        total = 0.0
        for meta in results.get("metadatas", []):
            if meta and "importance" in meta:
                total += float(meta["importance"])
        return total
