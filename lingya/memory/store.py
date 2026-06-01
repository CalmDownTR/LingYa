from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone


# ── Rule-based importance pre-scoring ───────────────────────────────────

_IDENTITY_KEYWORDS = [
    "我是", "我喜欢", "我讨厌", "我的名字", "我住", "我工作",
    "我害怕", "我想", "我希望", "我决定", "我不再", "我改名",
    "我从小", "我父母", "我的梦想", "我放弃",
]


def rule_based_importance(text: str) -> float:
    """Fast rule-based importance pre-score (1-10).

    Used immediately for memory storage; LLM refines the score asynchronously.
    """
    score = 5.0
    if len(text) > 200:
        score += 1.0
    if len(text) > 500:
        score += 1.0
    for kw in _IDENTITY_KEYWORDS:
        if kw in text:
            score += 0.5
    return max(1.0, min(10.0, score))


# ── MemoryStore ────────────────────────────────────────────────────────


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
        """Store a memory with an importance score (1-10).

        Sets retrieval_weight = importance initially and archived = False.
        """
        col = self.collection
        entry_id = f"mem_{time.monotonic_ns()}"
        col.add(
            documents=[text],
            ids=[entry_id],
            metadatas=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance": importance,
                "retrieval_weight": importance,
                "archived": False,
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
        """Weighted search: score = exp(-lambda * hours_since) * retrieval_weight * similarity.

        Uses retrieval_weight from metadata. Falls back to importance for
        backward compatibility with memories stored before v0.6.

        Excludes archived memories from search results.
        """
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
            if meta:
                # Exclude archived memories
                archived_val = meta.get("archived", False)
                if archived_val is True or archived_val == "true":
                    continue

                importance = float(meta.get("importance", 5.0))
                # Use retrieval_weight, fall back to importance for backward compat
                retrieval_weight = float(
                    meta.get("retrieval_weight", importance)
                )
                ts_str = meta.get("timestamp", "")
            else:
                importance = 5.0
                retrieval_weight = 5.0
                ts_str = ""

            hours_since = 0.0
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    hours_since = (now - ts).total_seconds() / 3600.0
                except ValueError:
                    pass
            recency_weight = math.exp(-recency_lambda * hours_since)
            combined = recency_weight * retrieval_weight
            items.append({
                "id": mem_id,
                "text": doc,
                "importance": importance,
                "retrieval_weight": retrieval_weight,
                "hours_since": hours_since,
                "weighted_score": combined,
            })

        items.sort(key=lambda x: x["weighted_score"], reverse=True)
        return items[:top_k]

    def update_importance(self, entry_id: str, importance: float) -> None:
        """Update the importance metadata for an existing memory entry."""
        col = self.collection
        col.update(ids=[entry_id], metadatas=[{"importance": importance}])

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

    # ── Memory Decay ──────────────────────────────────────────────────

    def apply_decay(self) -> int:
        """Apply three-level decay to all non-critical memories.

        Called periodically by BackgroundRunner.  Returns the number of
        memories affected (decayed + archived + hard-deleted).

        Three levels:

        * Critical (importance > 0.8): retrieval_weight locked, never decays.
          If retrieval_weight is missing (backward compat), it is set to
          importance on the first call.

        * Normal (0.3 <= importance <= 0.8): linear decay:
          retrieval_weight = importance * max(0, 1 - days / 180).
          Day 90 = half-life; day 180 = weight reaches 0.

        * Micro (importance < 0.3): 30 days -> soft delete (mark archived,
          retrieval_weight = 0).  Already-archived memories that have been
          archived for 90+ days are hard-deleted from ChromaDB.
        """
        col = self.collection
        count = col.count()
        if count == 0:
            return 0

        results = col.get()
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        if not ids:
            return 0

        now = datetime.now(timezone.utc)
        affected = 0
        ids_to_delete: list[str] = []
        ids_to_update: list[str] = []
        new_metadatas: list[dict] = []

        for mem_id, meta in zip(ids, metadatas):
            if not meta:
                continue

            importance = float(meta.get("importance", 5.0))
            ts_str = meta.get("timestamp", "")
            if not ts_str:
                continue

            try:
                ts = datetime.fromisoformat(ts_str)
                days_since = (now - ts).total_seconds() / 86400.0
            except ValueError:
                continue

            # ── Critical: locked ──────────────────────────────────
            if importance > 0.8:
                rw = meta.get("retrieval_weight")
                if rw is None:
                    # Backward compat: first run sets retrieval_weight
                    meta_copy = dict(meta)
                    meta_copy["retrieval_weight"] = importance
                    ids_to_update.append(mem_id)
                    new_metadatas.append(meta_copy)
                    affected += 1
                continue

            # ── Already archived? Check hard-delete threshold ──────
            archived_val = meta.get("archived", False)
            is_archived = archived_val is True or archived_val == "true"
            if is_archived:
                archived_at_str = meta.get("archived_at", "")
                if archived_at_str:
                    try:
                        archived_at = datetime.fromisoformat(archived_at_str)
                        archived_days = (
                            now - archived_at
                        ).total_seconds() / 86400.0
                        if archived_days >= 90:
                            ids_to_delete.append(mem_id)
                            affected += 1
                    except ValueError:
                        pass
                continue

            # ── Micro: archive after 30 days ──────────────────────
            if importance < 0.3 and days_since >= 30:
                meta_copy = dict(meta)
                meta_copy["archived"] = True
                meta_copy["archived_at"] = now.isoformat()
                meta_copy["retrieval_weight"] = 0.0
                ids_to_update.append(mem_id)
                new_metadatas.append(meta_copy)
                affected += 1
                continue

            # ── Normal: linear decay ──────────────────────────────
            new_weight = importance * max(0.0, 1.0 - days_since / 180.0)
            current_weight = meta.get("retrieval_weight")
            if current_weight is None:
                current_weight_float = None
            else:
                current_weight_float = float(current_weight)

            if (
                current_weight_float is None
                or abs(current_weight_float - new_weight) > 0.0001
            ):
                meta_copy = dict(meta)
                meta_copy["retrieval_weight"] = round(new_weight, 4)
                ids_to_update.append(mem_id)
                new_metadatas.append(meta_copy)
                affected += 1

        # Apply hard deletes
        for mem_id in ids_to_delete:
            col.delete(ids=[mem_id])

        # Apply metadata updates
        if ids_to_update:
            col.update(ids=ids_to_update, metadatas=new_metadatas)

        return affected

    def recover(self, mem_id: str) -> bool:
        """Recover an archived memory.

        Resets retrieval_weight to importance and clears the archived flag.
        Returns True if the memory was recovered, False if not found.
        """
        col = self.collection
        results = col.get(ids=[mem_id])
        if not results or not results.get("ids"):
            return False

        meta = results["metadatas"][0] if results["metadatas"] else {}
        if not meta:
            return False

        importance = float(meta.get("importance", 5.0))
        new_meta = dict(meta)
        new_meta["retrieval_weight"] = importance
        new_meta["archived"] = False
        new_meta["archived_at"] = ""  # clear (ChromaDB merges, not replaces)

        col.update(ids=[mem_id], metadatas=[new_meta])
        return True
