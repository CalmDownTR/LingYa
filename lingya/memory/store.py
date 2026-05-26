from __future__ import annotations

import time
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
