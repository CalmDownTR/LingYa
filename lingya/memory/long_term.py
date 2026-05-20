from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from lingya.ingestion.embedder import get_embedder


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class LongTermMemory:
    def __init__(
        self,
        persist_dir: str,
        embedding_model_name: str,
        collection_name: str = "memories",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._embedder = get_embedder(embedding_model_name)
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    def _ensure_initialized(self) -> None:
        if self._client is not None:
            return
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        self._ensure_initialized()
        assert self._collection is not None
        return self._collection

    async def store(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        texts = [e.text for e in entries]
        embeddings = await self._embedder.encode(texts)
        ids = [e.id or str(uuid.uuid4()) for e in entries]
        metadatas = [e.metadata for e in entries]

        await asyncio.to_thread(
            self.collection.add,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def search(
        self, query: str, top_k: int = 5, filter: dict | None = None
    ) -> list[MemoryEntry]:
        self._ensure_initialized()
        query_embedding = await self._embedder.encode([query])
        where_filter = filter if filter else None

        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        entries: list[MemoryEntry] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                entries.append(MemoryEntry(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                ))
        return entries

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await asyncio.to_thread(self.collection.delete, ids=ids)

    async def count(self) -> int:
        return await asyncio.to_thread(self.collection.count)
