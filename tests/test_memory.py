from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lingya.memory import MemoryStore


@pytest.fixture
def store():
    """MemoryStore backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = MemoryStore(persist_path=str(Path(tmpdir) / "chroma"))
        yield s


def _store_many(store: MemoryStore, texts: list[str]) -> list[str]:
    """Store multiple memories, each with a unique timestamp."""
    ids = []
    base = datetime.now(timezone.utc).timestamp()
    for i, text in enumerate(texts):
        fake_dt = datetime.fromtimestamp(base + i, tz=timezone.utc)
        with patch("lingya.memory.store.datetime") as mock_dt:
            mock_dt.now.return_value = fake_dt
            mock_dt.timezone = timezone
            ids.append(store.store(text))
    return ids


class TestMemoryStore:
    def test_store_returns_id(self, store):
        mem_id = store.store("Hello world")
        assert mem_id.startswith("mem_")

    def test_store_and_list_all(self, store):
        _store_many(store, ["First memory", "Second memory"])
        items = store.list_all()
        assert len(items) == 2
        texts = {item["text"] for item in items}
        assert texts == {"First memory", "Second memory"}

    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_search_returns_relevant_results(self, store):
        _store_many(store, [
            "Alice likes Python programming",
            "Bob enjoys cooking pasta",
            "Charlie plays guitar",
        ])

        results = store.search("coding and software", top_k=2)
        assert len(results) == 2
        assert "Python" in results[0]["text"]

    def test_search_empty_collection(self, store):
        assert store.search("anything") == []

    def test_search_respects_top_k(self, store):
        _store_many(store, [f"Memory number {i}" for i in range(5)])

        results = store.search("memory", top_k=3)
        assert len(results) == 3

    def test_delete_removes_memory(self, store):
        mem_id = store.store("Temporary memory")
        assert len(store.list_all()) == 1

        store.delete(mem_id)
        assert store.list_all() == []

    def test_delete_nonexistent_does_not_raise(self, store):
        store.delete("mem_nonexistent")

    def test_warmup_initializes_client(self, store):
        store.warmup()
        # After warmup, the client should be usable
        store.store("Post-warmup memory")
        assert len(store.list_all()) == 1
