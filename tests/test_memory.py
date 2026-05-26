from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lingya.memory import MemoryStore


@pytest.fixture
def store():
    """MemoryStore backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = MemoryStore(persist_path=str(Path(tmpdir) / "chroma"))
        yield s


class TestMemoryStore:
    def test_store_returns_id(self, store):
        mem_id = store.store("Hello world")
        assert mem_id.startswith("mem_")

    def test_store_and_list_all(self, store):
        store.store("First memory")
        store.store("Second memory")
        items = store.list_all()
        assert len(items) == 2
        texts = {item["text"] for item in items}
        assert texts == {"First memory", "Second memory"}

    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_search_returns_relevant_results(self, store):
        store.store("Alice likes Python programming")
        store.store("Bob enjoys cooking pasta")
        store.store("Charlie plays guitar")

        results = store.search("coding and software", top_k=2)
        assert len(results) == 2
        assert "Python" in results[0]["text"]

    def test_search_empty_collection(self, store):
        assert store.search("anything") == []

    def test_search_respects_top_k(self, store):
        for i in range(5):
            store.store(f"Memory number {i}")

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
