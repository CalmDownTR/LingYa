from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lingya.memory import EnhancedMemoryStore, MemoryStore


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


@pytest.fixture
def enhanced_store():
    """EnhancedMemoryStore backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = EnhancedMemoryStore(persist_path=str(Path(tmpdir) / "chroma_enhanced"))
        yield s


class TestEnhancedMemoryStore:
    def test_store_with_importance(self, enhanced_store):
        mem_id = enhanced_store.store_with_importance("Important fact", importance=9.0)
        assert mem_id.startswith("mem_")
        items = enhanced_store.list_all()
        assert len(items) == 1
        assert items[0]["text"] == "Important fact"

    def test_store_with_default_importance(self, enhanced_store):
        enhanced_store.store_with_importance("Default importance")
        items = enhanced_store.list_all()
        assert len(items) == 1

    def test_search_weighted_returns_items(self, enhanced_store):
        enhanced_store.store_with_importance("Python is great", importance=9.0)
        enhanced_store.store_with_importance("Random chatter", importance=2.0)

        results = enhanced_store.search_weighted("programming", top_k=2)
        assert len(results) >= 1
        assert "Python" in results[0]["text"]

    def test_search_weighted_empty(self, enhanced_store):
        results = enhanced_store.search_weighted("anything")
        assert results == []

    def test_get_cumulative_importance(self, enhanced_store):
        enhanced_store.store_with_importance("A", importance=5.0)
        enhanced_store.store_with_importance("B", importance=3.0)
        enhanced_store.store_with_importance("C", importance=2.0)

        total = enhanced_store.get_cumulative_importance()
        assert total == 10.0

    def test_get_cumulative_importance_empty(self, enhanced_store):
        assert enhanced_store.get_cumulative_importance() == 0.0

    def test_base_memory_store_methods_still_work(self, enhanced_store):
        # Existing MemoryStore methods should work on EnhancedMemoryStore
        mem_id = enhanced_store.store("Base method")
        assert mem_id.startswith("mem_")
        items = enhanced_store.list_all()
        assert len(items) == 1
        enhanced_store.delete(mem_id)
        assert enhanced_store.list_all() == []
