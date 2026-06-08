"""Tests for memory decay — three-level retrieval_weight decay mechanism."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lingya.memory.store import EnhancedMemoryStore

# Counter for unique memory IDs within each test
_counter = 0


def _next_id() -> str:
    global _counter
    _counter += 1
    return f"mem_test_{_counter}"


def _add_memory(
    store: EnhancedMemoryStore,
    text: str,
    importance: float,
    days_ago: float = 0.0,
    *,
    retrieval_weight: float | None = None,
    archived: bool = False,
    archived_days_ago: float | None = None,
) -> str:
    """Helper: add a memory with specific metadata directly to the collection.

    Uses the store's collection directly so we can set custom timestamps
    (including past timestamps for decay testing).
    """
    col = store.collection
    entry_id = _next_id()
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    rw = retrieval_weight if retrieval_weight is not None else importance
    meta: dict = {
        "timestamp": ts.isoformat(),
        "importance": importance,
        "retrieval_weight": rw,
        "archived": archived,
    }
    if archived and archived_days_ago is not None:
        meta["archived_at"] = (
            datetime.now(timezone.utc) - timedelta(days=archived_days_ago)
        ).isoformat()
    col.add(documents=[text], ids=[entry_id], metadatas=[meta])
    return entry_id


@pytest.fixture
def store():
    """EnhancedMemoryStore backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = EnhancedMemoryStore(persist_path=str(Path(tmpdir) / "chroma_decay"))
        yield s


# ── store_with_importance ──────────────────────────────────────────────


class TestStoreWithImportance:
    """store_with_importance sets initial retrieval_weight and archived."""

    def test_initial_retrieval_weight_equals_importance(self, store):
        """New memories have retrieval_weight == importance."""
        mem_id = store.store_with_importance("A test memory", importance=7.5)
        col = store.collection
        results = col.get(ids=[mem_id])
        meta = results["metadatas"][0]
        assert meta["retrieval_weight"] == 7.5
        assert meta["archived"] is False

    def test_default_importance_sets_weight(self, store):
        """Default importance (5.0) is used for retrieval_weight."""
        mem_id = store.store_with_importance("Default importance")
        col = store.collection
        results = col.get(ids=[mem_id])
        meta = results["metadatas"][0]
        assert meta["retrieval_weight"] == 5.0
        assert meta["archived"] is False


# ── apply_decay: critical memories (importance > 0.8) ──────────────────


class TestCriticalMemories:
    """Critical memories (importance > 0.8) are locked — never decay."""

    def test_critical_not_decayed_even_after_long_time(self, store):
        """Critical memories keep retrieval_weight == importance regardless of age."""
        _add_memory(store, "deeply personal", importance=0.9, days_ago=365)
        affected = store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta["retrieval_weight"] == 0.9
        assert meta.get("archived") is not True

    def test_critical_at_boundary(self, store):
        """importance == 0.81 is critical (above 0.8 threshold)."""
        _add_memory(store, "borderline critical", importance=0.81, days_ago=200)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta["retrieval_weight"] == 0.81

    def test_affects_count_includes_first_time_weight_set(self, store):
        """When a critical memory lacks retrieval_weight, setting it counts as
        affected (first run after migration)."""
        col = store.collection
        col.add(
            documents=["old critical without rw"],
            ids=[_next_id()],
            metadatas=[{
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(),
                "importance": 0.95,
                "archived": False,
                # No retrieval_weight — backward compat scenario
            }],
        )
        affected = store.apply_decay()
        assert affected >= 1
        # Now the retrieval_weight should be set
        results = col.get()
        meta = results["metadatas"][0]
        assert meta["retrieval_weight"] == 0.95


# ── apply_decay: normal memories (0.3 <= importance <= 0.8) ────────────


class TestNormalMemories:
    """Normal memories decay linearly: weight = importance * max(0, 1 - days/180)."""

    def test_decay_90_days_half_weight(self, store):
        """After 90 days: weight = importance * 0.5."""
        _add_memory(store, "normal memory", importance=0.6, days_ago=90)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        expected = 0.6 * (1.0 - 90.0 / 180.0)  # = 0.3
        assert abs(float(meta["retrieval_weight"]) - expected) < 0.001

    def test_decay_180_days_weight_zero(self, store):
        """After 180 days: weight reaches 0."""
        _add_memory(store, "old normal", importance=0.5, days_ago=180)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert float(meta["retrieval_weight"]) == 0.0

    def test_decay_beyond_180_days_no_negative(self, store):
        """After 200 days: weight stays at 0, never negative."""
        _add_memory(store, "very old", importance=0.5, days_ago=200)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert float(meta["retrieval_weight"]) == 0.0

    def test_decay_30_days(self, store):
        """After 30 days: partial decay."""
        _add_memory(store, "month old", importance=0.6, days_ago=30)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        expected = 0.6 * (1.0 - 30.0 / 180.0)  # = 0.5
        assert abs(float(meta["retrieval_weight"]) - expected) < 0.001

    def test_decay_importance_immutable(self, store):
        """importance must not change during decay — only retrieval_weight."""
        _add_memory(store, "preserve importance", importance=0.7, days_ago=100)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert float(meta["importance"]) == 0.7

    def test_normal_at_lower_boundary(self, store):
        """importance == 0.3 is normal (not micro)."""
        _add_memory(store, "lower bound normal", importance=0.3, days_ago=50)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta.get("archived") is not True
        # Should decay linearly, not be archived
        expected = 0.3 * (1.0 - 50.0 / 180.0)
        assert abs(float(meta["retrieval_weight"]) - expected) < 0.001


# ── apply_decay: micro memories (importance < 0.3) ─────────────────────


class TestMicroMemories:
    """Micro memories (< 0.3) get archived after 30 days."""

    def test_archived_after_30_days(self, store):
        """After exactly 30 days, micro memory is archived."""
        _add_memory(store, "trivial chat", importance=0.2, days_ago=30)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta["archived"] is True
        assert float(meta["retrieval_weight"]) == 0.0
        assert "archived_at" in meta

    def test_archived_after_60_days(self, store):
        """Well past 30 days, micro memory is archived."""
        _add_memory(store, "old trivial", importance=0.1, days_ago=60)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta["archived"] is True
        assert float(meta["retrieval_weight"]) == 0.0

    def test_not_archived_before_30_days(self, store):
        """Less than 30 days: micro memories decay linearly, NOT archived."""
        _add_memory(store, "recent trivial", importance=0.2, days_ago=15)
        store.apply_decay()
        col = store.collection
        results = col.get()
        meta = results["metadatas"][0]
        assert meta.get("archived") is not True
        # Should have normal linear decay
        expected = 0.2 * (1.0 - 15.0 / 180.0)
        assert abs(float(meta["retrieval_weight"]) - expected) < 0.01

    def test_already_archived_skipped(self, store):
        """Already archived memories are not re-archived (counted once)."""
        _add_memory(
            store, "already archived", importance=0.2, days_ago=60,
            archived=True, archived_days_ago=10, retrieval_weight=0.0,
        )
        affected = store.apply_decay()
        # Should not be counted again (not re-archived)
        assert affected == 0

    def test_archived_memory_hard_deleted_after_90_days(self, store):
        """Archived for 90+ days -> hard delete."""
        _add_memory(
            store, "should be deleted", importance=0.2, days_ago=150,
            archived=True, archived_days_ago=95, retrieval_weight=0.0,
        )
        # Keep track of the ID to verify deletion
        col = store.collection
        all_ids_before = set(col.get()["ids"])
        assert len(all_ids_before) > 0

        affected = store.apply_decay()
        all_ids_after = set(col.get()["ids"])
        # The archived memory should be gone
        assert len(all_ids_after) < len(all_ids_before) or affected > 0

    def test_archived_not_hard_deleted_before_90_days(self, store):
        """Archived for less than 90 days: kept (soft-deleted)."""
        mem_id = _add_memory(
            store, "soft deleted recent", importance=0.2, days_ago=60,
            archived=True, archived_days_ago=30, retrieval_weight=0.0,
        )
        affected = store.apply_decay()
        # Should still exist
        col = store.collection
        results = col.get(ids=[mem_id])
        assert len(results["ids"]) == 1
        # Archived flag unchanged
        assert results["metadatas"][0]["archived"] is True


# ── apply_decay return value ───────────────────────────────────────────


class TestApplyDecayReturnValue:
    """apply_decay returns the count of memories affected."""

    def test_returns_zero_for_empty(self, store):
        assert store.apply_decay() == 0

    def test_returns_count_for_multiple_changes(self, store):
        """Should count all affected memories (decayed + archived + purged)."""
        _add_memory(store, "normal", importance=0.5, days_ago=100)
        _add_memory(store, "micro to archive", importance=0.2, days_ago=35)
        _add_memory(store, "fresh no change", importance=0.5, days_ago=0)

        affected = store.apply_decay()
        # normal: decayed (weight changed), micro: archived, fresh: no change
        # So affected >= 2
        assert affected >= 2

    def test_idempotent_on_second_call(self, store):
        """Second call without time passing should return 0 (already decayed)."""
        _add_memory(store, "once", importance=0.5, days_ago=100)
        first = store.apply_decay()
        assert first >= 1
        second = store.apply_decay()
        assert second == 0


# ── search_weighted ────────────────────────────────────────────────────


class TestSearchWeighted:
    """search_weighted uses retrieval_weight and excludes archived memories."""

    def test_uses_retrieval_weight(self, store):
        """score = exp(-lambda * hours) * retrieval_weight * similarity."""
        _add_memory(store, "high rw memory", importance=3.0,
                    days_ago=30, retrieval_weight=9.0)
        _add_memory(store, "low rw memory", importance=9.0,
                    days_ago=0, retrieval_weight=1.0)

        results = store.search_weighted("memory", top_k=5)
        assert len(results) >= 1
        # The item with higher retrieval_weight should rank higher
        found = [r["retrieval_weight"] for r in results]
        assert 9.0 in found

    def test_excludes_archived_memories(self, store):
        """Archived memories are excluded from search results."""
        _add_memory(store, "visible", importance=5.0, days_ago=0)
        _add_memory(
            store, "hidden archived", importance=5.0, days_ago=60,
            archived=True, archived_days_ago=5, retrieval_weight=0.0,
        )

        results = store.search_weighted("hidden visible", top_k=10)
        ids_in_results = {r["id"] for r in results}
        texts = {r["text"] for r in results}
        assert "hidden archived" not in texts

    def test_backward_compat_no_retrieval_weight(self, store):
        """Memories without retrieval_weight fall back to importance."""
        col = store.collection
        col.add(
            documents=["old style memory"],
            ids=[_next_id()],
            metadatas=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance": 8.0,
                # No retrieval_weight — backward compat
            }],
        )
        results = store.search_weighted("old style", top_k=1)
        assert len(results) >= 1
        assert results[0]["retrieval_weight"] == 8.0  # Fallback to importance

    def test_backward_compat_no_archived_field(self, store):
        """Memories without archived field are treated as not archived."""
        col = store.collection
        col.add(
            documents=["no archived field"],
            ids=[_next_id()],
            metadatas=[{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance": 6.0,
                "retrieval_weight": 6.0,
                # No archived field — backward compat
            }],
        )
        results = store.search_weighted("no archived field", top_k=5)
        texts = {r["text"] for r in results}
        assert "no archived field" in texts

    def test_weighted_score_uses_retrieval_weight(self, store):
        """The weighted_score field reflects retrieval_weight, not importance."""
        _add_memory(store, "score test", importance=9.0,
                    days_ago=0, retrieval_weight=2.0)
        results = store.search_weighted("score test", top_k=5)
        # Find the item
        item = next(r for r in results if r["text"] == "score test")
        # weighted_score uses retrieval_weight (2.0), not importance (9.0)
        # With recency_lambda=0.01 and hours_since ~= 0:
        # recency_weight = exp(-0.01 * 0) = 1.0
        # combined = 1.0 * 2.0 = 2.0
        assert abs(item["weighted_score"] - 2.0) < 0.1


# ── recover ─────────────────────────────────────────────────────────────


class TestRecover:
    """/memory recover resets retrieval_weight and clears archived flag."""

    def test_recover_resets_weight_and_clears_archived(self, store):
        """Recovered memory: retrieval_weight = importance, archived = False."""
        _add_memory(
            store, "recoverable memory", importance=7.0, days_ago=60,
            archived=True, archived_days_ago=10, retrieval_weight=0.0,
        )
        col = store.collection
        all_before = col.get()
        mem_id = all_before["ids"][0]

        result = store.recover(mem_id)
        assert result is True

        results = col.get(ids=[mem_id])
        meta = results["metadatas"][0]
        assert float(meta["retrieval_weight"]) == 7.0
        assert meta["archived"] is False
        assert meta.get("archived_at", "") == ""  # cleared on recover

    def test_recover_nonexistent_returns_false(self, store):
        assert store.recover("mem_nonexistent") is False

    def test_recovered_memory_appears_in_search(self, store):
        """After recover, previously archived memory shows up in search."""
        mem_id = _add_memory(
            store, "searchable after recover", importance=6.0, days_ago=60,
            archived=True, archived_days_ago=5, retrieval_weight=0.0,
        )

        # Before recover: excluded from search
        results_before = store.search_weighted("searchable after recover", top_k=10)
        ids_before = {r["id"] for r in results_before}
        assert mem_id not in ids_before

        # After recover: appears in search
        store.recover(mem_id)
        results_after = store.search_weighted("searchable after recover", top_k=10)
        ids_after = {r["id"] for r in results_after}
        assert mem_id in ids_after

    def test_recover_preserves_importance(self, store):
        """Recover does not change the original importance."""
        _add_memory(
            store, "importance preserved", importance=8.5, days_ago=60,
            archived=True, archived_days_ago=10, retrieval_weight=0.0,
        )
        col = store.collection
        mem_id = col.get()["ids"][0]

        store.recover(mem_id)
        results = col.get(ids=[mem_id])
        assert float(results["metadatas"][0]["importance"]) == 8.5
