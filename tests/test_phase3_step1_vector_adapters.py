"""Tests for Phase 3 Step 1 — Vector DB storage adapters.

These tests use in-memory / local backends so no running server is needed:
  - Qdrant: uses qdrant_client.QdrantClient(":memory:")
  - Chroma: uses EphemeralClient (in-process, no persistence)

If the optional deps are not installed the tests are skipped cleanly.
"""

import unittest
import time


# ── Qdrant ────────────────────────────────────────────────────────────────

try:
    from qdrant_client import QdrantClient
    from synapsemem.memory.qdrant_storage import QdrantMemoryStorage
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


def _make_qdrant_storage() -> "QdrantMemoryStorage":
    """Creates an in-memory Qdrant storage instance (no Docker needed)."""
    storage = QdrantMemoryStorage.__new__(QdrantMemoryStorage)
    storage.user_id = "test_user"
    storage.agent_id = "test_agent"
    storage.session_id = "test_session"
    storage.embedding_dim = 384
    storage.client = QdrantClient(":memory:")
    storage.collection_name = storage._make_collection_name(
        "test_user", "test_agent", "test_session"
    )
    storage._ensure_collection()
    return storage


@unittest.skipUnless(_QDRANT_AVAILABLE, "qdrant-client not installed")
class TestQdrantStorageAdapter(unittest.TestCase):

    def setUp(self):
        self.storage = _make_qdrant_storage()

    def _sample_triplet(self, subject="user", predicate="likes", obj="hiking"):
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "topic": "preference",
            "priority": 5,
            "source_text": f"{subject} {predicate} {obj}",
        }

    # ── add / all ─────────────────────────────────────────────────────

    def test_add_and_all(self):
        self.storage.add_triplets([self._sample_triplet()])
        records = self.storage.all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["subject"], "user")
        self.assertEqual(records[0]["predicate"], "likes")
        self.assertEqual(records[0]["object"], "hiking")

    def test_all_returns_only_active(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([self._sample_triplet("user", "hates", "noise")])
        # delete one
        self.storage.delete_fact(subject="user", predicate="hates", obj="noise")
        records = self.storage.all()
        self.assertEqual(len(records), 1)

    def test_all_records_includes_pruned(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.delete_fact(subject="user", predicate="likes", obj="hiking")
        all_records = self.storage.all_records()
        active = self.storage.all()
        self.assertGreater(len(all_records), len(active))

    # ── find_exact ────────────────────────────────────────────────────

    def test_find_exact_hit(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.find_exact("user", "likes", "hiking")
        self.assertIsNotNone(record)
        self.assertEqual(record["object"], "hiking")

    def test_find_exact_miss(self):
        result = self.storage.find_exact("user", "likes", "nonexistent")
        self.assertIsNone(result)

    # ── find_by_subject_predicate ─────────────────────────────────────

    def test_find_by_subject_predicate(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([self._sample_triplet("user", "likes", "coffee")])
        results = self.storage.find_by_subject_predicate("user", "likes")
        self.assertEqual(len(results), 2)

    # ── update_fact ───────────────────────────────────────────────────

    def test_update_fact(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.find_exact("user", "likes", "hiking")
        updated = self.storage.update_fact(
            record["id"],
            {"subject": "user", "predicate": "likes", "object": "cycling",
             "topic": "preference", "priority": 6, "source_text": "user likes cycling"},
        )
        self.assertTrue(updated)
        self.assertIsNone(self.storage.find_exact("user", "likes", "hiking"))
        self.assertIsNotNone(self.storage.find_exact("user", "likes", "cycling"))

    # ── reinforce ─────────────────────────────────────────────────────

    def test_reinforce_increments_count(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.all()[0]
        self.assertEqual(record["reinforcement_count"], 0)
        self.storage.reinforce(record["id"])
        updated = self.storage.find_exact("user", "likes", "hiking")
        self.assertEqual(updated["reinforcement_count"], 1)

    # ── delete_topic ──────────────────────────────────────────────────

    def test_delete_topic(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([
            {"subject": "user", "predicate": "works_on", "object": "project",
             "topic": "work", "priority": 5, "source_text": "user works_on project"}
        ])
        deleted = self.storage.delete_topic("preference")
        self.assertEqual(deleted, 1)
        remaining = self.storage.all()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["topic"], "work")

    # ── delete_fact ───────────────────────────────────────────────────

    def test_delete_fact(self):
        self.storage.add_triplets([self._sample_triplet()])
        count = self.storage.delete_fact(subject="user", predicate="likes", obj="hiking")
        self.assertEqual(count, 1)
        self.assertEqual(len(self.storage.all()), 0)

    # ── merge_duplicates ──────────────────────────────────────────────

    def test_merge_duplicates(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([self._sample_triplet()])
        records = self.storage.all()
        self.assertEqual(len(records), 2)
        merge_actions = [{"record_id": records[1]["id"], "survivor_id": records[0]["id"]}]
        merged = self.storage.merge_duplicates(merge_actions)
        self.assertEqual(merged, 1)
        active = self.storage.all()
        self.assertEqual(len(active), 1)

    # ── prune_memories ────────────────────────────────────────────────

    def test_prune_memories(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.all()[0]
        pruned = self.storage.prune_memories([{"record_id": record["id"]}])
        self.assertEqual(pruned, 1)
        self.assertEqual(len(self.storage.all()), 0)

    # ── semantic memory ───────────────────────────────────────────────

    def test_promote_to_semantic(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([self._sample_triplet()])
        records = self.storage.all()
        result = self.storage.promote_to_semantic(records)
        self.assertIsNotNone(result)
        semantic = self.storage.find_semantic_memory("user", "likes", "hiking")
        self.assertIsNotNone(semantic)
        self.assertEqual(semantic["memory_type"], "semantic")

    # ── reset ─────────────────────────────────────────────────────────

    def test_reset_clears_all(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.reset()
        self.assertEqual(len(self.storage.all()), 0)


# ── Chroma ────────────────────────────────────────────────────────────────

try:
    import chromadb
    from synapsemem.memory.chroma_storage import ChromaMemoryStorage
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


def _make_chroma_storage() -> "ChromaMemoryStorage":
    """Creates an ephemeral (in-process) Chroma storage instance."""
    return ChromaMemoryStorage(
        persist_directory=None,   # EphemeralClient
        user_id="test_user",
        agent_id="test_agent",
        session_id="test_session",
    )


@unittest.skipUnless(_CHROMA_AVAILABLE, "chromadb not installed")
class TestChromaStorageAdapter(unittest.TestCase):

    def setUp(self):
        self.storage = _make_chroma_storage()

    def _sample_triplet(self, subject="user", predicate="likes", obj="hiking"):
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "topic": "preference",
            "priority": 5,
            "source_text": f"{subject} {predicate} {obj}",
        }

    def test_add_and_all(self):
        self.storage.add_triplets([self._sample_triplet()])
        records = self.storage.all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["subject"], "user")

    def test_find_exact_hit(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.find_exact("user", "likes", "hiking")
        self.assertIsNotNone(record)

    def test_find_exact_miss(self):
        result = self.storage.find_exact("user", "likes", "nonexistent")
        self.assertIsNone(result)

    def test_update_fact(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.find_exact("user", "likes", "hiking")
        updated = self.storage.update_fact(
            record["id"],
            {"subject": "user", "predicate": "likes", "object": "swimming",
             "topic": "preference", "priority": 6, "source_text": "user likes swimming"},
        )
        self.assertTrue(updated)
        self.assertIsNone(self.storage.find_exact("user", "likes", "hiking"))
        self.assertIsNotNone(self.storage.find_exact("user", "likes", "swimming"))

    def test_reinforce(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.all()[0]
        self.storage.reinforce(record["id"])
        updated = self.storage.find_exact("user", "likes", "hiking")
        self.assertEqual(updated["reinforcement_count"], 1)

    def test_delete_fact(self):
        self.storage.add_triplets([self._sample_triplet()])
        count = self.storage.delete_fact(subject="user", predicate="likes", obj="hiking")
        self.assertEqual(count, 1)
        self.assertEqual(len(self.storage.all()), 0)

    def test_promote_to_semantic(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.add_triplets([self._sample_triplet()])
        records = self.storage.all()
        result = self.storage.promote_to_semantic(records)
        self.assertIsNotNone(result)
        self.assertEqual(result["memory_type"], "semantic")

    def test_reset_clears_all(self):
        self.storage.add_triplets([self._sample_triplet()])
        self.storage.reset()
        self.assertEqual(len(self.storage.all()), 0)

    def test_consolidated_from_roundtrip(self):
        """Ensure consolidated_from JSON encoding survives a read/write cycle."""
        triplet = self._sample_triplet()
        triplet["consolidated_from"] = ["id-1", "id-2"]
        self.storage.add_triplets([triplet])
        record = self.storage.all()[0]
        self.assertIsInstance(record["consolidated_from"], list)

    def test_last_accessed_at_none_by_default(self):
        self.storage.add_triplets([self._sample_triplet()])
        record = self.storage.all()[0]
        self.assertIsNone(record["last_accessed_at"])


# ── Interface parity check ────────────────────────────────────────────────

class TestBaseStorageInterface(unittest.TestCase):
    """
    Ensures all storage backends expose the same method interface.
    This is a static check — no network or disk I/O.
    """

    REQUIRED_METHODS = [
        "add_triplets", "all", "all_records",
        "find_exact", "find_by_subject_predicate",
        "update_fact", "reinforce", "update_last_accessed",
        "delete_topic", "delete_fact",
        "merge_duplicates", "prune_memories",
        "find_semantic_memory", "promote_to_semantic",
        "reset",
    ]

    def _check_interface(self, cls):
        for method in self.REQUIRED_METHODS:
            self.assertTrue(
                hasattr(cls, method),
                f"{cls.__name__} is missing method: {method}",
            )

    def test_sqlite_interface(self):
        from synapsemem.memory.sqlite_storage import SQLiteMemoryStorage
        self._check_interface(SQLiteMemoryStorage)

    def test_memory_interface(self):
        from synapsemem.memory.storage import MemoryStorage
        self._check_interface(MemoryStorage)

    @unittest.skipUnless(_QDRANT_AVAILABLE, "qdrant-client not installed")
    def test_qdrant_interface(self):
        self._check_interface(QdrantMemoryStorage)

    @unittest.skipUnless(_CHROMA_AVAILABLE, "chromadb not installed")
    def test_chroma_interface(self):
        self._check_interface(ChromaMemoryStorage)


if __name__ == "__main__":
    unittest.main()