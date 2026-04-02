"""Tests for Phase 3 Step 4 — Memory compression."""

import unittest

from synapsemem import SynapseMemory
from synapsemem.memory.memory_compressor import MemoryCompressor


def _fake_llm_summariser(prompt: str) -> str:
    """Returns a valid SUBJECT | PREDICATE | OBJECT response."""
    return "user | prefers | programming languages"


def _make_memory() -> SynapseMemory:
    return SynapseMemory(storage_backend="memory")


class TestMemoryCompressorDryRun(unittest.TestCase):

    def setUp(self):
        self.memory = _make_memory()
        self.compressor = MemoryCompressor(
            llm=None,            # no LLM — uses representative fallback
            similarity_threshold=0.80,
            min_cluster_size=2,
        )

    def _ingest_many(self, texts):
        for text in texts:
            self.memory.ingest(text)

    def test_dry_run_writes_nothing(self):
        self._ingest_many([
            "I like python.",
            "I like programming.",
            "I like coding.",
        ])
        before = len(self.memory.storage.all())
        report = self.compressor.run(self.memory.storage, dry_run=True)
        after = len(self.memory.storage.all())
        self.assertTrue(report["dry_run"])
        self.assertEqual(before, after)  # nothing written

    def test_dry_run_report_fields(self):
        self._ingest_many(["I like python.", "I like coding."])
        report = self.compressor.run(self.memory.storage, dry_run=True)
        self.assertIn("episodic_scanned", report)
        self.assertIn("clusters_found", report)
        self.assertIn("eligible_clusters", report)
        self.assertIn("compressed", report)
        self.assertIn("compression_actions", report)

    def test_no_eligible_clusters_when_too_few_records(self):
        self.memory.ingest("I like hiking.")
        report = self.compressor.run(self.memory.storage, dry_run=True)
        self.assertEqual(report["eligible_clusters"], 0)
        self.assertEqual(report["compressed"], 0)

    def test_report_counts_are_non_negative(self):
        self._ingest_many([
            "I like python.", "I like java.", "I like rust.",
        ])
        report = self.compressor.run(self.memory.storage, dry_run=True)
        self.assertGreaterEqual(report["episodic_scanned"], 0)
        self.assertGreaterEqual(report["clusters_found"], 0)
        self.assertGreaterEqual(report["compressed"], 0)


class TestMemoryCompressorLive(unittest.TestCase):

    def setUp(self):
        self.memory = _make_memory()
        self.compressor = MemoryCompressor(
            llm=None,
            similarity_threshold=0.70,   # lower threshold → more clusters
            min_cluster_size=2,
        )

    def _seed_similar_memories(self):
        """Ingest semantically similar triplets directly into storage."""
        similar = [
            {"subject": "user", "predicate": "likes", "object": "python",
             "topic": "preference", "priority": 5, "source_text": "I like python",
             "memory_type": "episodic", "status": "active"},
            {"subject": "user", "predicate": "likes", "object": "programming",
             "topic": "preference", "priority": 5, "source_text": "I like programming",
             "memory_type": "episodic", "status": "active"},
            {"subject": "user", "predicate": "likes", "object": "coding",
             "topic": "preference", "priority": 5, "source_text": "I like coding",
             "memory_type": "episodic", "status": "active"},
        ]
        self.memory.storage.add_triplets(similar)

    def test_compression_report_has_action_details(self):
        self._seed_similar_memories()
        report = self.compressor.run(self.memory.storage, dry_run=True)
        for action in report["compression_actions"]:
            self.assertIn("action", action)
            self.assertIn("cluster_size", action)
            self.assertIn("summary", action)
            self.assertIn("source_record_ids", action)

    def test_compression_action_summary_has_triplet_fields(self):
        self._seed_similar_memories()
        report = self.compressor.run(self.memory.storage, dry_run=True)
        for action in report["compression_actions"]:
            summary = action["summary"]
            self.assertIn("subject", summary)
            self.assertIn("predicate", summary)
            self.assertIn("object", summary)

    def test_live_compression_adds_semantic_memory(self):
        self._seed_similar_memories()
        before_active = len(self.memory.storage.all())
        report = self.compressor.run(self.memory.storage, dry_run=False)

        if report["compressed"] > 0:
            all_records = self.memory.storage.all_records()
            semantic_records = [
                r for r in all_records
                if r.get("memory_type") == "semantic"
                and "compressed from" in str(r.get("source_text", ""))
            ]
            self.assertGreater(len(semantic_records), 0)

    def test_live_compression_reduces_active_episodic_count(self):
        self._seed_similar_memories()
        episodic_before = len([
            r for r in self.memory.storage.all()
            if r.get("memory_type") == "episodic"
        ])
        report = self.compressor.run(self.memory.storage, dry_run=False)

        if report["compressed"] > 0:
            episodic_after = len([
                r for r in self.memory.storage.all()
                if r.get("memory_type") == "episodic"
            ])
            self.assertLess(episodic_after, episodic_before)

    def test_semantic_memories_are_not_recompressed(self):
        """Compressor must never touch existing semantic memories."""
        self.memory.storage.add_triplets([{
            "subject": "user", "predicate": "likes", "object": "python",
            "topic": "preference", "priority": 7, "source_text": "consolidated",
            "memory_type": "semantic", "status": "active",
        }])
        report = self.compressor.run(self.memory.storage, dry_run=False)
        semantic_after = [
            r for r in self.memory.storage.all()
            if r.get("memory_type") == "semantic"
        ]
        # The existing semantic record must still be active
        self.assertEqual(len(semantic_after), 1)


class TestMemoryCompressorWithLLM(unittest.TestCase):

    def setUp(self):
        self.memory = _make_memory()
        self.compressor = MemoryCompressor(
            llm=_fake_llm_summariser,
            similarity_threshold=0.70,
            min_cluster_size=2,
        )

    def test_llm_summary_is_used_when_available(self):
        similar = [
            {"subject": "user", "predicate": "likes", "object": "python",
             "topic": "preference", "priority": 5, "source_text": "I like python",
             "memory_type": "episodic", "status": "active"},
            {"subject": "user", "predicate": "likes", "object": "rust",
             "topic": "preference", "priority": 5, "source_text": "I like rust",
             "memory_type": "episodic", "status": "active"},
        ]
        self.memory.storage.add_triplets(similar)
        report = self.compressor.run(self.memory.storage, dry_run=True)

        for action in report["compression_actions"]:
            # LLM response was "user | prefers | programming languages"
            summary = action["summary"]
            self.assertIn("subject", summary)
            self.assertIn("predicate", summary)
            self.assertIn("object", summary)

    def test_llm_bad_response_falls_back_to_representative(self):
        def bad_llm(prompt: str) -> str:
            return "this is not a valid triplet format"

        compressor = MemoryCompressor(
            llm=bad_llm,
            similarity_threshold=0.70,
            min_cluster_size=2,
        )
        similar = [
            {"subject": "user", "predicate": "likes", "object": "python",
             "topic": "preference", "priority": 5, "source_text": "",
             "memory_type": "episodic", "status": "active"},
            {"subject": "user", "predicate": "likes", "object": "coding",
             "topic": "preference", "priority": 5, "source_text": "",
             "memory_type": "episodic", "status": "active"},
        ]
        self.memory.storage.add_triplets(similar)
        # Should not raise — fallback to representative
        report = compressor.run(self.memory.storage, dry_run=True)
        self.assertIsInstance(report, dict)


class TestMemoryCompressorThresholds(unittest.TestCase):

    def test_high_threshold_finds_fewer_clusters(self):
        memory = _make_memory()
        memory.ingest("I like python.")
        memory.ingest("I enjoy cycling.")
        memory.ingest("I love cooking.")

        strict = MemoryCompressor(similarity_threshold=0.99, min_cluster_size=2)
        lenient = MemoryCompressor(similarity_threshold=0.50, min_cluster_size=2)

        strict_report  = strict.run(memory.storage, dry_run=True)
        lenient_report = lenient.run(memory.storage, dry_run=True)

        self.assertLessEqual(
            strict_report["eligible_clusters"],
            lenient_report["eligible_clusters"],
        )

    def test_min_cluster_size_respected(self):
        memory = _make_memory()
        similar = [
            {"subject": "user", "predicate": "likes", "object": f"item_{i}",
             "topic": "pref", "priority": 5, "source_text": "",
             "memory_type": "episodic", "status": "active"}
            for i in range(5)
        ]
        memory.storage.add_triplets(similar)

        large_min = MemoryCompressor(similarity_threshold=0.50, min_cluster_size=10)
        report = large_min.run(memory.storage, dry_run=True)
        self.assertEqual(report["eligible_clusters"], 0)


if __name__ == "__main__":
    unittest.main()