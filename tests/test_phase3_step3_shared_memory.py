"""Tests for Phase 3 Step 3 — Multi-agent shared memory.

Windows-safe: uses SQLite :memory: instead of temp files.
SQLite in-memory DB avoids WinError 32 file-lock issues entirely.
"""

import os
import unittest

from synapsemem.memory.shared_memory import SharedMemoryStore


def _make_store(
    workspace_id: str = "test_workspace",
    conflict_strategy: str = "last_write_wins",
    db_path: str = ":memory:",
) -> SharedMemoryStore:
    return SharedMemoryStore(
        workspace_id=workspace_id,
        db_path=db_path,
        conflict_strategy=conflict_strategy,
    )


def _triplet(subject="user", predicate="likes", obj="python", priority=5):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "topic": "preference",
        "priority": priority,
        "source_text": f"{subject} {predicate} {obj}",
    }


class TestSharedMemoryStore(unittest.TestCase):
    """Uses SQLite :memory: — no file I/O, no Windows lock issues."""

    def setUp(self):
        self.store = _make_store(db_path=":memory:")

    # ── write / read ──────────────────────────────────────────────────

    def test_write_and_read_fact(self):
        result = self.store.write_fact(_triplet(), agent_id="agent_a")
        self.assertEqual(result["action"], "ADD")
        facts = self.store.read_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["subject"], "user")
        self.assertEqual(facts[0]["object"], "python")

    def test_write_multiple_facts(self):
        self.store.write_fact(_triplet("user", "likes", "python"), agent_id="agent_a")
        self.store.write_fact(_triplet("user", "uses", "vim"), agent_id="agent_b")
        self.store.write_fact(_triplet("user", "works_on", "synapsemem"), agent_id="agent_c")
        facts = self.store.read_facts()
        self.assertEqual(len(facts), 3)

    def test_read_facts_empty_workspace(self):
        facts = self.store.read_facts()
        self.assertEqual(facts, [])

    def test_read_facts_topic_filter(self):
        self.store.write_fact(
            {**_triplet(), "topic": "preference"}, agent_id="agent_a"
        )
        self.store.write_fact(
            {**_triplet("user", "works_on", "synapsemem"), "topic": "work"},
            agent_id="agent_b",
        )
        pref_facts = self.store.read_facts(topic="preference")
        self.assertEqual(len(pref_facts), 1)
        self.assertEqual(pref_facts[0]["object"], "python")

    # ── facts_by_agent ────────────────────────────────────────────────

    def test_facts_by_agent(self):
        # Use different predicates so each write is a clean ADD (no conflict)
        self.store.write_fact(_triplet("user", "likes", "python"), agent_id="agent_a")
        self.store.write_fact(_triplet("user", "loves", "rust"), agent_id="agent_b")
        self.store.write_fact(_triplet("user", "uses", "vim"), agent_id="agent_a")

        a_facts = self.store.facts_by_agent("agent_a")
        b_facts = self.store.facts_by_agent("agent_b")

        self.assertEqual(len(a_facts), 2)
        self.assertEqual(len(b_facts), 1)

    def test_facts_by_unknown_agent_empty(self):
        facts = self.store.facts_by_agent("nobody")
        self.assertEqual(facts, [])

    # ── workspace_stats ───────────────────────────────────────────────

    def test_workspace_stats_empty(self):
        stats = self.store.workspace_stats()
        self.assertEqual(stats["total_records"], 0)
        self.assertEqual(stats["active_records"], 0)
        self.assertEqual(stats["contributing_agents"], 0)
        self.assertEqual(stats["workspace_id"], "test_workspace")

    def test_workspace_stats_populated(self):
        # Different predicates → both write as ADD, giving 2 active records
        self.store.write_fact(_triplet("user", "likes", "python"), agent_id="agent_a")
        self.store.write_fact(_triplet("user", "loves", "rust"), agent_id="agent_b")
        stats = self.store.workspace_stats()
        self.assertEqual(stats["active_records"], 2)
        self.assertEqual(stats["contributing_agents"], 2)

    # ── delete_fact ───────────────────────────────────────────────────

    def test_delete_fact(self):
        self.store.write_fact(_triplet(), agent_id="agent_a")
        deleted = self.store.delete_fact("user", "likes", "python", agent_id="agent_a")
        self.assertTrue(deleted)
        self.assertEqual(len(self.store.read_facts()), 0)

    def test_delete_nonexistent_fact_returns_false(self):
        deleted = self.store.delete_fact("user", "likes", "nonexistent", agent_id="agent_a")
        self.assertFalse(deleted)

    # ── workspace isolation ───────────────────────────────────────────

    def test_workspaces_are_isolated(self):
        # Two independent in-memory DBs — truly isolated
        store_alpha = SharedMemoryStore(workspace_id="alpha", db_path=":memory:")
        store_beta  = SharedMemoryStore(workspace_id="beta",  db_path=":memory:")

        store_alpha.write_fact(_triplet("user", "likes", "python"), agent_id="agent_a")
        store_beta.write_fact(_triplet("user", "likes", "java"), agent_id="agent_b")

        self.assertEqual(len(store_alpha.read_facts()), 1)
        self.assertEqual(len(store_beta.read_facts()), 1)
        self.assertEqual(store_alpha.read_facts()[0]["object"], "python")
        self.assertEqual(store_beta.read_facts()[0]["object"], "java")


class TestConflictResolutionLastWriteWins(unittest.TestCase):

    def setUp(self):
        self.store = _make_store(conflict_strategy="last_write_wins")

    def test_second_write_replaces_first(self):
        self.store.write_fact(
            _triplet("user", "likes", "python"), agent_id="agent_a"
        )
        result = self.store.write_fact(
            _triplet("user", "likes", "rust"), agent_id="agent_b"
        )
        self.assertEqual(result["action"], "UPDATE")
        facts = self.store.read_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["object"], "rust")

    def test_same_predicate_different_object_updates(self):
        self.store.write_fact(_triplet("user", "uses", "vim"), agent_id="agent_a")
        self.store.write_fact(_triplet("user", "uses", "neovim"), agent_id="agent_b")
        facts = self.store.read_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["object"], "neovim")


class TestConflictResolutionNoOverwrite(unittest.TestCase):

    def setUp(self):
        self.store = _make_store(conflict_strategy="no_overwrite")

    def test_first_write_wins(self):
        self.store.write_fact(
            _triplet("user", "likes", "python"), agent_id="agent_a"
        )
        result = self.store.write_fact(
            _triplet("user", "likes", "rust"), agent_id="agent_b"
        )
        self.assertEqual(result["action"], "NOOP")
        self.assertEqual(result["reason"], "no_overwrite_policy")
        facts = self.store.read_facts()
        self.assertEqual(facts[0]["object"], "python")

    def test_new_predicate_is_allowed(self):
        self.store.write_fact(_triplet("user", "likes", "python"), agent_id="agent_a")
        result = self.store.write_fact(
            _triplet("user", "uses", "vim"), agent_id="agent_b"
        )
        self.assertEqual(result["action"], "ADD")
        self.assertEqual(len(self.store.read_facts()), 2)


class TestConflictResolutionAnchorWeighted(unittest.TestCase):

    def setUp(self):
        self.store = _make_store(conflict_strategy="anchor_weighted")

    def test_higher_priority_incoming_wins(self):
        self.store.write_fact(
            _triplet("user", "likes", "python", priority=3), agent_id="agent_a"
        )
        result = self.store.write_fact(
            _triplet("user", "likes", "rust", priority=8), agent_id="agent_b"
        )
        self.assertEqual(result["action"], "UPDATE")
        self.assertEqual(result["reason"], "higher_priority_wins")
        facts = self.store.read_facts()
        self.assertEqual(facts[0]["object"], "rust")

    def test_lower_priority_incoming_loses(self):
        self.store.write_fact(
            _triplet("user", "likes", "python", priority=8), agent_id="agent_a"
        )
        result = self.store.write_fact(
            _triplet("user", "likes", "rust", priority=2), agent_id="agent_b"
        )
        self.assertEqual(result["action"], "NOOP")
        self.assertEqual(result["reason"], "existing_anchor_wins")
        facts = self.store.read_facts()
        self.assertEqual(facts[0]["object"], "python")

    def test_equal_priority_existing_wins(self):
        self.store.write_fact(
            _triplet("user", "likes", "python", priority=5), agent_id="agent_a"
        )
        result = self.store.write_fact(
            _triplet("user", "likes", "rust", priority=5), agent_id="agent_b"
        )
        # equal priority: existing wins (not strictly greater)
        self.assertEqual(result["action"], "NOOP")


if __name__ == "__main__":
    unittest.main()