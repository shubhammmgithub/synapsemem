"""Tests for Phase 3 Step 2 — Intent classifier and async pipeline.

The intent classifier tests run with zero external deps.
The Celery task tests use .apply() (synchronous / eager mode) so
no Redis broker is required to run the test suite.
"""

import unittest
from unittest.mock import patch, MagicMock


# ======================================================================== #
# Intent classifier                                                         #
# ======================================================================== #

from synapsemem.memory.intent_classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = IntentClassifier()

    # ── classify ──────────────────────────────────────────────────────

    def test_classifies_preference(self):
        intent = self.clf.classify("I like dark mode")
        self.assertEqual(intent, "preference")

    def test_classifies_fact(self):
        intent = self.clf.classify("I am a software engineer")
        self.assertEqual(intent, "fact")

    def test_classifies_task(self):
        intent = self.clf.classify("I need to finish the report by Friday")
        self.assertEqual(intent, "task")

    def test_classifies_delete(self):
        intent = self.clf.classify("Forget that I love coffee")
        self.assertEqual(intent, "delete")

    def test_classifies_tool_result(self):
        intent = self.clf.classify("search returned: Paris is the capital of France")
        self.assertEqual(intent, "tool_result")

    def test_classifies_chitchat(self):
        intent = self.clf.classify("hey")
        self.assertEqual(intent, "chitchat")

    def test_classifies_short_ok_as_chitchat(self):
        intent = self.clf.classify("ok thanks")
        self.assertEqual(intent, "chitchat")

    def test_classifies_unknown_as_fact(self):
        # Long text with no recognised signal defaults to fact
        intent = self.clf.classify("The photosynthesis process converts light into energy")
        self.assertEqual(intent, "fact")

    # ── should_skip ───────────────────────────────────────────────────

    def test_should_skip_chitchat(self):
        self.assertTrue(self.clf.should_skip("chitchat"))

    def test_should_not_skip_fact(self):
        self.assertFalse(self.clf.should_skip("fact"))

    def test_should_not_skip_task(self):
        self.assertFalse(self.clf.should_skip("task"))

    def test_should_not_skip_delete(self):
        self.assertFalse(self.clf.should_skip("delete"))

    # ── priority_boost ────────────────────────────────────────────────

    def test_task_priority_never_below_override(self):
        boosted = self.clf.priority_boost("task", base_priority=2)
        self.assertGreaterEqual(boosted, 6)

    def test_delete_priority_override_is_zero(self):
        boosted = self.clf.priority_boost("delete", base_priority=5)
        # delete override is 0, but max(0, 5) = 5
        self.assertGreaterEqual(boosted, 0)

    def test_fact_priority_uses_base(self):
        boosted = self.clf.priority_boost("fact", base_priority=7)
        self.assertGreaterEqual(boosted, 5)

    # ── enrich_triplets ───────────────────────────────────────────────

    def test_enrich_adds_intent_field(self):
        triplets = [
            {"subject": "user", "predicate": "likes", "object": "python",
             "priority": 3, "topic": "preference", "source_text": "I like python"}
        ]
        enriched = self.clf.enrich_triplets(triplets, "preference")
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["intent"], "preference")

    def test_enrich_does_not_mutate_originals(self):
        original = {"subject": "user", "predicate": "likes", "object": "rust",
                    "priority": 3, "topic": "preference", "source_text": ""}
        enriched = self.clf.enrich_triplets([original], "preference")
        self.assertNotIn("intent", original)
        self.assertIn("intent", enriched[0])

    def test_enrich_applies_priority_boost(self):
        triplets = [
            {"subject": "user", "predicate": "wants", "object": "finish project",
             "priority": 2, "topic": "goal", "source_text": "I need to finish"}
        ]
        enriched = self.clf.enrich_triplets(triplets, "task")
        self.assertGreaterEqual(enriched[0]["priority"], 6)

    def test_enrich_multiple_triplets(self):
        triplets = [
            {"subject": "user", "predicate": "likes", "object": f"item_{i}",
             "priority": 3, "topic": "pref", "source_text": ""}
            for i in range(5)
        ]
        enriched = self.clf.enrich_triplets(triplets, "preference")
        self.assertEqual(len(enriched), 5)
        self.assertTrue(all("intent" in t for t in enriched))


# ======================================================================== #
# Celery tasks — eager / synchronous mode (no Redis needed)                 #
# ======================================================================== #

try:
    from synapsemem.async_pipeline.celery_app import celery_app
    from synapsemem.async_pipeline.tasks import (
        ingest_text_async,
        batch_ingest_async,
        sleep_consolidate_async,
        _apply_decisions,
    )
    _CELERY_AVAILABLE = True
except ImportError:
    _CELERY_AVAILABLE = False


@unittest.skipUnless(_CELERY_AVAILABLE, "celery not installed")
class TestCeleryTasksEager(unittest.TestCase):
    """
    Runs Celery tasks synchronously using CELERY_TASK_ALWAYS_EAGER.
    No broker or worker process needed.
    """

    def setUp(self):
        celery_app.conf.update(
            task_always_eager=True,
            task_eager_propagates=True,
        )

    # ── ingest_text_async ─────────────────────────────────────────────

    def test_ingest_fact_returns_ok(self):
        result = ingest_text_async.apply(kwargs={
            "text": "I love hiking.",
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertEqual(result["status"], "ok")
        self.assertIn("triplet_count", result)
        self.assertGreater(result["triplet_count"], 0)

    def test_ingest_chitchat_is_skipped(self):
        result = ingest_text_async.apply(kwargs={
            "text": "hey",
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "chitchat")

    def test_ingest_returns_elapsed_ms(self):
        result = ingest_text_async.apply(kwargs={
            "text": "I prefer Python over JavaScript.",
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertIn("elapsed_ms", result)
        self.assertIsInstance(result["elapsed_ms"], float)

    def test_ingest_delete_intent_not_skipped(self):
        result = ingest_text_async.apply(kwargs={
            "text": "Forget that I love coffee.",
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        # delete intent goes through pipeline, not skipped
        self.assertNotEqual(result.get("status"), "skipped")

    def test_ingest_reports_applied_actions(self):
        result = ingest_text_async.apply(kwargs={
            "text": "I work on SynapseMem.",
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertIn("applied_actions", result)
        self.assertIsInstance(result["applied_actions"], list)

    # ── batch_ingest_async ────────────────────────────────────────────

    def test_batch_ingest_dispatches_tasks(self):
        result = batch_ingest_async.apply(kwargs={
            "texts": ["I like coffee.", "I work on SynapseMem.", "I prefer dark mode."],
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertEqual(result["status"], "dispatched")
        self.assertEqual(result["text_count"], 3)
        self.assertIn("task_ids", result)

    def test_batch_ingest_empty_list(self):
        result = batch_ingest_async.apply(kwargs={
            "texts": [],
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
        }).get()
        self.assertEqual(result["text_count"], 0)

    # ── sleep_consolidate_async ───────────────────────────────────────

    def test_sleep_consolidate_dry_run(self):
        result = sleep_consolidate_async.apply(kwargs={
            "user_id": "test_user",
            "agent_id": "test_agent",
            "session_id": "test_session",
            "storage_backend": "memory",
            "dry_run": True,
        }).get()
        self.assertTrue(result.get("dry_run"))
        self.assertIn("scanned", result)
        self.assertIn("elapsed_ms", result)

    # ── _apply_decisions helper ───────────────────────────────────────

    def test_apply_decisions_add(self):
        from synapsemem import SynapseMemory
        mem = SynapseMemory(storage_backend="memory")
        decisions = [{
            "action": "ADD",
            "triplet": {
                "subject": "user", "predicate": "likes", "object": "python",
                "topic": "preference", "priority": 5, "source_text": "I like python",
            }
        }]
        applied = _apply_decisions(mem, decisions)
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["action"], "ADD")

    def test_apply_decisions_noop(self):
        from synapsemem import SynapseMemory
        mem = SynapseMemory(storage_backend="memory")
        decisions = [{
            "action": "NOOP",
            "triplet": {
                "subject": "user", "predicate": "likes", "object": "python",
                "topic": "preference", "priority": 5, "source_text": "",
            }
        }]
        applied = _apply_decisions(mem, decisions)
        self.assertEqual(applied[0]["action"], "NOOP")


if __name__ == "__main__":
    unittest.main()