"""Tests for Phase 3 Step 5 — Framework integrations (LangChain + CrewAI).

Both test classes use lightweight fakes so no real LLM or framework
install is required. The @skipUnless guards handle missing optional deps.
"""

import unittest

from synapsemem import SynapseMemory


def _make_memory() -> SynapseMemory:
    return SynapseMemory(storage_backend="memory")


def _fake_llm(prompt: str) -> str:
    return f"FAKE_RESPONSE | {prompt[:40]}"


# ======================================================================== #
# LangChain integration                                                     #
# ======================================================================== #

try:
    from synapsemem.integrations.langchain_integration import (
        SynapseMemLangChainMemory,
        SynapseMemTool,
    )
    _LANGCHAIN_INTEGRATION_IMPORTABLE = True
except ImportError:
    _LANGCHAIN_INTEGRATION_IMPORTABLE = False

# The integration file imports langchain at module level only inside
# _require_langchain(). We can still test the logic if langchain
# itself is not installed — we just skip the actual interface tests.
try:
    import langchain  # noqa: F401
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


@unittest.skipUnless(
    _LANGCHAIN_INTEGRATION_IMPORTABLE and _LANGCHAIN_AVAILABLE,
    "langchain not installed",
)
class TestSynapseMemLangChainMemory(unittest.TestCase):

    def setUp(self):
        self.synapse = _make_memory()
        self.lc_memory = SynapseMemLangChainMemory(
            synapse=self.synapse,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            top_k=5,
        )

    def test_memory_variables_property(self):
        self.assertEqual(self.lc_memory.memory_variables, ["chat_history"])

    def test_load_empty_memory_returns_empty_string(self):
        result = self.lc_memory.load_memory_variables({"input": "what do I like?"})
        self.assertEqual(result["chat_history"], "")

    def test_save_context_ingests_human_input(self):
        self.lc_memory.save_context(
            {"input": "I love hiking."},
            {"output": "Great to know!"},
        )
        records = self.synapse.storage.all()
        self.assertGreater(len(records), 0)

    def test_load_after_save_returns_relevant_memories(self):
        self.lc_memory.save_context(
            {"input": "I love hiking."},
            {"output": "Sounds fun."},
        )
        result = self.lc_memory.load_memory_variables({"input": "what do I love?"})
        self.assertIn("hiking", result["chat_history"].lower())

    def test_load_returns_text_by_default(self):
        self.lc_memory.save_context(
            {"input": "I work on SynapseMem."},
            {"output": "Cool project."},
        )
        result = self.lc_memory.load_memory_variables({"input": "what am I working on?"})
        self.assertIsInstance(result["chat_history"], str)

    def test_load_returns_list_when_configured(self):
        lc_memory_list = SynapseMemLangChainMemory(
            synapse=self.synapse,
            return_as_text=False,
        )
        self.synapse.ingest("I prefer dark mode.")
        result = lc_memory_list.load_memory_variables({"input": "preferences"})
        self.assertIsInstance(result["chat_history"], list)

    def test_clear_resets_storage(self):
        self.synapse.ingest("I like coffee.")
        self.lc_memory.clear()
        self.assertEqual(len(self.synapse.storage.all()), 0)

    def test_empty_input_returns_empty_string(self):
        result = self.lc_memory.load_memory_variables({"input": ""})
        self.assertEqual(result["chat_history"], "")

    def test_custom_memory_key(self):
        lc = SynapseMemLangChainMemory(
            synapse=self.synapse,
            memory_key="history",
        )
        self.assertEqual(lc.memory_variables, ["history"])
        result = lc.load_memory_variables({"input": "test"})
        self.assertIn("history", result)


@unittest.skipUnless(
    _LANGCHAIN_INTEGRATION_IMPORTABLE and _LANGCHAIN_AVAILABLE,
    "langchain not installed",
)
class TestSynapseMemTool(unittest.TestCase):

    def setUp(self):
        self.synapse = _make_memory()
        self.tool = SynapseMemTool(synapse=self.synapse, top_k=5)

    def test_store_command_ingests_memory(self):
        response = self.tool.run("store: I prefer Python over JavaScript")
        self.assertIn("Stored", response)
        records = self.synapse.storage.all()
        self.assertGreater(len(records), 0)

    def test_retrieve_command_with_no_memories(self):
        response = self.tool.run("retrieve: what languages do I like")
        self.assertIn("No relevant memories", response)

    def test_retrieve_command_returns_results(self):
        self.synapse.ingest("I like Python.")
        response = self.tool.run("retrieve: programming language preferences")
        self.assertIn("python", response.lower())

    def test_bare_query_treated_as_retrieve(self):
        self.synapse.ingest("I work on SynapseMem.")
        response = self.tool.run("current projects")
        self.assertIsInstance(response, str)

    def test_store_empty_text_returns_error(self):
        response = self.tool.run("store: ")
        self.assertIn("Error", response)

    def test_store_returns_count(self):
        response = self.tool.run("store: I love hiking and outdoor activities.")
        self.assertIn("Stored", response)


# ======================================================================== #
# CrewAI integration                                                        #
# ======================================================================== #

try:
    from synapsemem.integrations.crewai_integration import (
        SynapseMemCrewAITool,
        SynapseMemCrewStorage,
    )
    _CREWAI_INTEGRATION_IMPORTABLE = True
except ImportError:
    _CREWAI_INTEGRATION_IMPORTABLE = False

try:
    import crewai  # noqa: F401
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


@unittest.skipUnless(
    _CREWAI_INTEGRATION_IMPORTABLE and _CREWAI_AVAILABLE,
    "crewai not installed",
)
class TestSynapseMemCrewAITool(unittest.TestCase):

    def setUp(self):
        self.synapse = _make_memory()
        self.tool = SynapseMemCrewAITool(synapse=self.synapse, top_k=5)

    def test_store_command(self):
        response = self.tool.run("store: I prefer remote work")
        self.assertIn("Stored", response)

    def test_retrieve_command_empty(self):
        response = self.tool.run("retrieve: work preferences")
        self.assertIn("No relevant memories", response)

    def test_retrieve_command_with_data(self):
        self.synapse.ingest("I prefer remote work.")
        response = self.tool.run("retrieve: work style")
        self.assertIn("remote", response.lower())

    def test_facts_about_command(self):
        self.synapse.ingest("I love hiking.")
        response = self.tool.run("facts about: user")
        self.assertIsInstance(response, str)

    def test_facts_about_unknown_entity(self):
        response = self.tool.run("facts about: nobody_xyz")
        self.assertIn("No known facts", response)

    def test_bare_query_as_retrieve(self):
        self.synapse.ingest("I work on SynapseMem.")
        response = self.tool.run("what am I building")
        self.assertIsInstance(response, str)

    def test_tool_name_and_description_set(self):
        self.assertIsInstance(self.tool.name, str)
        self.assertIsInstance(self.tool.description, str)
        self.assertGreater(len(self.tool.name), 0)
        self.assertGreater(len(self.tool.description), 0)

    def test_store_reports_add_and_update_counts(self):
        response = self.tool.run("store: I prefer Python over JavaScript.")
        self.assertIn("Added", response)


@unittest.skipUnless(
    _CREWAI_INTEGRATION_IMPORTABLE and _CREWAI_AVAILABLE,
    "crewai not installed",
)
class TestSynapseMemCrewStorage(unittest.TestCase):

    def setUp(self):
        self.synapse = _make_memory()
        self.crew_storage = SynapseMemCrewStorage(synapse=self.synapse)

    def test_save_ingests_into_memory(self):
        self.crew_storage.save("I love hiking.")
        records = self.synapse.storage.all()
        self.assertGreater(len(records), 0)

    def test_search_returns_list(self):
        self.synapse.ingest("I love hiking.")
        results = self.crew_storage.search("outdoor activities")
        self.assertIsInstance(results, list)

    def test_search_result_has_required_fields(self):
        self.synapse.ingest("I work on SynapseMem.")
        results = self.crew_storage.search("projects", top_k=3)
        if results:
            for r in results:
                self.assertIn("id", r)
                self.assertIn("text", r)
                self.assertIn("score", r)
                self.assertIn("metadata", r)

    def test_search_empty_returns_empty_list(self):
        results = self.crew_storage.search("nothing here")
        self.assertIsInstance(results, list)

    def test_reset_clears_memory(self):
        self.synapse.ingest("I like coffee.")
        self.crew_storage.reset()
        self.assertEqual(len(self.synapse.storage.all()), 0)

    def test_save_with_metadata(self):
        # metadata param is accepted but optional — should not raise
        self.crew_storage.save("I prefer dark mode.", metadata={"source": "chat"})
        records = self.synapse.storage.all()
        self.assertGreater(len(records), 0)


# ======================================================================== #
# Integration smoke test — manager wires up new backends correctly          #
# ======================================================================== #

class TestManagerPhase3Backends(unittest.TestCase):
    """
    Verifies that manager.py correctly routes to each new backend.
    Uses in-memory / local backends only — no servers needed.
    """

    def test_memory_backend_still_works(self):
        mem = SynapseMemory(storage_backend="memory")
        mem.ingest("I like hiking.")
        results = mem.retrieve("What do I like?")
        self.assertGreater(len(results), 0)

    def test_sqlite_backend_still_works(self):
        import tempfile, os
        # Use a temp file but don't delete it on Windows —
        # SQLite holds a lock until the process exits.
        # We just verify functionality, cleanup is best-effort.
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_name = tmp.name
        tmp.close()
        try:
            mem = SynapseMemory(
                storage_backend="sqlite",
                sqlite_db_path=tmp_name,
                user_id="test", agent_id="test", session_id="test",
            )
            mem.ingest("I prefer dark mode.")
            results = mem.retrieve("display preferences")
            self.assertGreater(len(results), 0)
        finally:
            # Best-effort cleanup — silently ignore Windows lock errors
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def test_invalid_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            SynapseMemory(storage_backend="unsupported_backend_xyz")


if __name__ == "__main__":
    unittest.main()