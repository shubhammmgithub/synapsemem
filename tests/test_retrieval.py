"""Tests for retrieval engine"""

import unittest

from synapsemem import SynapseMemory


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.memory = SynapseMemory()

    def test_ingest_and_retrieve(self):
        self.memory.ingest("I love hiking.")
        self.memory.ingest("I work on SynapseMem.")

        results = self.memory.retrieve("What do I love?")
        self.assertTrue(len(results) > 0)

        found = any("hiking" in r["object"].lower() for r in results)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()