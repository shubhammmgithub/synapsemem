"""End-to-end integration tests"""

import unittest

from synapsemem import SynapseMemory


def fake_llm(prompt: str) -> str:
    return f"FAKE RESPONSE\n{prompt}"


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.memory = SynapseMemory(
            llm=fake_llm,
            pinned_facts=["You are a helpful assistant."]
        )

    def test_full_flow(self):
        response = self.memory.chat("I love hiking.")
        self.assertIn("FAKE RESPONSE", response)

        results = self.memory.retrieve("What do I love?")
        self.assertTrue(len(results) > 0)

        prompt = self.memory.build_prompt("Suggest activities", results)
        self.assertIn("helpful assistant", prompt)
        self.assertIn("hiking", prompt)


if __name__ == "__main__":
    unittest.main()