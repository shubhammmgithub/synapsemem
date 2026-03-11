import unittest

from synapsemem.prompt.builder import PromptBuilder


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_prompt_building(self):
        anchors = ["You are helpful."]
        memories = [
            {
                "subject": "user",
                "predicate": "likes",
                "object": "hiking",
                "topic": "preference",
                "score": 0.9,
            }
        ]
        query = "Suggest activities."

        prompt = self.builder.build(anchors, memories, query)

        self.assertIn("You are helpful.", prompt)
        self.assertIn("user likes hiking", prompt)
        self.assertIn("Suggest activities.", prompt)


if __name__ == "__main__":
    unittest.main()