"""Tests for triplet extraction module"""

import unittest

from synapsemem.memory.extractor import TripletExtractor


class TestTripletExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TripletExtractor()

    def test_extract_love(self):
        triplets = self.extractor.extract("I love hiking.")
        self.assertTrue(len(triplets) > 0)
        self.assertEqual(triplets[0]["subject"], "user")
        self.assertEqual(triplets[0]["predicate"], "loves")
        self.assertIn("hiking", triplets[0]["object"].lower())

    def test_extract_project(self):
        triplets = self.extractor.extract("I work on SynapseMem.")
        self.assertTrue(len(triplets) > 0)
        self.assertEqual(triplets[0]["predicate"], "works_on")


if __name__ == "__main__":
    unittest.main()