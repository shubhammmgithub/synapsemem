"""Tests for memory decay logic"""

import time
import unittest

from synapsemem.memory.decay import compute_decay_score


class TestDecay(unittest.TestCase):
    def test_decay_recent_vs_old(self):
        now = time.time()
        recent = compute_decay_score(now - 60, decay_rate=0.05, now=now)
        old = compute_decay_score(now - (60 * 60 * 24), decay_rate=0.05, now=now)

        self.assertGreater(recent, old)


if __name__ == "__main__":
    unittest.main()