"""Biological decay logic for memory fade"""

from __future__ import annotations

import math
import time


def compute_decay_score(
    last_accessed_at: float | None,
    decay_rate: float = 0.05,
    now: float | None = None,
) -> float:
    """
    Returns a score in (0, 1].
    More recent memories get higher scores.

    Formula:
        exp(-decay_rate * age_hours)
    """
    if last_accessed_at is None:
        return 1.0

    current_time = now if now is not None else time.time()
    age_seconds = max(0.0, current_time - last_accessed_at)
    age_hours = age_seconds / 3600.0
    return math.exp(-decay_rate * age_hours)