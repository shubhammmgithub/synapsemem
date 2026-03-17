"""Biological decay logic for memory fade"""

from __future__ import annotations

import math
import time


def compute_decay_score(
    last_accessed_at: float | None,
    created_at: float | None = None,
    reinforcement_count: int = 0,
    priority: int = 3,
    decay_rate: float = 0.05,
    now: float | None = None,
) -> float:
    """
    Bio-inspired decay score in (0, 1].

    Stronger memories decay more slowly if they have:
    - higher reinforcement_count
    - higher priority
    - recent access

    Formula idea:
        base_decay = exp(-effective_decay_rate * age_hours)

    effective_decay_rate is reduced when:
    - reinforcement_count is higher
    - priority is higher
    """
    current_time = now if now is not None else time.time()

    # If never accessed, fall back to created_at
    reference_time = last_accessed_at if last_accessed_at is not None else created_at

    if reference_time is None:
        return 1.0

    age_seconds = max(0.0, current_time - reference_time)
    age_hours = age_seconds / 3600.0

    reinforcement_factor = 1.0 / (1.0 + 0.25 * max(reinforcement_count, 0))
    priority_factor = 1.0 / (1.0 + 0.1 * max(priority, 0))

    effective_decay_rate = decay_rate * reinforcement_factor * priority_factor
    score = math.exp(-effective_decay_rate * age_hours)

    return max(0.0, min(1.0, score))


def compute_synaptic_strength(
    reinforcement_count: int,
    priority: int,
    decay_score: float,
) -> float:
    """
    Synaptic strength in [0, 1].

    Combines:
    - reinforcement_count
    - priority
    - current decay score
    """
    reinforcement_component = min(reinforcement_count / 10.0, 1.0)
    priority_component = min(priority / 10.0, 1.0)

    strength = (
        reinforcement_component * 0.4
        + priority_component * 0.3
        + decay_score * 0.3
    )
    return max(0.0, min(1.0, strength))