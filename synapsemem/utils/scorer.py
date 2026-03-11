"""Scoring and ranking utilities"""
import math
from typing import List


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity manually."""
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)