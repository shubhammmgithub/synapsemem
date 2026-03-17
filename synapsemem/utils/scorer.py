"""Scoring and ranking utilities"""
import math
import re
from typing import Iterable, List, Set


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def final_memory_score(
    semantic_similarity: float,
    priority_score: float,
    decay_score: float,
    synaptic_strength: float,
    anchor_bonus: float = 0.0,
    graph_bonus: float = 0.0,
) -> float:
    return (
        semantic_similarity * 0.38
        + priority_score * 0.18
        + decay_score * 0.12
        + synaptic_strength * 0.18
        + anchor_bonus * 0.07
        + graph_bonus * 0.07
    )


def tokenize_keywords(text: str) -> Set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    stopwords = {
        "the", "a", "an", "is", "am", "are", "was", "were", "be", "to", "of", "and",
        "or", "in", "on", "for", "with", "that", "this", "it", "i", "me", "my",
        "you", "your", "what", "where", "when", "how", "do", "does", "did", "should",
        "suggest", "idea", "place", "go"
    }
    return {w for w in words if w not in stopwords and len(w) > 1}


def compute_anchor_bonus(record_text: str, anchors: Iterable[str]) -> float:
    record_tokens = tokenize_keywords(record_text)
    if not record_tokens:
        return 0.0

    best_overlap = 0.0
    for anchor in anchors:
        anchor_tokens = tokenize_keywords(anchor)
        if not anchor_tokens:
            continue
        overlap = len(record_tokens & anchor_tokens) / max(len(record_tokens), 1)
        best_overlap = max(best_overlap, overlap)

    return min(best_overlap, 1.0)


def compute_graph_bonus(
    record_subject: str,
    record_object: str,
    query_entities: Iterable[str],
    direct_entities: Iterable[str],
    nearby_entities: Iterable[str],
) -> float:
    """
    Graph bonus strategy:
    - exact hit with direct graph entities from query => strong bonus
    - overlap with nearby graph neighborhood => medium bonus
    """
    subject = str(record_subject).strip().lower()
    obj = str(record_object).strip().lower()

    query_set = {str(x).strip().lower() for x in query_entities}
    direct_set = {str(x).strip().lower() for x in direct_entities}
    nearby_set = {str(x).strip().lower() for x in nearby_entities}

    bonus = 0.0

    # Exact direct hits are strongest
    if subject in direct_set:
        bonus += 0.45
    if obj in direct_set:
        bonus += 0.45

    # Nearby graph neighbors add smaller bonus
    if subject in nearby_set:
        bonus += 0.25
    if obj in nearby_set:
        bonus += 0.25

    # If object text contains query entities, add a small semantic graph hint
    obj_tokens = tokenize_keywords(obj)
    if obj_tokens & query_set:
        bonus += 0.15

    return min(bonus, 1.0)