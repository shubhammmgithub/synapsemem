"""Hybrid retrieval engine for memory access"""

from __future__ import annotations

from typing import Dict, List

from ..utils.embeddings import get_embedding
from ..utils.scorer import cosine_sim
from .decay import compute_decay_score


class MemoryRetriever:
    """
    Retrieves the most relevant memories from storage.

    Scoring formula:
        score = semantic_similarity * 0.6
              + priority_score      * 0.3
              + decay_score         * 0.1
    """

    def __init__(self, storage) -> None:
        self.storage = storage

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = get_embedding(query)
        scored: List[Dict] = []

        for record in self.storage.all():
            similarity = cosine_sim(query_embedding, record["embedding"])
            priority_score = min(record["priority"] / 10.0, 1.0)
            decay_score = compute_decay_score(record.get("last_accessed_at"), decay_rate=0.05)

            score = (similarity * 0.6) + (priority_score * 0.3) + (decay_score * 0.1)

            enriched = dict(record)
            enriched["score"] = round(score, 6)
            enriched["semantic_similarity"] = round(similarity, 6)
            enriched["priority_score"] = round(priority_score, 6)
            enriched["decay_score"] = round(decay_score, 6)
            scored.append(enriched)

        scored.sort(key=lambda item: item["score"], reverse=True)
        top_records = scored[:top_k]

        for record in top_records:
            self.storage.update_last_accessed(record["id"])

        return top_records