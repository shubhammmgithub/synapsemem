"""Hybrid retrieval engine for memory access"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..utils.embeddings import get_embedding
from ..utils.scorer import (
    compute_anchor_bonus,
    compute_graph_bonus,
    cosine_sim,
    final_memory_score,
    tokenize_keywords,
)
from .decay import compute_decay_score, compute_synaptic_strength


class MemoryRetriever:
    def __init__(self, storage, anchor_manager=None, graph_query=None) -> None:
        self.storage = storage
        self.anchor_manager = anchor_manager
        self.graph_query = graph_query

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = get_embedding(query)
        query_entities = tokenize_keywords(query)

        anchors = self.anchor_manager.get_anchors() if self.anchor_manager else []

        direct_entities, nearby_entities = self._collect_graph_context(query_entities)

        scored: List[Dict] = []

        for record in self.storage.all():
            similarity = cosine_sim(query_embedding, record["embedding"])
            priority_score = min(record["priority"] / 10.0, 1.0)

            decay_score = compute_decay_score(
                last_accessed_at=record.get("last_accessed_at"),
                created_at=record.get("created_at"),
                reinforcement_count=int(record.get("reinforcement_count", 0)),
                priority=int(record.get("priority", 3)),
                decay_rate=0.05,
            )

            synaptic_strength = compute_synaptic_strength(
                reinforcement_count=int(record.get("reinforcement_count", 0)),
                priority=int(record.get("priority", 3)),
                decay_score=decay_score,
            )

            record_text = f"{record['subject']} {record['predicate']} {record['object']}"
            anchor_bonus = compute_anchor_bonus(record_text, anchors)

            graph_bonus = compute_graph_bonus(
                record_subject=record["subject"],
                record_object=record["object"],
                query_entities=query_entities,
                direct_entities=direct_entities,
                nearby_entities=nearby_entities,
            )

            score = final_memory_score(
                semantic_similarity=similarity,
                priority_score=priority_score,
                decay_score=decay_score,
                synaptic_strength=synaptic_strength,
                anchor_bonus=anchor_bonus,
                graph_bonus=graph_bonus,
            )

            enriched = dict(record)
            enriched["score"] = round(score, 6)
            enriched["semantic_similarity"] = round(similarity, 6)
            enriched["priority_score"] = round(priority_score, 6)
            enriched["decay_score"] = round(decay_score, 6)
            enriched["synaptic_strength"] = round(synaptic_strength, 6)
            enriched["anchor_bonus"] = round(anchor_bonus, 6)
            enriched["graph_bonus"] = round(graph_bonus, 6)
            scored.append(enriched)

        scored.sort(key=lambda item: item["score"], reverse=True)
        top_records = scored[:top_k]

        for record in top_records:
            self.storage.reinforce(record["id"])
            record["reinforcement_count"] = int(record.get("reinforcement_count", 0)) + 1

        return top_records

    def _collect_graph_context(self, query_entities: Set[str]) -> tuple[list[str], list[str]]:
        """
        Build graph context for the query.

        direct_entities:
            query terms that are actual graph nodes

        nearby_entities:
            neighbors within 2 hops of matched graph nodes
        """
        if not self.graph_query:
            return [], []

        direct: Set[str] = set()
        nearby: Set[str] = set()

        for entity in query_entities:
            try:
                # If entity itself is a graph node, treat as direct
                facts = self.graph_query.facts_about(entity)
                reverse = self.graph_query.facts_pointing_to(entity)

                if facts or reverse:
                    direct.add(entity)

                # Expand nearby nodes
                for item in self.graph_query.related_entities(entity, max_depth=2):
                    nearby.add(str(item).strip().lower())

            except Exception:
                continue

        return list(direct), list(nearby)