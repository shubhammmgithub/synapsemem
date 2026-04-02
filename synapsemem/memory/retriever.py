"""Hybrid retrieval engine for memory access."""

from __future__ import annotations

from typing import Dict, List, Set

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

        # Phase 2 retrieval weights
        self.semantic_memory_bonus = 0.20
        self.source_count_weight = 0.03
        self.max_source_count_bonus = 0.20

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = get_embedding(query)
        query_entities = tokenize_keywords(query)

        anchors = self.anchor_manager.get_anchors() if self.anchor_manager else []
        direct_entities, nearby_entities = self._collect_graph_context(query_entities)

        scored: List[Dict] = []

        for record in self.storage.all():
            if record.get("status", "active") != "active":
                continue

            # ── Embedding ──────────────────────────────────────────────
            # Qdrant and Chroma store vectors in the vector store, not in
            # the payload. The record returns embedding=[] in those cases.
            # We recompute it from the triplet text when needed.
            record_embedding = record.get("embedding")
            if not record_embedding:
                text_repr = (
                    f"{record['subject']} "
                    f"{record['predicate']} "
                    f"{record['object']}"
                )
                record_embedding = get_embedding(text_repr)

            similarity = cosine_sim(query_embedding, record_embedding)
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

            record_text = (
                f"{record['subject']} "
                f"{record['predicate']} "
                f"{record['object']}"
            )
            anchor_bonus = compute_anchor_bonus(record_text, anchors)

            graph_bonus = compute_graph_bonus(
                record_subject=record["subject"],
                record_object=record["object"],
                query_entities=query_entities,
                direct_entities=direct_entities,
                nearby_entities=nearby_entities,
            )

            base_score = final_memory_score(
                semantic_similarity=similarity,
                priority_score=priority_score,
                decay_score=decay_score,
                synaptic_strength=synaptic_strength,
                anchor_bonus=anchor_bonus,
                graph_bonus=graph_bonus,
            )

            memory_type = str(record.get("memory_type", "episodic")).lower()
            source_count = int(record.get("source_count", 1))

            semantic_bonus = self.semantic_memory_bonus if memory_type == "semantic" else 0.0
            source_count_bonus = min(
                max(source_count - 1, 0) * self.source_count_weight,
                self.max_source_count_bonus,
            )

            final_score = base_score + semantic_bonus + source_count_bonus

            enriched = dict(record)
            enriched["score"] = round(final_score, 6)
            enriched["base_score"] = round(base_score, 6)
            enriched["semantic_similarity"] = round(similarity, 6)
            enriched["priority_score"] = round(priority_score, 6)
            enriched["decay_score"] = round(decay_score, 6)
            enriched["synaptic_strength"] = round(synaptic_strength, 6)
            enriched["anchor_bonus"] = round(anchor_bonus, 6)
            enriched["graph_bonus"] = round(graph_bonus, 6)
            enriched["semantic_memory_bonus"] = round(semantic_bonus, 6)
            enriched["source_count_bonus"] = round(source_count_bonus, 6)
            enriched["memory_type"] = memory_type
            enriched["source_count"] = source_count

            scored.append(enriched)

        scored.sort(
            key=lambda item: (
                item["score"],
                item.get("memory_type") == "semantic",
                item.get("source_count", 1),
            ),
            reverse=True,
        )

        top_records = scored[:top_k]

        for record in top_records:
            self.storage.reinforce(record["id"])
            record["reinforcement_count"] = int(record.get("reinforcement_count", 0)) + 1

        return top_records

    def _collect_graph_context(self, query_entities: Set[str]) -> tuple[list[str], list[str]]:
        if not self.graph_query:
            return [], []

        direct: Set[str] = set()
        nearby: Set[str] = set()

        for entity in query_entities:
            try:
                facts = self.graph_query.facts_about(entity)

                reverse = []
                if hasattr(self.graph_query, "facts_pointing_to"):
                    reverse = self.graph_query.facts_pointing_to(entity)

                if facts or reverse:
                    direct.add(entity)

                for item in self.graph_query.related_entities(entity, max_depth=2):
                    nearby.add(str(item).strip().lower())

            except Exception:
                continue

        return list(direct), list(nearby)