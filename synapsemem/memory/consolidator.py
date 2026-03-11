"""Memory consolidation - deduplication and conflict resolution"""

from __future__ import annotations

from typing import Dict, List, Tuple


class MemoryConsolidator:
    """
    Consolidates extracted triplets:
    - removes duplicates
    - keeps higher-priority versions
    - performs simple normalization
    """

    def process(self, triplets: List[Dict]) -> List[Dict]:
        normalized = [self._normalize_triplet(t) for t in triplets]
        deduped = self._deduplicate(normalized)
        return deduped

    def _normalize_triplet(self, triplet: Dict) -> Dict:
        triplet = dict(triplet)
        triplet["subject"] = str(triplet["subject"]).strip().lower()
        triplet["predicate"] = str(triplet["predicate"]).strip().lower()
        triplet["object"] = str(triplet["object"]).strip()
        triplet["topic"] = str(triplet.get("topic", "general")).strip().lower()
        triplet["priority"] = int(triplet.get("priority", 3))
        triplet["source_text"] = str(triplet.get("source_text", "")).strip()
        return triplet

    def _deduplicate(self, triplets: List[Dict]) -> List[Dict]:
        best_by_key: dict[Tuple[str, str, str], Dict] = {}

        for triplet in triplets:
            key = (
                triplet["subject"],
                triplet["predicate"],
                triplet["object"].lower(),
            )
            current = best_by_key.get(key)

            if current is None or triplet["priority"] > current["priority"]:
                best_by_key[key] = triplet

        return list(best_by_key.values())