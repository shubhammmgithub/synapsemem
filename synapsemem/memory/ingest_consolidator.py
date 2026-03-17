"""Online ingest consolidator - deduplication and conflict resolution."""

from __future__ import annotations

from typing import Dict, List, Tuple, Any


class IngestConsolidator:
    """
    Consolidates extracted triplets during ingest and decides whether each one should be:
    - ADD
    - UPDATE
    - DELETE
    - NOOP

    This class is for the online ingest path only.
    It should stay lightweight and deterministic.
    """

    DELETE_PREFIXES = (
        "forget that",
        "remove that",
        "delete that",
        "forget ",
        "remove ",
        "delete ",
    )

    def process(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = [self._normalize_triplet(t) for t in triplets]
        return self._deduplicate(normalized)

    def decide_actions(
        self,
        triplets: List[Dict[str, Any]],
        storage,
    ) -> List[Dict[str, Any]]:
        """
        Compare incoming triplets against existing storage and assign actions.
        """
        processed = self.process(triplets)
        decisions: List[Dict[str, Any]] = []

        for triplet in processed:
            source_text = str(triplet.get("source_text", "")).strip().lower()

            if self._is_delete_request(source_text):
                decisions.append({
                    "action": "DELETE",
                    "triplet": triplet,
                })
                continue

            exact = storage.find_exact(
                triplet["subject"],
                triplet["predicate"],
                triplet["object"],
            )
            if exact is not None:
                decisions.append({
                    "action": "NOOP",
                    "triplet": triplet,
                    "existing": exact,
                })
                continue

            same_relation = storage.find_by_subject_predicate(
                triplet["subject"],
                triplet["predicate"],
            )
            if same_relation:
                decisions.append({
                    "action": "UPDATE",
                    "triplet": triplet,
                    "existing": same_relation[0],
                })
                continue

            decisions.append({
                "action": "ADD",
                "triplet": triplet,
            })

        return decisions

    def _normalize_triplet(self, triplet: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(triplet)
        normalized["subject"] = str(normalized["subject"]).strip().lower()
        normalized["predicate"] = str(normalized["predicate"]).strip().lower()
        normalized["object"] = str(normalized["object"]).strip()
        normalized["topic"] = str(normalized.get("topic", "general")).strip().lower()
        normalized["priority"] = int(normalized.get("priority", 3))
        normalized["source_text"] = str(normalized.get("source_text", "")).strip()
        return normalized

    def _deduplicate(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

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

    def _is_delete_request(self, source_text: str) -> bool:
        return source_text.startswith(self.DELETE_PREFIXES)