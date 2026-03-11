"""Storage adapter - Chroma + Graph database integration"""


from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from ..utils.embeddings import get_embedding


class MemoryStorage:
    """
    Minimal in-memory storage for SynapseMem MVP.

    Each record has:
    - id
    - subject
    - predicate
    - object
    - topic
    - priority
    - source_text
    - embedding
    - created_at
    - last_accessed_at
    """

    def __init__(self) -> None:
        self.records: List[Dict] = []

    def add_triplets(self, triplets: List[Dict]) -> None:
        now = time.time()

        for triplet in triplets:
            text_repr = self._triplet_to_text(triplet)
            record = {
                "id": str(uuid.uuid4()),
                "subject": triplet["subject"],
                "predicate": triplet["predicate"],
                "object": triplet["object"],
                "topic": triplet.get("topic", "general"),
                "priority": int(triplet.get("priority", 3)),
                "source_text": triplet.get("source_text", ""),
                "embedding": get_embedding(text_repr),
                "created_at": now,
                "last_accessed_at": None,
            }
            self.records.append(record)

    def all(self) -> List[Dict]:
        return list(self.records)

    def update_last_accessed(self, record_id: str) -> None:
        for record in self.records:
            if record["id"] == record_id:
                record["last_accessed_at"] = time.time()
                break

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        before = len(self.records)
        self.records = [r for r in self.records if str(r.get("topic", "")).lower() != topic]
        return before - len(self.records)

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        before = len(self.records)

        def keep(record: Dict) -> bool:
            if subject is not None and record["subject"] != subject:
                return True
            if predicate is not None and record["predicate"] != predicate:
                return True
            if obj is not None and record["object"] != obj:
                return True

            subject_match = subject is None or record["subject"] == subject
            predicate_match = predicate is None or record["predicate"] == predicate
            object_match = obj is None or record["object"] == obj
            return not (subject_match and predicate_match and object_match)

        self.records = [r for r in self.records if keep(r)]
        return before - len(self.records)

    def reset(self) -> None:
        self.records.clear()

    def _triplet_to_text(self, triplet: Dict) -> str:
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"