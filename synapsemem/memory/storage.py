"""Storage adapter - Chroma + Graph database integration"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from ..utils.embeddings import get_embedding


class MemoryStorage:
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
                "reinforcement_count": 0,
            }
            self.records.append(record)

    def all(self) -> List[Dict]:
        return list(self.records)

    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        for record in self.records:
            if (
                record["subject"] == subject
                and record["predicate"] == predicate
                and record["object"] == obj
            ):
                return record
        return None

    def find_by_subject_predicate(self, subject: str, predicate: str) -> List[Dict]:
        return [
            record
            for record in self.records
            if record["subject"] == subject and record["predicate"] == predicate
        ]

    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool:
        for idx, record in enumerate(self.records):
            if record["id"] == old_record_id:
                updated = {
                    "id": record["id"],
                    "subject": new_triplet["subject"],
                    "predicate": new_triplet["predicate"],
                    "object": new_triplet["object"],
                    "topic": new_triplet.get("topic", record.get("topic", "general")),
                    "priority": int(new_triplet.get("priority", record.get("priority", 3))),
                    "source_text": new_triplet.get("source_text", ""),
                    "embedding": get_embedding(self._triplet_to_text(new_triplet)),
                    "created_at": record["created_at"],
                    "last_accessed_at": record.get("last_accessed_at"),
                    "reinforcement_count": record.get("reinforcement_count", 0),
                }
                self.records[idx] = updated
                return True
        return False

    def reinforce(self, record_id: str) -> None:
        for record in self.records:
            if record["id"] == record_id:
                record["last_accessed_at"] = time.time()
                record["reinforcement_count"] = int(record.get("reinforcement_count", 0)) + 1
                break

    def update_last_accessed(self, record_id: str) -> None:
        self.reinforce(record_id)

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
            subject_match = subject is None or record["subject"] == subject
            predicate_match = predicate is None or record["predicate"] == predicate
            object_match = obj is None or record["object"] == obj
            return not (subject_match and predicate_match and object_match)

        self.records = [r for r in self.records if keep(r)]
        return before - len(self.records)

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        duplicate_ids = {action["record_id"] for action in merge_actions}
        before = len(self.records)
        self.records = [r for r in self.records if r["id"] not in duplicate_ids]
        return before - len(self.records)

    def prune_memories(self, prune_actions: List[Dict]) -> int:
        prune_ids = {action["record_id"] for action in prune_actions}
        before = len(self.records)
        self.records = [r for r in self.records if r["id"] not in prune_ids]
        return before - len(self.records)

    def reset(self) -> None:
        self.records.clear()

    def _triplet_to_text(self, triplet: Dict) -> str:
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"