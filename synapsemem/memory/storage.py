"""Storage adapter - in-memory memory store."""

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
                "updated_at": now,
                "last_accessed_at": None,
                "reinforcement_count": 0,
                "memory_type": triplet.get("memory_type", "episodic"),
                "status": triplet.get("status", "active"),
                "source_count": int(triplet.get("source_count", 1)),
                "consolidated_from": list(triplet.get("consolidated_from", [])),
            }
            self.records.append(record)

    def all(self) -> List[Dict]:
        return [r for r in self.records if r.get("status", "active") == "active"]

    def all_records(self) -> List[Dict]:
        return list(self.records)

    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        for record in self.records:
            if record.get("status", "active") != "active":
                continue
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
            if record.get("status", "active") == "active"
            and record["subject"] == subject
            and record["predicate"] == predicate
        ]

    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool:
        now = time.time()

        for idx, record in enumerate(self.records):
            if record["id"] == old_record_id and record.get("status", "active") == "active":
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
                    "updated_at": now,
                    "last_accessed_at": record.get("last_accessed_at"),
                    "reinforcement_count": record.get("reinforcement_count", 0),
                    "memory_type": record.get("memory_type", "episodic"),
                    "status": record.get("status", "active"),
                    "source_count": int(record.get("source_count", 1)),
                    "consolidated_from": list(record.get("consolidated_from", [])),
                }
                self.records[idx] = updated
                return True
        return False

    def reinforce(self, record_id: str) -> None:
        now = time.time()
        for record in self.records:
            if record["id"] == record_id and record.get("status", "active") == "active":
                record["last_accessed_at"] = now
                record["reinforcement_count"] = int(record.get("reinforcement_count", 0)) + 1
                record["updated_at"] = now
                break

    def update_last_accessed(self, record_id: str) -> None:
        self.reinforce(record_id)

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        count = 0
        now = time.time()

        for record in self.records:
            if (
                record.get("status", "active") == "active"
                and str(record.get("topic", "")).lower() == topic
            ):
                record["status"] = "pruned"
                record["updated_at"] = now
                count += 1

        return count

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        count = 0
        now = time.time()

        for record in self.records:
            if record.get("status", "active") != "active":
                continue

            subject_match = subject is None or record["subject"] == subject
            predicate_match = predicate is None or record["predicate"] == predicate
            object_match = obj is None or record["object"] == obj

            if subject_match and predicate_match and object_match:
                record["status"] = "pruned"
                record["updated_at"] = now
                count += 1

        return count

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        merged_count = 0
        now = time.time()

        by_id = {record["id"]: record for record in self.records}

        for action in merge_actions:
            record = by_id.get(action["record_id"])
            survivor = by_id.get(action["survivor_id"])

            if not record or not survivor:
                continue
            if record.get("status", "active") != "active":
                continue
            if survivor.get("status", "active") != "active":
                continue

            record["status"] = "merged"
            record["updated_at"] = now

            survivor["source_count"] = int(survivor.get("source_count", 1)) + int(
                record.get("source_count", 1)
            )
            survivor["updated_at"] = now

            consolidated_from = list(survivor.get("consolidated_from", []))
            consolidated_from.append(record["id"])
            survivor["consolidated_from"] = consolidated_from

            merged_count += 1

        return merged_count

    def prune_memories(self, prune_actions: List[Dict]) -> int:
        pruned_count = 0
        now = time.time()

        prune_ids = {action["record_id"] for action in prune_actions}

        for record in self.records:
            if record["id"] in prune_ids and record.get("status", "active") == "active":
                record["status"] = "pruned"
                record["updated_at"] = now
                pruned_count += 1

        return pruned_count

    def find_semantic_memory(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        for record in self.records:
            if record.get("status", "active") != "active":
                continue
            if record.get("memory_type") != "semantic":
                continue
            if (
                record["subject"] == subject
                and record["predicate"] == predicate
                and record["object"] == obj
            ):
                return record
        return None

    def promote_to_semantic(self, source_records: List[Dict]) -> Optional[Dict]:
        if not source_records:
            return None

        survivor = source_records[0]
        existing = self.find_semantic_memory(
            survivor["subject"],
            survivor["predicate"],
            survivor["object"],
        )
        now = time.time()

        if existing is not None:
            existing["source_count"] = int(existing.get("source_count", 1)) + len(source_records)
            existing["updated_at"] = now

            consolidated_from = list(existing.get("consolidated_from", []))
            consolidated_from.extend([r["id"] for r in source_records])
            existing["consolidated_from"] = consolidated_from

            for record in source_records:
                if record["id"] == existing["id"]:
                    continue
                if record.get("status", "active") == "active":
                    record["status"] = "merged"
                    record["updated_at"] = now

            return existing

        semantic_record = {
            "subject": survivor["subject"],
            "predicate": survivor["predicate"],
            "object": survivor["object"],
            "topic": survivor.get("topic", "general"),
            "priority": max(int(r.get("priority", 3)) for r in source_records),
            "source_text": survivor.get("source_text", ""),
            "memory_type": "semantic",
            "status": "active",
            "source_count": sum(int(r.get("source_count", 1)) for r in source_records),
            "consolidated_from": [r["id"] for r in source_records],
        }
        self.add_triplets([semantic_record])

        for record in source_records:
            if record.get("status", "active") == "active":
                record["status"] = "merged"
                record["updated_at"] = now

        return self.find_semantic_memory(
            survivor["subject"],
            survivor["predicate"],
            survivor["object"],
        )

    def reset(self) -> None:
        self.records.clear()

    def _triplet_to_text(self, triplet: Dict) -> str:
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"