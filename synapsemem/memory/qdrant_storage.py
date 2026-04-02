"""Qdrant vector DB storage adapter for SynapseMem.

Install:
    pip install qdrant-client

Run Qdrant locally (Docker):
    docker run -p 6333:6333 qdrant/qdrant

Or use Qdrant Cloud — just pass url + api_key.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ..utils.embeddings import get_embedding
from .base_storage import BaseMemoryStorage

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError as exc:
    raise ImportError(
        "qdrant-client is required. Install with: pip install qdrant-client"
    ) from exc

# Embedding dimension for all-MiniLM-L6-v2
_EMBEDDING_DIM = 384

# Payload field that stores the full record (minus the vector)
_PAYLOAD_KEY = "record"

# Status value used to soft-delete / merge / prune records
_ACTIVE = "active"


class QdrantMemoryStorage(BaseMemoryStorage):
    """
    Qdrant-backed memory storage.

    One Qdrant collection per (user_id, agent_id, session_id) scope.
    Vectors are stored in Qdrant; all metadata lives in the payload.

    Collection name format:
        synapsemem__{user_id}__{agent_id}__{session_id}
    (sanitised: spaces → underscores, non-alnum stripped)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
        embedding_dim: int = _EMBEDDING_DIM,
    ) -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.embedding_dim = embedding_dim

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = self._make_collection_name(user_id, agent_id, session_id)
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Collection bootstrap                                                 #
    # ------------------------------------------------------------------ #

    def _make_collection_name(self, user_id: str, agent_id: str, session_id: str) -> str:
        import re
        def sanitise(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", s)
        return f"synapsemem__{sanitise(user_id)}__{sanitise(agent_id)}__{sanitise(session_id)}"

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Payload indices for fast filtering
            for field in ("status", "memory_type", "subject", "predicate", "topic"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=f"{_PAYLOAD_KEY}.{field}",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add_triplets(self, triplets: List[Dict]) -> None:
        now = time.time()
        points: List[qmodels.PointStruct] = []

        for triplet in triplets:
            text_repr = self._triplet_to_text(triplet)
            embedding = get_embedding(text_repr)

            record_id = str(uuid.uuid4())
            record = {
                "id": record_id,
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "session_id": self.session_id,
                "subject": str(triplet["subject"]),
                "predicate": str(triplet["predicate"]),
                "object": str(triplet["object"]),
                "topic": str(triplet.get("topic", "general")),
                "priority": int(triplet.get("priority", 3)),
                "source_text": str(triplet.get("source_text", "")),
                "created_at": now,
                "updated_at": now,
                "last_accessed_at": None,
                "reinforcement_count": 0,
                "memory_type": str(triplet.get("memory_type", "episodic")),
                "status": str(triplet.get("status", _ACTIVE)),
                "source_count": int(triplet.get("source_count", 1)),
                "consolidated_from": list(triplet.get("consolidated_from", [])),
            }

            points.append(
                qmodels.PointStruct(
                    id=record_id,
                    vector=embedding,
                    payload={_PAYLOAD_KEY: record},
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool:
        existing = self._fetch_by_id(old_record_id)
        if not existing or existing.get("status", _ACTIVE) != _ACTIVE:
            return False

        text_repr = self._triplet_to_text(new_triplet)
        new_embedding = get_embedding(text_repr)

        updated_record = dict(existing)
        updated_record.update({
            "subject": str(new_triplet["subject"]),
            "predicate": str(new_triplet["predicate"]),
            "object": str(new_triplet["object"]),
            "topic": str(new_triplet.get("topic", existing.get("topic", "general"))),
            "priority": int(new_triplet.get("priority", existing.get("priority", 3))),
            "source_text": str(new_triplet.get("source_text", "")),
            "updated_at": time.time(),
        })

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=old_record_id,
                    vector=new_embedding,
                    payload={_PAYLOAD_KEY: updated_record},
                )
            ],
        )
        return True

    def reinforce(self, record_id: str) -> None:
        record = self._fetch_by_id(record_id)
        if not record or record.get("status", _ACTIVE) != _ACTIVE:
            return

        now = time.time()
        record["last_accessed_at"] = now
        record["reinforcement_count"] = int(record.get("reinforcement_count", 0)) + 1
        record["updated_at"] = now

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={_PAYLOAD_KEY: record},
            points=[record_id],
        )

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        records = self._filter_records(status=_ACTIVE)
        matching = [r for r in records if str(r.get("topic", "")).lower() == topic]

        now = time.time()
        for record in matching:
            record["status"] = "pruned"
            record["updated_at"] = now
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={_PAYLOAD_KEY: record},
                points=[record["id"]],
            )
        return len(matching)

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        records = self._filter_records(status=_ACTIVE)
        now = time.time()
        count = 0

        for record in records:
            match = (
                (subject is None or record["subject"] == subject)
                and (predicate is None or record["predicate"] == predicate)
                and (obj is None or record["object"] == obj)
            )
            if match:
                record["status"] = "pruned"
                record["updated_at"] = now
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={_PAYLOAD_KEY: record},
                    points=[record["id"]],
                )
                count += 1

        return count

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        merged_count = 0
        now = time.time()

        for action in merge_actions:
            record = self._fetch_by_id(action["record_id"])
            survivor = self._fetch_by_id(action["survivor_id"])

            if not record or not survivor:
                continue
            if record.get("status") != _ACTIVE or survivor.get("status") != _ACTIVE:
                continue

            # Update survivor
            survivor["source_count"] = int(survivor.get("source_count", 1)) + int(record.get("source_count", 1))
            cf = list(survivor.get("consolidated_from", []))
            cf.append(record["id"])
            survivor["consolidated_from"] = cf
            survivor["updated_at"] = now

            # Mark record as merged
            record["status"] = "merged"
            record["updated_at"] = now

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={_PAYLOAD_KEY: survivor},
                points=[survivor["id"]],
            )
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={_PAYLOAD_KEY: record},
                points=[record["id"]],
            )
            merged_count += 1

        return merged_count

    def prune_memories(self, prune_actions: List[Dict]) -> int:
        prune_ids = [a["record_id"] for a in prune_actions]
        if not prune_ids:
            return 0

        now = time.time()
        pruned = 0
        for record_id in prune_ids:
            record = self._fetch_by_id(record_id)
            if record and record.get("status") == _ACTIVE:
                record["status"] = "pruned"
                record["updated_at"] = now
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={_PAYLOAD_KEY: record},
                    points=[record_id],
                )
                pruned += 1

        return pruned

    def promote_to_semantic(self, source_records: List[Dict]) -> Optional[Dict]:
        if not source_records:
            return None

        survivor = source_records[0]
        existing = self.find_semantic_memory(
            survivor["subject"], survivor["predicate"], survivor["object"]
        )
        now = time.time()

        if existing is not None:
            cf = list(existing.get("consolidated_from", []))
            cf.extend([r["id"] for r in source_records])
            existing["consolidated_from"] = cf
            existing["source_count"] = int(existing.get("source_count", 1)) + len(source_records)
            existing["updated_at"] = now
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={_PAYLOAD_KEY: existing},
                points=[existing["id"]],
            )
            for record in source_records:
                if record["id"] == existing["id"]:
                    continue
                record["status"] = "merged"
                record["updated_at"] = now
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={_PAYLOAD_KEY: record},
                    points=[record["id"]],
                )
            return self.find_semantic_memory(survivor["subject"], survivor["predicate"], survivor["object"])

        self.add_triplets([{
            "subject": survivor["subject"],
            "predicate": survivor["predicate"],
            "object": survivor["object"],
            "topic": survivor.get("topic", "general"),
            "priority": max(int(r.get("priority", 3)) for r in source_records),
            "source_text": survivor.get("source_text", ""),
            "memory_type": "semantic",
            "status": _ACTIVE,
            "source_count": sum(int(r.get("source_count", 1)) for r in source_records),
            "consolidated_from": [r["id"] for r in source_records],
        }])

        for record in source_records:
            if record.get("status") == _ACTIVE:
                record["status"] = "merged"
                record["updated_at"] = now
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={_PAYLOAD_KEY: record},
                    points=[record["id"]],
                )

        return self.find_semantic_memory(survivor["subject"], survivor["predicate"], survivor["object"])

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def all(self) -> List[Dict]:
        return self._filter_records(status=_ACTIVE)

    def all_records(self) -> List[Dict]:
        return self._filter_records()

    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        records = self._filter_records(status=_ACTIVE, subject=subject, predicate=predicate)
        for record in records:
            if record["object"] == obj:
                return record
        return None

    def find_by_subject_predicate(self, subject: str, predicate: str) -> List[Dict]:
        return self._filter_records(status=_ACTIVE, subject=subject, predicate=predicate)

    def find_semantic_memory(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        records = self._filter_records(status=_ACTIVE, subject=subject, predicate=predicate)
        for record in records:
            if record.get("memory_type") == "semantic" and record["object"] == obj:
                return record
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_by_id(self, record_id: str) -> Optional[Dict]:
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[record_id],
            with_payload=True,
        )
        if not results:
            return None
        payload = results[0].payload or {}
        return payload.get(_PAYLOAD_KEY)

    def _filter_records(
        self,
        status: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[Dict]:
        """
        Scroll through all points, applying optional payload filters.
        Uses Qdrant's scroll API (no vector search needed here).
        """
        must_conditions = []

        if status is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key=f"{_PAYLOAD_KEY}.status",
                    match=qmodels.MatchValue(value=status),
                )
            )
        if subject is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key=f"{_PAYLOAD_KEY}.subject",
                    match=qmodels.MatchValue(value=subject),
                )
            )
        if predicate is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key=f"{_PAYLOAD_KEY}.predicate",
                    match=qmodels.MatchValue(value=predicate),
                )
            )

        scroll_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        records: List[Dict] = []
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
                limit=100,
                offset=offset,
            )
            for point in results:
                payload = point.payload or {}
                record = payload.get(_PAYLOAD_KEY)
                if record:
                    records.append(record)

            if next_offset is None:
                break
            offset = next_offset

        return records