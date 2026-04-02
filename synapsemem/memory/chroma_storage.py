"""Chroma vector DB storage adapter for SynapseMem.

Install:
    pip install chromadb

Chroma runs in-process by default (no server needed for local use).
For a persistent client:
    ChromaMemoryStorage(persist_directory="./chroma_db")

For a remote HTTP client:
    ChromaMemoryStorage(host="localhost", port=8000)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from ..utils.embeddings import get_embedding
from .base_storage import BaseMemoryStorage

try:
    import chromadb
except ImportError as exc:
    raise ImportError(
        "chromadb is required. Install with: pip install chromadb"
    ) from exc

_ACTIVE = "active"


class ChromaMemoryStorage(BaseMemoryStorage):
    """
    Chroma-backed memory storage.

    One Chroma collection per (user_id, agent_id, session_id) scope.
    Vectors are stored in Chroma; metadata lives in the document/metadata fields.

    KEY RULE: 'embedding' is NEVER stored in Chroma metadata.
    It lives only in the vector store. All update() calls must use
    _build_metadata() which enforces this invariant.

    Chroma metadata values must be str | int | float | bool.
    Lists (consolidated_from) are JSON-encoded as strings.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = "./chroma_db",
        host: Optional[str] = None,
        port: int = 8000,
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
    ) -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id

        if host is not None:
            self.client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory is not None:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # EphemeralClient — each instance is fully isolated in-process.
            self.client = chromadb.EphemeralClient()

        self.collection_name = self._make_collection_name(user_id, agent_id, session_id)
        self.collection = self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Collection bootstrap                                                 #
    # ------------------------------------------------------------------ #

    def _make_collection_name(self, user_id: str, agent_id: str, session_id: str) -> str:
        import re
        def sanitise(s: str) -> str:
            s = re.sub(r"[^a-zA-Z0-9\-]", "-", s)
            return s[:20]
        # Unique suffix per instance prevents test cross-contamination
        # when multiple ChromaMemoryStorage objects share the same process.
        suffix = str(uuid.uuid4())[:8]
        return f"sm--{sanitise(user_id)}--{sanitise(agent_id)}--{suffix}"

    def _ensure_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def add_triplets(self, triplets: List[Dict]) -> None:
        now = time.time()
        ids, embeddings, metadatas, documents = [], [], [], []

        for triplet in triplets:
            text_repr = self._triplet_to_text(triplet)
            embedding = get_embedding(text_repr)
            record_id = str(uuid.uuid4())

            meta = self._build_metadata(
                record_id=record_id,
                triplet=triplet,
                now=now,
                last_accessed_at=None,
                reinforcement_count=0,
                memory_type=str(triplet.get("memory_type", "episodic")),
                status=str(triplet.get("status", _ACTIVE)),
                source_count=int(triplet.get("source_count", 1)),
                consolidated_from=list(triplet.get("consolidated_from", [])),
            )

            ids.append(record_id)
            embeddings.append(embedding)
            metadatas.append(meta)
            documents.append(text_repr)

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool:
        existing = self._fetch_by_id(old_record_id)
        if not existing or existing.get("status", _ACTIVE) != _ACTIVE:
            return False

        text_repr = self._triplet_to_text(new_triplet)
        new_embedding = get_embedding(text_repr)

        meta = self._build_metadata(
            record_id=old_record_id,
            triplet=new_triplet,
            now=time.time(),
            last_accessed_at=existing.get("last_accessed_at"),
            reinforcement_count=int(existing.get("reinforcement_count", 0)),
            memory_type=str(existing.get("memory_type", "episodic")),
            status=str(existing.get("status", _ACTIVE)),
            source_count=int(existing.get("source_count", 1)),
            consolidated_from=list(existing.get("consolidated_from", [])),
            created_at=existing.get("created_at"),
        )

        self.collection.update(
            ids=[old_record_id],
            embeddings=[new_embedding],
            metadatas=[meta],
            documents=[text_repr],
        )
        return True

    def reinforce(self, record_id: str) -> None:
        existing = self._fetch_by_id(record_id)
        if not existing or existing.get("status", _ACTIVE) != _ACTIVE:
            return

        now = time.time()
        meta = self._build_metadata(
            record_id=record_id,
            triplet=existing,
            now=now,
            last_accessed_at=now,
            reinforcement_count=int(existing.get("reinforcement_count", 0)) + 1,
            memory_type=str(existing.get("memory_type", "episodic")),
            status=str(existing.get("status", _ACTIVE)),
            source_count=int(existing.get("source_count", 1)),
            consolidated_from=list(existing.get("consolidated_from", [])),
            created_at=existing.get("created_at"),
        )
        text_repr = self._triplet_to_text(existing)
        embedding = get_embedding(text_repr)
        self.collection.update(
            ids=[record_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[text_repr],
        )

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        records = self._get_all_records_raw()
        now = time.time()
        count = 0

        for record_id, meta in records:
            if str(meta.get("topic", "")).lower() == topic and meta.get("status") == _ACTIVE:
                meta["status"] = "pruned"
                meta["updated_at"] = now
                self.collection.update(ids=[record_id], metadatas=[meta])
                count += 1

        return count

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        records = self._get_all_records_raw()
        now = time.time()
        count = 0

        for record_id, meta in records:
            if meta.get("status") != _ACTIVE:
                continue
            match = (
                (subject is None or meta.get("subject") == subject)
                and (predicate is None or meta.get("predicate") == predicate)
                and (obj is None or meta.get("object") == obj)
            )
            if match:
                meta["status"] = "pruned"
                meta["updated_at"] = now
                self.collection.update(ids=[record_id], metadatas=[meta])
                count += 1

        return count

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        now = time.time()
        merged_count = 0

        for action in merge_actions:
            record = self._fetch_by_id(action["record_id"])
            survivor = self._fetch_by_id(action["survivor_id"])

            if not record or not survivor:
                continue
            if record.get("status") != _ACTIVE or survivor.get("status") != _ACTIVE:
                continue

            survivor_meta = self._build_metadata(
                record_id=survivor["id"],
                triplet=survivor,
                now=now,
                last_accessed_at=survivor.get("last_accessed_at"),
                reinforcement_count=int(survivor.get("reinforcement_count", 0)),
                memory_type=str(survivor.get("memory_type", "episodic")),
                status=_ACTIVE,
                source_count=int(survivor.get("source_count", 1)) + int(record.get("source_count", 1)),
                consolidated_from=list(survivor.get("consolidated_from", [])) + [record["id"]],
                created_at=survivor.get("created_at"),
            )
            record_meta = self._build_metadata(
                record_id=record["id"],
                triplet=record,
                now=now,
                last_accessed_at=record.get("last_accessed_at"),
                reinforcement_count=int(record.get("reinforcement_count", 0)),
                memory_type=str(record.get("memory_type", "episodic")),
                status="merged",
                source_count=int(record.get("source_count", 1)),
                consolidated_from=list(record.get("consolidated_from", [])),
                created_at=record.get("created_at"),
            )

            self.collection.update(ids=[survivor["id"]], metadatas=[survivor_meta])
            self.collection.update(ids=[record["id"]], metadatas=[record_meta])
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
                meta = self._build_metadata(
                    record_id=record_id,
                    triplet=record,
                    now=now,
                    last_accessed_at=record.get("last_accessed_at"),
                    reinforcement_count=int(record.get("reinforcement_count", 0)),
                    memory_type=str(record.get("memory_type", "episodic")),
                    status="pruned",
                    source_count=int(record.get("source_count", 1)),
                    consolidated_from=list(record.get("consolidated_from", [])),
                    created_at=record.get("created_at"),
                )
                self.collection.update(ids=[record_id], metadatas=[meta])
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
            new_cf = list(existing.get("consolidated_from", []))
            new_cf.extend([r["id"] for r in source_records])
            meta = self._build_metadata(
                record_id=existing["id"],
                triplet=existing,
                now=now,
                last_accessed_at=existing.get("last_accessed_at"),
                reinforcement_count=int(existing.get("reinforcement_count", 0)),
                memory_type="semantic",
                status=_ACTIVE,
                source_count=int(existing.get("source_count", 1)) + len(source_records),
                consolidated_from=new_cf,
                created_at=existing.get("created_at"),
            )
            emb = get_embedding(self._triplet_to_text(existing))
            self.collection.update(
                ids=[existing["id"]], embeddings=[emb],
                metadatas=[meta], documents=[self._triplet_to_text(existing)],
            )
            for record in source_records:
                if record["id"] == existing["id"]:
                    continue
                rec_meta = self._build_metadata(
                    record_id=record["id"],
                    triplet=record,
                    now=now,
                    last_accessed_at=record.get("last_accessed_at"),
                    reinforcement_count=int(record.get("reinforcement_count", 0)),
                    memory_type=str(record.get("memory_type", "episodic")),
                    status="merged",
                    source_count=int(record.get("source_count", 1)),
                    consolidated_from=list(record.get("consolidated_from", [])),
                    created_at=record.get("created_at"),
                )
                emb_r = get_embedding(self._triplet_to_text(record))
                self.collection.update(
                    ids=[record["id"]], embeddings=[emb_r],
                    metadatas=[rec_meta], documents=[self._triplet_to_text(record)],
                )
            return self.find_semantic_memory(
                survivor["subject"], survivor["predicate"], survivor["object"]
            )

        # No existing semantic — create one
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
                rec_meta = self._build_metadata(
                    record_id=record["id"],
                    triplet=record,
                    now=now,
                    last_accessed_at=record.get("last_accessed_at"),
                    reinforcement_count=int(record.get("reinforcement_count", 0)),
                    memory_type=str(record.get("memory_type", "episodic")),
                    status="merged",
                    source_count=int(record.get("source_count", 1)),
                    consolidated_from=list(record.get("consolidated_from", [])),
                    created_at=record.get("created_at"),
                )
                emb_r = get_embedding(self._triplet_to_text(record))
                self.collection.update(
                    ids=[record["id"]], embeddings=[emb_r],
                    metadatas=[rec_meta], documents=[self._triplet_to_text(record)],
                )

        return self.find_semantic_memory(
            survivor["subject"], survivor["predicate"], survivor["object"]
        )

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def all(self) -> List[Dict]:
        return [r for r in self._fetch_all_as_dicts() if r.get("status") == _ACTIVE]

    def all_records(self) -> List[Dict]:
        return self._fetch_all_as_dicts()

    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        for r in self.all():
            if r["subject"] == subject and r["predicate"] == predicate and r["object"] == obj:
                return r
        return None

    def find_by_subject_predicate(self, subject: str, predicate: str) -> List[Dict]:
        return [r for r in self.all() if r["subject"] == subject and r["predicate"] == predicate]

    def find_semantic_memory(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        for r in self.all():
            if (
                r.get("memory_type") == "semantic"
                and r["subject"] == subject
                and r["predicate"] == predicate
                and r["object"] == obj
            ):
                return r
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_metadata(
        self,
        record_id: str,
        triplet: Dict,
        now: float,
        last_accessed_at: Optional[float],
        reinforcement_count: int,
        memory_type: str,
        status: str,
        source_count: int,
        consolidated_from: List,
        created_at: Optional[float] = None,
    ) -> Dict:
        """
        Build a Chroma-safe metadata dict.
        NEVER includes 'embedding' — that lives only in the vector store.
        All values are primitive types (str, int, float, bool).
        """
        return {
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
            "created_at": float(created_at if created_at is not None else now),
            "updated_at": float(now),
            "last_accessed_at": float(last_accessed_at) if last_accessed_at is not None else -1.0,
            "reinforcement_count": int(reinforcement_count),
            "memory_type": memory_type,
            "status": status,
            "source_count": int(source_count),
            "consolidated_from": json.dumps(consolidated_from),
        }

    def _meta_to_dict(self, meta: Dict) -> Dict:
        """Convert raw Chroma metadata back to a SynapseMem record dict."""
        last_accessed = meta.get("last_accessed_at", -1.0)
        return {
            "id": meta.get("id"),
            "user_id": meta.get("user_id"),
            "agent_id": meta.get("agent_id"),
            "session_id": meta.get("session_id"),
            "subject": meta.get("subject"),
            "predicate": meta.get("predicate"),
            "object": meta.get("object"),
            "topic": meta.get("topic"),
            "priority": meta.get("priority"),
            "source_text": meta.get("source_text"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "last_accessed_at": None if last_accessed == -1.0 else last_accessed,
            "reinforcement_count": meta.get("reinforcement_count", 0),
            "memory_type": meta.get("memory_type", "episodic"),
            "status": meta.get("status", _ACTIVE),
            "source_count": meta.get("source_count", 1),
            "consolidated_from": json.loads(meta.get("consolidated_from", "[]")),
            # Vectors live in Chroma's store, not here. Retriever recomputes when needed.
            "embedding": [],
        }

    def _fetch_by_id(self, record_id: str) -> Optional[Dict]:
        result = self.collection.get(ids=[record_id], include=["metadatas"])
        if not result["ids"]:
            return None
        return self._meta_to_dict(result["metadatas"][0])

    def _get_all_records_raw(self) -> List[tuple]:
        """Returns list of (id, raw_metadata_dict) — no conversion."""
        result = self.collection.get(include=["metadatas"])
        return list(zip(result["ids"], result["metadatas"]))

    def _fetch_all_as_dicts(self) -> List[Dict]:
        result = self.collection.get(include=["metadatas"])
        return [self._meta_to_dict(meta) for meta in result["metadatas"]]