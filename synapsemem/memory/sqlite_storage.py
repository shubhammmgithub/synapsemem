from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.embeddings import get_embedding


class SQLiteMemoryStorage:
    def __init__(
        self,
        db_path: str = "synapsemem.db",
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
    ) -> None:
        self.db_path = str(Path(db_path))
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    source_text TEXT,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL,
                    reinforcement_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "reinforcement_count" not in existing_columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0"
                )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories (user_id, agent_id, session_id)
                """
            )
            conn.commit()

    def add_triplets(self, triplets: List[Dict]) -> None:
        now = time.time()

        with self._connect() as conn:
            for triplet in triplets:
                text_repr = self._triplet_to_text(triplet)
                embedding = get_embedding(text_repr)

                conn.execute(
                    """
                    INSERT INTO memories (
                        id, user_id, agent_id, session_id,
                        subject, predicate, object, topic, priority,
                        source_text, embedding, created_at, last_accessed_at,
                        reinforcement_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        self.user_id,
                        self.agent_id,
                        self.session_id,
                        triplet["subject"],
                        triplet["predicate"],
                        triplet["object"],
                        triplet.get("topic", "general"),
                        int(triplet.get("priority", 3)),
                        triplet.get("source_text", ""),
                        json.dumps(embedding),
                        now,
                        None,
                        0,
                    ),
                )
            conn.commit()

    def all(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                ORDER BY created_at ASC
                """,
                (self.user_id, self.agent_id, self.session_id),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND subject = ? AND predicate = ? AND object = ?
                LIMIT 1
                """,
                (self.user_id, self.agent_id, self.session_id, subject, predicate, obj),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_by_subject_predicate(self, subject: str, predicate: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND subject = ? AND predicate = ?
                ORDER BY created_at ASC
                """,
                (self.user_id, self.agent_id, self.session_id, subject, predicate),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET subject = ?,
                    predicate = ?,
                    object = ?,
                    topic = ?,
                    priority = ?,
                    source_text = ?,
                    embedding = ?
                WHERE id = ? AND user_id = ? AND agent_id = ? AND session_id = ?
                """,
                (
                    new_triplet["subject"],
                    new_triplet["predicate"],
                    new_triplet["object"],
                    new_triplet.get("topic", "general"),
                    int(new_triplet.get("priority", 3)),
                    new_triplet.get("source_text", ""),
                    json.dumps(get_embedding(self._triplet_to_text(new_triplet))),
                    old_record_id,
                    self.user_id,
                    self.agent_id,
                    self.session_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def reinforce(self, record_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?,
                    reinforcement_count = reinforcement_count + 1
                WHERE id = ? AND user_id = ? AND agent_id = ? AND session_id = ?
                """,
                (time.time(), record_id, self.user_id, self.agent_id, self.session_id),
            )
            conn.commit()

    def update_last_accessed(self, record_id: str) -> None:
        self.reinforce(record_id)

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE lower(topic) = ?
                  AND user_id = ? AND agent_id = ? AND session_id = ?
                """,
                (topic, self.user_id, self.agent_id, self.session_id),
            )
            conn.commit()
            return cursor.rowcount

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        conditions = ["user_id = ?", "agent_id = ?", "session_id = ?"]
        params: List[object] = [self.user_id, self.agent_id, self.session_id]

        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate)
        if obj is not None:
            conditions.append("object = ?")
            params.append(obj)

        where_clause = " AND ".join(conditions)

        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM memories WHERE {where_clause}", params)
            conn.commit()
            return cursor.rowcount

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        duplicate_ids = [action["record_id"] for action in merge_actions]
        if not duplicate_ids:
            return 0

        placeholders = ", ".join("?" for _ in duplicate_ids)
        params: List[object] = [
            self.user_id,
            self.agent_id,
            self.session_id,
            *duplicate_ids,
        ]

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                DELETE FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND id IN ({placeholders})
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount

    def prune_memories(self, prune_actions: List[Dict]) -> int:
        prune_ids = [action["record_id"] for action in prune_actions]
        if not prune_ids:
            return 0

        placeholders = ", ".join("?" for _ in prune_ids)
        params: List[object] = [
            self.user_id,
            self.agent_id,
            self.session_id,
            *prune_ids,
        ]

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                DELETE FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND id IN ({placeholders})
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                """,
                (self.user_id, self.agent_id, self.session_id),
            )
            conn.commit()

    def _triplet_to_text(self, triplet: Dict) -> str:
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"

    def _row_to_record(self, row: sqlite3.Row) -> Dict:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "agent_id": row["agent_id"],
            "session_id": row["session_id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "topic": row["topic"],
            "priority": row["priority"],
            "source_text": row["source_text"],
            "embedding": json.loads(row["embedding"]),
            "created_at": row["created_at"],
            "last_accessed_at": row["last_accessed_at"],
            "reinforcement_count": row["reinforcement_count"],
        }