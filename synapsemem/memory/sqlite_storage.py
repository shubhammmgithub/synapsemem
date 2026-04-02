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
                    updated_at REAL,
                    last_accessed_at REAL,
                    reinforcement_count INTEGER NOT NULL DEFAULT 0,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    status TEXT NOT NULL DEFAULT 'active',
                    source_count INTEGER NOT NULL DEFAULT 1,
                    consolidated_from TEXT NOT NULL DEFAULT '[]'
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }

            migrations = [
                ("reinforcement_count", "ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0"),
                ("updated_at", "ALTER TABLE memories ADD COLUMN updated_at REAL"),
                ("memory_type", "ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'episodic'"),
                ("status", "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"),
                ("source_count", "ALTER TABLE memories ADD COLUMN source_count INTEGER NOT NULL DEFAULT 1"),
                ("consolidated_from", "ALTER TABLE memories ADD COLUMN consolidated_from TEXT NOT NULL DEFAULT '[]'"),
            ]

            for column_name, sql in migrations:
                if column_name not in existing_columns:
                    conn.execute(sql)

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories (user_id, agent_id, session_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_status_type
                ON memories (status, memory_type)
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
                        source_text, embedding, created_at, updated_at,
                        last_accessed_at, reinforcement_count,
                        memory_type, status, source_count, consolidated_from
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        now,
                        None,
                        0,
                        triplet.get("memory_type", "episodic"),
                        triplet.get("status", "active"),
                        int(triplet.get("source_count", 1)),
                        json.dumps(triplet.get("consolidated_from", [])),
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
                  AND status = 'active'
                ORDER BY created_at ASC
                """,
                (self.user_id, self.agent_id, self.session_id),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def all_records(self) -> List[Dict]:
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
                  AND status = 'active'
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
                  AND status = 'active'
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
                    embedding = ?,
                    updated_at = ?
                WHERE id = ? AND user_id = ? AND agent_id = ? AND session_id = ?
                  AND status = 'active'
                """,
                (
                    new_triplet["subject"],
                    new_triplet["predicate"],
                    new_triplet["object"],
                    new_triplet.get("topic", "general"),
                    int(new_triplet.get("priority", 3)),
                    new_triplet.get("source_text", ""),
                    json.dumps(get_embedding(self._triplet_to_text(new_triplet))),
                    time.time(),
                    old_record_id,
                    self.user_id,
                    self.agent_id,
                    self.session_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def reinforce(self, record_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?,
                    reinforcement_count = reinforcement_count + 1,
                    updated_at = ?
                WHERE id = ? AND user_id = ? AND agent_id = ? AND session_id = ?
                  AND status = 'active'
                """,
                (now, now, record_id, self.user_id, self.agent_id, self.session_id),
            )
            conn.commit()

    def update_last_accessed(self, record_id: str) -> None:
        self.reinforce(record_id)

    def delete_topic(self, topic: str) -> int:
        topic = topic.strip().lower()
        now = time.time()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET status = 'pruned',
                    updated_at = ?
                WHERE lower(topic) = ?
                  AND user_id = ? AND agent_id = ? AND session_id = ?
                  AND status = 'active'
                """,
                (now, topic, self.user_id, self.agent_id, self.session_id),
            )
            conn.commit()
            return cursor.rowcount

    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        conditions = [
            "user_id = ?",
            "agent_id = ?",
            "session_id = ?",
            "status = 'active'",
        ]
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
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET status = 'pruned',
                    updated_at = ?
                WHERE {where_clause}
                """,
                [time.time(), *params],
            )
            conn.commit()
            return cursor.rowcount

    def merge_duplicates(self, merge_actions: List[Dict]) -> int:
        if not merge_actions:
            return 0

        merged_count = 0
        now = time.time()

        with self._connect() as conn:
            for action in merge_actions:
                record = self._fetch_by_id(conn, action["record_id"])
                survivor = self._fetch_by_id(conn, action["survivor_id"])

                if not record or not survivor:
                    continue
                if record["status"] != "active" or survivor["status"] != "active":
                    continue

                survivor_source_count = int(survivor["source_count"]) + int(record["source_count"])
                survivor_consolidated_from = json.loads(survivor["consolidated_from"] or "[]")
                survivor_consolidated_from.append(record["id"])

                conn.execute(
                    """
                    UPDATE memories
                    SET source_count = ?,
                        consolidated_from = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        survivor_source_count,
                        json.dumps(survivor_consolidated_from),
                        now,
                        survivor["id"],
                    ),
                )

                conn.execute(
                    """
                    UPDATE memories
                    SET status = 'merged',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, record["id"]),
                )
                merged_count += 1

            conn.commit()

        return merged_count

    def prune_memories(self, prune_actions: List[Dict]) -> int:
        prune_ids = [action["record_id"] for action in prune_actions]
        if not prune_ids:
            return 0

        placeholders = ", ".join("?" for _ in prune_ids)
        params: List[object] = [
            time.time(),
            self.user_id,
            self.agent_id,
            self.session_id,
            *prune_ids,
        ]

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET status = 'pruned',
                    updated_at = ?
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND status = 'active'
                  AND id IN ({placeholders})
                """,
                params,
            )
            conn.commit()
            return cursor.rowcount

    def find_semantic_memory(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM memories
                WHERE user_id = ? AND agent_id = ? AND session_id = ?
                  AND status = 'active'
                  AND memory_type = 'semantic'
                  AND subject = ? AND predicate = ? AND object = ?
                LIMIT 1
                """,
                (self.user_id, self.agent_id, self.session_id, subject, predicate, obj),
            ).fetchone()
        return self._row_to_record(row) if row else None

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

        with self._connect() as conn:
            if existing is not None:
                consolidated_from = list(existing.get("consolidated_from", []))
                consolidated_from.extend([r["id"] for r in source_records])

                conn.execute(
                    """
                    UPDATE memories
                    SET source_count = ?,
                        consolidated_from = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        int(existing.get("source_count", 1)) + len(source_records),
                        json.dumps(consolidated_from),
                        now,
                        existing["id"],
                    ),
                )

                for record in source_records:
                    if record["id"] == existing["id"]:
                        continue
                    conn.execute(
                        """
                        UPDATE memories
                        SET status = 'merged',
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (now, record["id"]),
                    )

                conn.commit()
                return self.find_semantic_memory(
                    survivor["subject"],
                    survivor["predicate"],
                    survivor["object"],
                )

            self.add_triplets([
                {
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
            ])

            for record in source_records:
                conn.execute(
                    """
                    UPDATE memories
                    SET status = 'merged',
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (now, record["id"]),
                )

            conn.commit()

        return self.find_semantic_memory(
            survivor["subject"],
            survivor["predicate"],
            survivor["object"],
        )

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

    def _fetch_by_id(self, conn: sqlite3.Connection, record_id: str) -> Optional[sqlite3.Row]:
        return conn.execute(
            """
            SELECT *
            FROM memories
            WHERE id = ? AND user_id = ? AND agent_id = ? AND session_id = ?
            LIMIT 1
            """,
            (record_id, self.user_id, self.agent_id, self.session_id),
        ).fetchone()

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
            "updated_at": row["updated_at"],
            "last_accessed_at": row["last_accessed_at"],
            "reinforcement_count": row["reinforcement_count"],
            "memory_type": row["memory_type"],
            "status": row["status"],
            "source_count": row["source_count"],
            "consolidated_from": json.loads(row["consolidated_from"] or "[]"),
        }