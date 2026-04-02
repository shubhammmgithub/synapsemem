"""Multi-agent shared memory for SynapseMem Phase 3."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from ..utils.embeddings import get_embedding

ConflictStrategy = Literal["last_write_wins", "anchor_weighted", "no_overwrite"]

_ACTIVE = "active"


class SharedMemoryStore:
    """
    Shared knowledge graph namespace for a workspace of agents.
    Backed by SQLite. Supports ':memory:' for testing.
    """

    def __init__(
        self,
        workspace_id: str,
        db_path: str = "synapsemem.db",
        conflict_strategy: ConflictStrategy = "last_write_wins",
    ) -> None:
        self.workspace_id = workspace_id
        self.db_path = ":memory:" if db_path == ":memory:" else str(Path(db_path))
        self.conflict_strategy = conflict_strategy
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._init_table()

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        """
        Returns a SQLite connection.
        For :memory: databases, reuses a single persistent connection so
        tables created in _init_table remain visible across all queries.
        """
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # Schema                                                               #
    # ------------------------------------------------------------------ #

    def _init_table(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_memories (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                topic TEXT NOT NULL DEFAULT 'general',
                priority INTEGER NOT NULL DEFAULT 3,
                source_text TEXT,
                embedding TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL,
                reinforcement_count INTEGER NOT NULL DEFAULT 0,
                memory_type TEXT NOT NULL DEFAULT 'shared',
                status TEXT NOT NULL DEFAULT 'active',
                source_count INTEGER NOT NULL DEFAULT 1,
                trust_score REAL NOT NULL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shared_workspace
            ON shared_memories (workspace_id, status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shared_subpred
            ON shared_memories (workspace_id, subject, predicate, status)
        """)
        conn.commit()

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def write_fact(self, triplet: Dict, agent_id: str) -> Dict:
        existing = self._find_by_subject_predicate(
            triplet["subject"], triplet["predicate"]
        )
        if not existing:
            record = self._insert(triplet, agent_id)
            return {"action": "ADD", "record": record}
        return self._resolve_conflict(
            incoming=triplet, existing=existing, agent_id=agent_id
        )

    def write_facts(self, triplets: List[Dict], agent_id: str) -> List[Dict]:
        return [self.write_fact(t, agent_id) for t in triplets]

    def delete_fact(
        self, subject: str, predicate: str, obj: str, agent_id: str
    ) -> bool:
        conn = self._connect()
        row = conn.execute("""
            SELECT id FROM shared_memories
            WHERE workspace_id = ? AND subject = ? AND predicate = ?
              AND object = ? AND status = 'active'
            LIMIT 1
        """, (self.workspace_id, subject, predicate, obj)).fetchone()

        if not row:
            return False

        conn.execute("""
            UPDATE shared_memories SET status = 'deleted', updated_at = ?
            WHERE id = ?
        """, (time.time(), row["id"]))
        conn.commit()
        return True

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def read_facts(self, topic: Optional[str] = None) -> List[Dict]:
        conn = self._connect()
        if topic:
            rows = conn.execute("""
                SELECT * FROM shared_memories
                WHERE workspace_id = ? AND status = 'active' AND topic = ?
                ORDER BY priority DESC, created_at ASC
            """, (self.workspace_id, topic)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM shared_memories
                WHERE workspace_id = ? AND status = 'active'
                ORDER BY priority DESC, created_at ASC
            """, (self.workspace_id,)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def facts_by_agent(self, agent_id: str) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM shared_memories
            WHERE workspace_id = ? AND agent_id = ? AND status = 'active'
            ORDER BY created_at ASC
        """, (self.workspace_id, agent_id)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def workspace_stats(self) -> Dict:
        conn = self._connect()
        total = conn.execute(
            "SELECT COUNT(*) FROM shared_memories WHERE workspace_id = ?",
            (self.workspace_id,)
        ).fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM shared_memories WHERE workspace_id = ? AND status = 'active'",
            (self.workspace_id,)
        ).fetchone()[0]
        agents = conn.execute(
            "SELECT COUNT(DISTINCT agent_id) FROM shared_memories WHERE workspace_id = ? AND status = 'active'",
            (self.workspace_id,)
        ).fetchone()[0]
        return {
            "workspace_id": self.workspace_id,
            "total_records": total,
            "active_records": active,
            "contributing_agents": agents,
            "conflict_strategy": self.conflict_strategy,
        }

    # ------------------------------------------------------------------ #
    # Conflict resolution                                                  #
    # ------------------------------------------------------------------ #

    def _resolve_conflict(self, incoming: Dict, existing: Dict, agent_id: str) -> Dict:
        strategy = self.conflict_strategy

        if strategy == "no_overwrite":
            return {"action": "NOOP", "record": existing, "reason": "no_overwrite_policy"}

        if strategy == "last_write_wins":
            record = self._update(existing["id"], incoming, agent_id)
            return {"action": "UPDATE", "record": record, "reason": "last_write_wins"}

        if strategy == "anchor_weighted":
            incoming_priority = int(incoming.get("priority", 3))
            existing_priority = int(existing.get("priority", 3))
            trust = float(existing.get("trust_score", 1.0))
            effective_existing = existing_priority * trust

            if incoming_priority > effective_existing:
                record = self._update(existing["id"], incoming, agent_id)
                return {"action": "UPDATE", "record": record, "reason": "higher_priority_wins"}
            else:
                self._reinforce(existing["id"])
                return {"action": "NOOP", "record": existing, "reason": "existing_anchor_wins"}

        return {"action": "NOOP", "record": existing}

    # ------------------------------------------------------------------ #
    # Internal DB helpers                                                  #
    # ------------------------------------------------------------------ #

    def _insert(self, triplet: Dict, agent_id: str) -> Dict:
        now = time.time()
        text = f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"
        embedding = get_embedding(text)
        record_id = str(uuid.uuid4())

        conn = self._connect()
        conn.execute("""
            INSERT INTO shared_memories (
                id, workspace_id, agent_id,
                subject, predicate, object,
                topic, priority, source_text, embedding,
                created_at, updated_at, last_accessed_at,
                reinforcement_count, memory_type, status,
                source_count, trust_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, self.workspace_id, agent_id,
            str(triplet["subject"]), str(triplet["predicate"]), str(triplet["object"]),
            str(triplet.get("topic", "general")), int(triplet.get("priority", 3)),
            str(triplet.get("source_text", "")), json.dumps(embedding),
            now, now, None,
            0, "shared", _ACTIVE, 1, 1.0,
        ))
        conn.commit()
        return self._fetch_by_id(record_id)

    def _update(self, record_id: str, new_triplet: Dict, agent_id: str) -> Dict:
        now = time.time()
        text = f"{new_triplet['subject']} {new_triplet['predicate']} {new_triplet['object']}"
        embedding = get_embedding(text)

        conn = self._connect()
        conn.execute("""
            UPDATE shared_memories
            SET subject = ?, predicate = ?, object = ?,
                topic = ?, priority = ?, source_text = ?,
                embedding = ?, updated_at = ?, agent_id = ?,
                source_count = source_count + 1
            WHERE id = ?
        """, (
            str(new_triplet["subject"]), str(new_triplet["predicate"]), str(new_triplet["object"]),
            str(new_triplet.get("topic", "general")), int(new_triplet.get("priority", 3)),
            str(new_triplet.get("source_text", "")),
            json.dumps(embedding), now, agent_id, record_id,
        ))
        conn.commit()
        return self._fetch_by_id(record_id)

    def _reinforce(self, record_id: str) -> None:
        now = time.time()
        conn = self._connect()
        conn.execute("""
            UPDATE shared_memories
            SET reinforcement_count = reinforcement_count + 1,
                last_accessed_at = ?, updated_at = ?
            WHERE id = ?
        """, (now, now, record_id))
        conn.commit()

    def _find_by_subject_predicate(self, subject: str, predicate: str) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute("""
            SELECT * FROM shared_memories
            WHERE workspace_id = ? AND subject = ? AND predicate = ?
              AND status = 'active'
            ORDER BY priority DESC, created_at DESC
            LIMIT 1
        """, (self.workspace_id, subject, predicate)).fetchone()
        return self._row_to_dict(row) if row else None

    def _fetch_by_id(self, record_id: str) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM shared_memories WHERE id = ?", (record_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "agent_id": row["agent_id"],
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
            "trust_score": row["trust_score"],
        }