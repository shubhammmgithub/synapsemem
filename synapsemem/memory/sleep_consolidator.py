"""Offline sleep consolidation for stored memories."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class SleepConsolidator:
    """
    Phase 1 sleep consolidation:
    - merge exact duplicate memories
    - prune weak / stale memories
    - support dry-run reporting

    This is intentionally separate from the online IngestConsolidator.
    """

    def __init__(
        self,
        min_age_seconds: int = 30 * 60,
        prune_age_seconds: int = 7 * 24 * 60 * 60,
        min_priority_to_keep: int = 2,
        protect_reinforced: bool = True,
    ) -> None:
        self.min_age_seconds = int(min_age_seconds)
        self.prune_age_seconds = int(prune_age_seconds)
        self.min_priority_to_keep = int(min_priority_to_keep)
        self.protect_reinforced = bool(protect_reinforced)

    def run(
        self,
        storage,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        now = time.time()
        records = storage.all()

        candidates = [
            record
            for record in records
            if self._is_old_enough(record, now=now)
        ]

        duplicate_actions = self._plan_duplicate_merges(candidates)
        duplicate_ids = {action["record_id"] for action in duplicate_actions}

        prune_actions = self._plan_pruning(
            candidates,
            now=now,
            skip_ids=duplicate_ids,
        )

        prune_ids = {action["record_id"] for action in prune_actions}

        keep_ids = {
            record["id"]
            for record in candidates
            if record["id"] not in duplicate_ids and record["id"] not in prune_ids
        }

        report = {
            "dry_run": dry_run,
            "scanned": len(records),
            "eligible": len(candidates),
            "merged": len(duplicate_actions),
            "pruned": len(prune_actions),
            "kept": len(keep_ids),
            "merge_actions": duplicate_actions,
            "prune_actions": prune_actions,
        }

        if not dry_run:
            if duplicate_actions:
                storage.merge_duplicates(duplicate_actions)

            if prune_actions:
                storage.prune_memories(prune_actions)

        return report

    def _plan_duplicate_merges(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        for record in records:
            key = (
                str(record["subject"]).strip().lower(),
                str(record["predicate"]).strip().lower(),
                str(record["object"]).strip().lower(),
            )
            grouped[key].append(record)

        actions: List[Dict[str, Any]] = []

        for key, cluster in grouped.items():
            if len(cluster) < 2:
                continue

            survivor = self._select_survivor(cluster)

            for record in cluster:
                if record["id"] == survivor["id"]:
                    continue

                actions.append({
                    "action": "MERGE",
                    "record_id": record["id"],
                    "survivor_id": survivor["id"],
                    "subject": key[0],
                    "predicate": key[1],
                    "object": key[2],
                    "reason": "exact_duplicate_memory",
                })

        return actions

    def _plan_pruning(
        self,
        records: List[Dict[str, Any]],
        now: float,
        skip_ids: set[str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        for record in records:
            if record["id"] in skip_ids:
                continue

            if self._should_prune(record, now=now):
                actions.append({
                    "action": "PRUNE",
                    "record_id": record["id"],
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "reason": "stale_low_priority_unreinforced",
                })

        return actions

    def _select_survivor(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        def score(record: Dict[str, Any]) -> Tuple[int, int, float]:
            return (
                int(record.get("priority", 3)),
                int(record.get("reinforcement_count", 0)),
                float(record.get("last_accessed_at") or record.get("created_at") or 0.0),
            )

        return max(cluster, key=score)

    def _should_prune(self, record: Dict[str, Any], now: float) -> bool:
        priority = int(record.get("priority", 3))
        reinforcement_count = int(record.get("reinforcement_count", 0))
        created_at = float(record.get("created_at") or 0.0)
        last_accessed_at = record.get("last_accessed_at")

        age = max(0.0, now - created_at)

        if age < self.prune_age_seconds:
            return False

        if priority > self.min_priority_to_keep:
            return False

        if self.protect_reinforced and reinforcement_count > 0:
            return False

        if last_accessed_at is not None and self.protect_reinforced:
            return False

        return True

    def _is_old_enough(self, record: Dict[str, Any], now: float) -> bool:
        created_at = float(record.get("created_at") or 0.0)
        age = max(0.0, now - created_at)
        return age >= self.min_age_seconds