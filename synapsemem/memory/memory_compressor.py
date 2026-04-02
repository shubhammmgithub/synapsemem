"""Memory compression for SynapseMem Phase 3.

Runs as a final pass after the sleep consolidator.
Finds clusters of semantically similar episodic memories and
compresses them into a single summarised semantic fact using the LLM.

This is different from the sleep consolidator's promotion step:
  - Sleep promotion:   exact (subject, predicate, object) matches → semantic
  - Compression:       semantically similar but NOT exact → LLM summarisation

Usage:
    compressor = MemoryCompressor(llm=my_llm_fn, similarity_threshold=0.85)
    report = compressor.run(storage=memory.storage, dry_run=False)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.embeddings import get_embedding
from ..utils.scorer import cosine_sim


class MemoryCompressor:
    """
    Semantic compression pass for long-running memory stores.

    Algorithm:
        1. Fetch all active episodic memories
        2. Cluster by cosine similarity of their embeddings
        3. For clusters above min_cluster_size:
             a. Build a summary prompt
             b. Call LLM to produce a compressed fact
             c. Store the summary as a new semantic memory
             d. Mark source episodic memories as 'merged'
        4. Return a compression report

    If no LLM is provided, step 3b is skipped and we fall back to
    using the highest-priority memory in the cluster as the summary
    (same as sleep consolidator promotion — useful for testing).
    """

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        max_summary_tokens: int = 60,
        protect_semantic: bool = True,
    ) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_summary_tokens = max_summary_tokens
        self.protect_semantic = protect_semantic

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, storage, dry_run: bool = True) -> Dict[str, Any]:
        """
        Run compression pass over the given storage backend.
        Returns a report dict compatible with sleep consolidator reports.
        """
        records = storage.all()
        episodic = [
            r for r in records
            if r.get("memory_type", "episodic") == "episodic"
            and r.get("status", "active") == "active"
        ]

        clusters = self._cluster_by_similarity(episodic)
        eligible_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        compression_actions = []
        for cluster in eligible_clusters:
            action = self._plan_compression(cluster)
            if action:
                compression_actions.append(action)

        report = {
            "dry_run": dry_run,
            "episodic_scanned": len(episodic),
            "clusters_found": len(clusters),
            "eligible_clusters": len(eligible_clusters),
            "compressed": len(compression_actions),
            "compression_actions": self._serialize_actions(compression_actions),
        }

        if not dry_run:
            for action in compression_actions:
                self._apply_compression(storage, action)

        return report

    # ------------------------------------------------------------------ #
    # Clustering                                                           #
    # ------------------------------------------------------------------ #

    def _cluster_by_similarity(self, records: List[Dict]) -> List[List[Dict]]:
        """
        Greedy single-linkage clustering based on embedding cosine similarity.
        O(n²) — acceptable for memory stores up to a few thousand records.
        """
        if not records:
            return []

        assigned = [False] * len(records)
        clusters: List[List[Dict]] = []

        for i, record in enumerate(records):
            if assigned[i]:
                continue

            cluster = [record]
            assigned[i] = True
            emb_i = record.get("embedding", [])

            for j in range(i + 1, len(records)):
                if assigned[j]:
                    continue
                emb_j = records[j].get("embedding", [])
                if not emb_i or not emb_j:
                    continue
                sim = cosine_sim(emb_i, emb_j)
                if sim >= self.similarity_threshold:
                    cluster.append(records[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------------ #
    # Compression planning                                                 #
    # ------------------------------------------------------------------ #

    def _plan_compression(self, cluster: List[Dict]) -> Optional[Dict]:
        if len(cluster) < self.min_cluster_size:
            return None

        # Pick the highest-priority record as the representative
        representative = max(cluster, key=lambda r: (int(r.get("priority", 3)), int(r.get("reinforcement_count", 0))))

        facts_text = "\n".join(
            f"- {r['subject']} {r['predicate']} {r['object']}"
            for r in cluster
        )

        if self.llm:
            summary = self._call_llm_for_summary(facts_text, representative)
        else:
            # Fallback: use representative's text as-is
            summary = {
                "subject": representative["subject"],
                "predicate": representative["predicate"],
                "object": representative["object"],
            }

        return {
            "action": "COMPRESS",
            "representative": representative,
            "source_records": cluster,
            "summary": summary,
            "cluster_size": len(cluster),
        }

    def _call_llm_for_summary(self, facts_text: str, representative: Dict) -> Dict:
        """
        Ask the LLM to produce a single compressed fact from a cluster.
        Returns a triplet dict: {subject, predicate, object}.
        Falls back to representative on any failure.
        """
        prompt = (
            f"The following facts are semantically related:\n{facts_text}\n\n"
            f"Write a single concise fact that captures the core meaning of all of them.\n"
            f"Format: SUBJECT | PREDICATE | OBJECT\n"
            f"Use at most {self.max_summary_tokens} words total.\n"
            f"Reply with only the formatted fact, nothing else."
        )

        try:
            response = self.llm(prompt).strip()
            parts = [p.strip() for p in response.split("|")]
            if len(parts) == 3:
                return {
                    "subject": parts[0].lower(),
                    "predicate": parts[1].lower(),
                    "object": parts[2],
                }
        except Exception:
            pass

        # Fallback
        return {
            "subject": representative["subject"],
            "predicate": representative["predicate"],
            "object": representative["object"],
        }

    # ------------------------------------------------------------------ #
    # Apply                                                                #
    # ------------------------------------------------------------------ #

    def _apply_compression(self, storage, action: Dict) -> None:
        """
        Store the compressed summary as a new semantic memory,
        then mark all source records as merged.
        """
        cluster = action["source_records"]
        summary = action["summary"]
        representative = action["representative"]

        compressed_triplet = {
            "subject": summary["subject"],
            "predicate": summary["predicate"],
            "object": summary["object"],
            "topic": representative.get("topic", "general"),
            "priority": max(int(r.get("priority", 3)) for r in cluster),
            "source_text": f"[compressed from {len(cluster)} memories]",
            "memory_type": "semantic",
            "status": "active",
            "source_count": sum(int(r.get("source_count", 1)) for r in cluster),
            "consolidated_from": [r["id"] for r in cluster],
        }

        storage.add_triplets([compressed_triplet])

        # Mark source memories as merged
        now = time.time()
        merge_actions = [
            {"record_id": r["id"], "survivor_id": cluster[0]["id"]}
            for r in cluster[1:]
        ]
        if merge_actions:
            storage.merge_duplicates(merge_actions)

        # Also prune the representative (it's been superseded by compressed)
        storage.prune_memories([{"record_id": representative["id"]}])

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def _serialize_actions(self, actions: List[Dict]) -> List[Dict]:
        return [
            {
                "action": a["action"],
                "cluster_size": a["cluster_size"],
                "summary": a["summary"],
                "source_record_ids": [r["id"] for r in a["source_records"]],
            }
            for a in actions
        ]