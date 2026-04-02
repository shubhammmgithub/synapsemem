"""Celery tasks — async ingest pipeline for SynapseMem."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .celery_app import celery_app
from ..memory.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

_classifier = IntentClassifier()


def _build_memory(
    storage_backend: str,
    user_id: str,
    agent_id: str,
    session_id: str,
    storage_config: Optional[Dict] = None,
):
    """
    Build a SynapseMemory instance inside the worker process.
    Only passes kwargs that the current manager.py __init__ accepts.
    Detects available params at import time to stay compatible with
    both the Phase 2 manager and the updated Phase 3 manager.
    """
    from ..manager import SynapseMemory
    import inspect

    cfg = storage_config or {}

    # Base kwargs — always supported
    kwargs = dict(
        storage_backend=storage_backend,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        sqlite_db_path=cfg.get("sqlite_db_path", "synapsemem.db"),
    )

    # Only add Phase 3 params if manager.__init__ accepts them
    sig = inspect.signature(SynapseMemory.__init__)
    params = sig.parameters

    if "qdrant_url" in params:
        kwargs["qdrant_url"] = cfg.get("qdrant_url", "http://localhost:6333")
        kwargs["qdrant_api_key"] = cfg.get("qdrant_api_key")

    if "chroma_persist_directory" in params:
        kwargs["chroma_persist_directory"] = cfg.get(
            "chroma_persist_directory", "./chroma_db"
        )
        kwargs["chroma_host"] = cfg.get("chroma_host")
        kwargs["chroma_port"] = int(cfg.get("chroma_port", 8000))

    return SynapseMemory(**kwargs)


# ── Task 1: single text ingest ──────────────────────────────────────────

@celery_app.task(
    name="synapsemem.async_pipeline.tasks.ingest_text_async",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    queue="ingest",
)
def ingest_text_async(
    self,
    text: str,
    user_id: str = "default_user",
    agent_id: str = "default_agent",
    session_id: str = "default_session",
    storage_backend: str = "sqlite",
    storage_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    start = time.time()

    try:
        intent = _classifier.classify(text)

        if _classifier.should_skip(intent):
            return {
                "status": "skipped",
                "reason": "chitchat",
                "intent": intent,
                "text": text[:80],
                "elapsed_ms": round((time.time() - start) * 1000, 1),
            }

        memory = _build_memory(
            storage_backend, user_id, agent_id, session_id, storage_config
        )

        raw_triplets = memory.extractor.extract(text)
        enriched_triplets = _classifier.enrich_triplets(raw_triplets, intent)

        decisions = memory.consolidator.decide_actions(enriched_triplets, memory.storage)
        applied = _apply_decisions(memory, decisions)
        memory._rebuild_graph_from_storage()

        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(
            "ingest_text_async ok | intent=%s triplets=%d actions=%d elapsed=%sms",
            intent, len(raw_triplets), len(applied), elapsed,
        )

        return {
            "status": "ok",
            "intent": intent,
            "triplet_count": len(raw_triplets),
            "applied_actions": applied,
            "elapsed_ms": elapsed,
        }

    except Exception as exc:
        logger.warning(
            "ingest_text_async failed (attempt %d): %s",
            self.request.retries + 1, exc,
        )
        raise self.retry(exc=exc)


# ── Task 2: batch ingest ────────────────────────────────────────────────

@celery_app.task(
    name="synapsemem.async_pipeline.tasks.batch_ingest_async",
    queue="ingest",
)
def batch_ingest_async(
    texts: List[str],
    user_id: str = "default_user",
    agent_id: str = "default_agent",
    session_id: str = "default_session",
    storage_backend: str = "sqlite",
    storage_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    task_ids = []
    for text in texts:
        result = ingest_text_async.apply_async(
            kwargs={
                "text": text,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "storage_backend": storage_backend,
                "storage_config": storage_config,
            }
        )
        task_ids.append(result.id)

    return {
        "status": "dispatched",
        "text_count": len(texts),
        "task_ids": task_ids,
    }


# ── Task 3: scheduled sleep consolidation ──────────────────────────────

@celery_app.task(
    name="synapsemem.async_pipeline.tasks.sleep_consolidate_async",
    queue="maintenance",
)
def sleep_consolidate_async(
    user_id: str = "default_user",
    agent_id: str = "default_agent",
    session_id: str = "default_session",
    storage_backend: str = "sqlite",
    storage_config: Optional[Dict] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    start = time.time()
    memory = _build_memory(
        storage_backend, user_id, agent_id, session_id, storage_config
    )
    report = memory.sleep_consolidate(dry_run=dry_run)
    report["elapsed_ms"] = round((time.time() - start) * 1000, 1)

    logger.info(
        "sleep_consolidate_async ok | promoted=%d merged=%d pruned=%d elapsed=%sms",
        report.get("promoted", 0),
        report.get("merged", 0),
        report.get("pruned", 0),
        report["elapsed_ms"],
    )
    return report


# ── Internal helper ─────────────────────────────────────────────────────

def _apply_decisions(memory, decisions: List[Dict]) -> List[Dict]:
    applied = []
    for decision in decisions:
        action = decision["action"]
        triplet = decision["triplet"]

        if action == "ADD":
            memory.storage.add_triplets([triplet])
            applied.append({"action": "ADD", "triplet": triplet})

        elif action == "UPDATE":
            existing = decision["existing"]
            updated = memory.storage.update_fact(existing["id"], triplet)
            if updated:
                applied.append({
                    "action": "UPDATE",
                    "triplet": triplet,
                    "replaced_record_id": existing["id"],
                })

        elif action == "DELETE":
            deleted_count = memory.storage.delete_fact(
                subject=triplet["subject"],
                predicate=triplet["predicate"],
                obj=triplet["object"],
            )
            applied.append({
                "action": "DELETE",
                "triplet": triplet,
                "deleted_count": deleted_count,
            })

        elif action == "NOOP":
            applied.append({"action": "NOOP", "triplet": triplet})

    return applied