"""SynapseMem Dashboard API — Phase 3.

New endpoints added over Phase 2:

  Async ingest (Step 2):
    POST /memory/ingest/async          — fire-and-forget ingest via Celery
    POST /memory/ingest/batch/async    — batch fire-and-forget ingest
    GET  /tasks/{task_id}              — poll Celery task status

  Shared memory (Step 3):
    POST /shared/{workspace_id}/write  — write a fact to shared workspace
    GET  /shared/{workspace_id}/facts  — read all shared facts
    GET  /shared/{workspace_id}/stats  — workspace stats
    DELETE /shared/{workspace_id}/fact — delete a shared fact

  Memory compression (Step 4):
    POST /memory/compress              — run compression pass (dry_run or live)

All Phase 2 endpoints are unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI is required. Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from synapsemem import SynapseMemory
from synapsemem.memory.shared_memory import SharedMemoryStore
from synapsemem.memory.memory_compressor import MemoryCompressor

# ── Try importing Celery tasks (optional — only if async deps installed) ──
try:
    from synapsemem.async_pipeline.tasks import (
        ingest_text_async,
        batch_ingest_async,
        sleep_consolidate_async,
    )
    from celery.result import AsyncResult
    from synapsemem.async_pipeline.celery_app import celery_app
    _CELERY_AVAILABLE = True
except ImportError:
    _CELERY_AVAILABLE = False


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SynapseMem Dashboard API",
    version="0.3.0",
    description="Bio-inspired memory engine for AI agents — Phase 3 API.",
)

# ── Default memory instance (synchronous path) ───────────────────────────

memory = SynapseMemory(
    storage_backend="sqlite",
    sqlite_db_path="synapsemem.db",
    user_id="dashboard_user",
    agent_id="dashboard_agent",
    session_id="dashboard_session",
)

# ── Shared memory store cache (one per workspace_id) ─────────────────────
# In production you'd use a proper registry / DI container.
_shared_stores: Dict[str, SharedMemoryStore] = {}


def _get_shared_store(
    workspace_id: str,
    conflict_strategy: str = "last_write_wins",
    db_path: str = "synapsemem.db",
) -> SharedMemoryStore:
    if workspace_id not in _shared_stores:
        _shared_stores[workspace_id] = SharedMemoryStore(
            workspace_id=workspace_id,
            db_path=db_path,
            conflict_strategy=conflict_strategy,  # type: ignore[arg-type]
        )
    return _shared_stores[workspace_id]


# ── Pydantic models ───────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str

class IngestAsyncRequest(BaseModel):
    text: str
    user_id: str = "default_user"
    agent_id: str = "default_agent"
    session_id: str = "default_session"
    storage_backend: str = "sqlite"
    storage_config: Optional[Dict] = None

class BatchIngestAsyncRequest(BaseModel):
    texts: List[str]
    user_id: str = "default_user"
    agent_id: str = "default_agent"
    session_id: str = "default_session"
    storage_backend: str = "sqlite"
    storage_config: Optional[Dict] = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class AnchorRequest(BaseModel):
    text: str

class SleepRequest(BaseModel):
    dry_run: bool = True

class CompressRequest(BaseModel):
    dry_run: bool = True
    similarity_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    min_cluster_size: int = Field(default=3, ge=2)

class SharedWriteRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    topic: str = "general"
    priority: int = Field(default=3, ge=1, le=10)
    source_text: str = ""
    agent_id: str
    conflict_strategy: str = "last_write_wins"

class SharedDeleteRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    agent_id: str


# ── Serialiser ────────────────────────────────────────────────────────────

def serialize_memory_record(record: Dict) -> Dict:
    return {
        "id": record.get("id"),
        "subject": record.get("subject"),
        "predicate": record.get("predicate"),
        "object": record.get("object"),
        "topic": record.get("topic"),
        "priority": record.get("priority"),
        "source_text": record.get("source_text"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "last_accessed_at": record.get("last_accessed_at"),
        "reinforcement_count": record.get("reinforcement_count"),
        "memory_type": record.get("memory_type"),
        "status": record.get("status"),
        "source_count": record.get("source_count"),
        "consolidated_from": record.get("consolidated_from", []),
        "score": record.get("score"),
        "base_score": record.get("base_score"),
        "semantic_similarity": record.get("semantic_similarity"),
        "priority_score": record.get("priority_score"),
        "decay_score": record.get("decay_score"),
        "synaptic_strength": record.get("synaptic_strength"),
        "anchor_bonus": record.get("anchor_bonus"),
        "graph_bonus": record.get("graph_bonus"),
        "semantic_memory_bonus": record.get("semantic_memory_bonus"),
        "source_count_bonus": record.get("source_count_bonus"),
    }


# ======================================================================== #
# Phase 2 endpoints — unchanged                                             #
# ======================================================================== #

@app.get("/")
def root() -> Dict:
    return {
        "message": "SynapseMem Dashboard API is running",
        "version": "0.3.0",
        "celery_available": _CELERY_AVAILABLE,
    }


@app.post("/memory/ingest")
def ingest_memory(payload: IngestRequest) -> Dict:
    extracted_triplets = memory.extractor.extract(payload.text)
    applied_actions = memory.ingest(payload.text)
    clean_triplets = [
        {
            "subject": t.get("subject"),
            "predicate": t.get("predicate"),
            "object": t.get("object"),
            "topic": t.get("topic"),
            "priority": t.get("priority"),
            "source_text": t.get("source_text"),
        }
        for t in extracted_triplets
    ]
    return {
        "status": "ok",
        "message": "Memory ingested successfully",
        "extracted_triplets": clean_triplets,
        "triplet_count": len(clean_triplets),
        "applied_actions": applied_actions,
    }


@app.post("/memory/retrieve")
def retrieve_memory(payload: RetrieveRequest) -> Dict:
    results = memory.retrieve(payload.query, top_k=payload.top_k)
    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "result_count": len(results),
        "results": [serialize_memory_record(r) for r in results],
    }


@app.get("/memory/all")
def get_all_active_memory() -> Dict:
    records = memory.storage.all()
    return {
        "record_count": len(records),
        "records": [serialize_memory_record(r) for r in records],
    }


@app.get("/memory/all-records")
def get_all_memory_records() -> Dict:
    records = (
        memory.storage.all_records()
        if hasattr(memory.storage, "all_records")
        else memory.storage.all()
    )
    return {
        "record_count": len(records),
        "records": [serialize_memory_record(r) for r in records],
    }


@app.get("/memory/stats")
def get_memory_stats() -> Dict:
    records = (
        memory.storage.all_records()
        if hasattr(memory.storage, "all_records")
        else memory.storage.all()
    )
    active   = [r for r in records if r.get("status") == "active"]
    episodic = [r for r in active   if r.get("memory_type") == "episodic"]
    semantic = [r for r in active   if r.get("memory_type") == "semantic"]
    merged   = [r for r in records  if r.get("status") == "merged"]
    pruned   = [r for r in records  if r.get("status") == "pruned"]
    return {
        "total_records": len(records),
        "active_records": len(active),
        "episodic_active_records": len(episodic),
        "semantic_active_records": len(semantic),
        "merged_records": len(merged),
        "pruned_records": len(pruned),
    }


@app.post("/memory/sleep")
def run_sleep_consolidation(payload: SleepRequest) -> Dict:
    report = memory.sleep_consolidate(dry_run=payload.dry_run)
    return {
        "status": "ok",
        "message": "Sleep consolidation executed",
        "report": report,
    }


@app.post("/anchors/add")
def add_anchor(payload: AnchorRequest) -> Dict:
    memory.add_anchor(payload.text)
    return {
        "status": "ok",
        "message": "Anchor added successfully",
        "anchors": memory.get_anchors(),
    }


@app.get("/anchors")
def list_anchors() -> Dict:
    return {"anchors": memory.get_anchors()}


@app.delete("/memory/topic/{topic}")
def delete_topic(topic: str) -> Dict:
    deleted = memory.delete_topic(topic)
    return {"status": "ok", "topic": topic, "deleted": deleted}


@app.post("/memory/reset")
def reset_memory() -> Dict:
    memory.reset()
    return {"status": "ok", "message": "Memory reset successfully"}


@app.get("/graph/facts/{entity}")
def graph_facts(entity: str) -> Dict:
    facts = memory.graph_facts_about(entity)
    return {"entity": entity, "fact_count": len(facts), "facts": facts}


@app.get("/graph/related/{entity}")
def graph_related(entity: str, max_depth: int = 2) -> Dict:
    related = memory.graph_related_entities(entity, max_depth=max_depth)
    return {
        "entity": entity,
        "max_depth": max_depth,
        "related_count": len(related),
        "related_entities": related,
    }


@app.get("/graph/path")
def graph_path(start: str, target: str, max_hops: int = 3) -> Dict:
    path = memory.graph_find_path(start, target, max_hops=max_hops)
    return {
        "start": start,
        "target": target,
        "max_hops": max_hops,
        "path": path,
        "found": path is not None,
    }


# ======================================================================== #
# Phase 3 — Step 2: Async ingest endpoints                                  #
# ======================================================================== #

def _require_celery():
    if not _CELERY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Celery is not installed. "
                "Run: pip install synapsemem[async] and start a Redis broker."
            ),
        )


@app.post("/memory/ingest/async")
def ingest_memory_async(payload: IngestAsyncRequest) -> Dict:
    """
    Fire-and-forget ingest via Celery.
    Returns a task_id immediately — poll /tasks/{task_id} for the result.
    """
    _require_celery()
    result = ingest_text_async.apply_async(
        kwargs={
            "text": payload.text,
            "user_id": payload.user_id,
            "agent_id": payload.agent_id,
            "session_id": payload.session_id,
            "storage_backend": payload.storage_backend,
            "storage_config": payload.storage_config,
        }
    )
    return {
        "status": "queued",
        "task_id": result.id,
        "message": "Ingest task queued. Poll /tasks/{task_id} for result.",
    }


@app.post("/memory/ingest/batch/async")
def ingest_batch_async(payload: BatchIngestAsyncRequest) -> Dict:
    """
    Batch fire-and-forget ingest.
    Dispatches one Celery task per text. Returns all task IDs.
    """
    _require_celery()
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts list is empty.")
    if len(payload.texts) > 500:
        raise HTTPException(status_code=400, detail="Max 500 texts per batch.")

    result = batch_ingest_async.apply_async(
        kwargs={
            "texts": payload.texts,
            "user_id": payload.user_id,
            "agent_id": payload.agent_id,
            "session_id": payload.session_id,
            "storage_backend": payload.storage_backend,
            "storage_config": payload.storage_config,
        }
    )
    return {
        "status": "queued",
        "task_id": result.id,
        "text_count": len(payload.texts),
        "message": "Batch ingest queued. Poll /tasks/{task_id} for result.",
    }


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str) -> Dict:
    """
    Poll the status of any Celery task by ID.
    States: PENDING → STARTED → SUCCESS | FAILURE | RETRY
    """
    _require_celery()
    result = AsyncResult(task_id, app=celery_app)

    response: Dict = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    elif result.status == "RETRY":
        response["info"] = str(result.info)

    return response


# ======================================================================== #
# Phase 3 — Step 3: Shared memory endpoints                                 #
# ======================================================================== #

@app.post("/shared/{workspace_id}/write")
def shared_write(workspace_id: str, payload: SharedWriteRequest) -> Dict:
    """Write a fact to a shared workspace. Applies conflict resolution policy."""
    store = _get_shared_store(
        workspace_id=workspace_id,
        conflict_strategy=payload.conflict_strategy,
    )
    triplet = {
        "subject": payload.subject,
        "predicate": payload.predicate,
        "object": payload.object,
        "topic": payload.topic,
        "priority": payload.priority,
        "source_text": payload.source_text,
    }
    result = store.write_fact(triplet, agent_id=payload.agent_id)
    return {
        "status": "ok",
        "workspace_id": workspace_id,
        "action": result["action"],
        "reason": result.get("reason"),
        "record": {
            k: v for k, v in (result.get("record") or {}).items()
            if k != "embedding"
        },
    }


@app.get("/shared/{workspace_id}/facts")
def shared_read(workspace_id: str, topic: Optional[str] = None) -> Dict:
    """Return all active facts in a shared workspace."""
    store = _get_shared_store(workspace_id)
    facts = store.read_facts(topic=topic)
    clean = [
        {k: v for k, v in f.items() if k != "embedding"}
        for f in facts
    ]
    return {
        "workspace_id": workspace_id,
        "topic_filter": topic,
        "fact_count": len(clean),
        "facts": clean,
    }


@app.get("/shared/{workspace_id}/stats")
def shared_stats(workspace_id: str) -> Dict:
    """Return stats for a shared workspace."""
    store = _get_shared_store(workspace_id)
    return store.workspace_stats()


@app.get("/shared/{workspace_id}/agent/{agent_id}")
def shared_facts_by_agent(workspace_id: str, agent_id: str) -> Dict:
    """Return shared facts contributed by a specific agent."""
    store = _get_shared_store(workspace_id)
    facts = store.facts_by_agent(agent_id)
    clean = [{k: v for k, v in f.items() if k != "embedding"} for f in facts]
    return {
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "fact_count": len(clean),
        "facts": clean,
    }


@app.delete("/shared/{workspace_id}/fact")
def shared_delete(workspace_id: str, payload: SharedDeleteRequest) -> Dict:
    """Soft-delete a fact from a shared workspace."""
    store = _get_shared_store(workspace_id)
    deleted = store.delete_fact(
        subject=payload.subject,
        predicate=payload.predicate,
        obj=payload.object,
        agent_id=payload.agent_id,
    )
    return {
        "status": "ok" if deleted else "not_found",
        "workspace_id": workspace_id,
        "deleted": deleted,
    }


# ======================================================================== #
# Phase 3 — Step 4: Memory compression endpoint                             #
# ======================================================================== #

@app.post("/memory/compress")
def run_memory_compression(payload: CompressRequest) -> Dict:
    """
    Run the semantic compression pass over episodic memories.

    dry_run=true  → shows what would be compressed, writes nothing.
    dry_run=false → actually compresses and stores summarised semantic facts.

    Uses the LLM attached to the global memory instance if available.
    Falls back to representative-selection if no LLM is configured.
    """
    compressor = MemoryCompressor(
        llm=memory.llm,
        similarity_threshold=payload.similarity_threshold,
        min_cluster_size=payload.min_cluster_size,
    )
    report = compressor.run(storage=memory.storage, dry_run=payload.dry_run)
    return {
        "status": "ok",
        "message": (
            "Compression dry-run complete — no changes written."
            if payload.dry_run
            else "Compression complete."
        ),
        "report": report,
    }