from __future__ import annotations

from typing import Dict, List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for dashboard API support. "
        "Install with: pip install fastapi uvicorn pydantic"
    ) from exc

from synapsemem import SynapseMemory

app = FastAPI(title="SynapseMem Dashboard API", version="0.1.0")
memory = SynapseMemory()


class IngestRequest(BaseModel):
    text: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class AnchorRequest(BaseModel):
    text: str


def serialize_memory_record(record: Dict) -> Dict:
    """
    Clean a storage/retrieval record for API display.
    Removes large/noisy fields like embeddings.
    """
    return {
        "id": record.get("id"),
        "subject": record.get("subject"),
        "predicate": record.get("predicate"),
        "object": record.get("object"),
        "topic": record.get("topic"),
        "priority": record.get("priority"),
        "source_text": record.get("source_text"),
        "created_at": record.get("created_at"),
        "last_accessed_at": record.get("last_accessed_at"),
        "score": record.get("score"),
        "semantic_similarity": record.get("semantic_similarity"),
        "priority_score": record.get("priority_score"),
        "decay_score": record.get("decay_score"),
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "SynapseMem Dashboard API is running"}


@app.post("/memory/ingest")
def ingest_memory(payload: IngestRequest) -> Dict[str, object]:
    extracted_triplets = memory.extractor.extract(payload.text)
    memory.ingest(payload.text)

    clean_triplets = [
        {
            "subject": triplet.get("subject"),
            "predicate": triplet.get("predicate"),
            "object": triplet.get("object"),
            "topic": triplet.get("topic"),
            "priority": triplet.get("priority"),
            "source_text": triplet.get("source_text"),
        }
        for triplet in extracted_triplets
    ]

    return {
        "status": "ok",
        "message": "Memory ingested successfully",
        "extracted_triplets": clean_triplets,
        "triplet_count": len(clean_triplets),
    }


@app.post("/memory/retrieve")
def retrieve_memory(payload: RetrieveRequest) -> Dict[str, object]:
    results = memory.retrieve(payload.query, top_k=payload.top_k)
    clean_results = [serialize_memory_record(record) for record in results]

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "result_count": len(clean_results),
        "results": clean_results,
    }


@app.get("/memory/all")
def get_all_memory() -> Dict[str, object]:
    records = memory.storage.all()
    clean_records = [serialize_memory_record(record) for record in records]

    return {
        "record_count": len(clean_records),
        "records": clean_records,
    }


@app.post("/anchors/add")
def add_anchor(payload: AnchorRequest) -> Dict[str, object]:
    memory.add_anchor(payload.text)
    return {
        "status": "ok",
        "message": "Anchor added successfully",
        "anchors": memory.get_anchors(),
    }


@app.get("/anchors")
def list_anchors() -> Dict[str, List[str]]:
    return {"anchors": memory.get_anchors()}


@app.delete("/memory/topic/{topic}")
def delete_topic(topic: str) -> Dict[str, object]:
    deleted = memory.delete_topic(topic)
    return {
        "status": "ok",
        "topic": topic,
        "deleted": deleted,
    }


@app.post("/memory/reset")
def reset_memory() -> Dict[str, str]:
    memory.reset()
    return {"status": "ok", "message": "Memory reset successfully"}


@app.get("/graph/facts/{entity}")
def graph_facts(entity: str) -> Dict[str, object]:
    facts = memory.graph_facts_about(entity)
    return {
        "entity": entity,
        "fact_count": len(facts),
        "facts": facts,
    }


@app.get("/graph/related/{entity}")
def graph_related(entity: str, max_depth: int = 2) -> Dict[str, object]:
    related = memory.graph_related_entities(entity, max_depth=max_depth)
    return {
        "entity": entity,
        "max_depth": max_depth,
        "related_count": len(related),
        "related_entities": related,
    }


@app.get("/graph/path")
def graph_path(start: str, target: str, max_hops: int = 3) -> Dict[str, object]:
    path = memory.graph_find_path(start, target, max_hops=max_hops)
    return {
        "start": start,
        "target": target,
        "max_hops": max_hops,
        "path": path,
        "found": path is not None,
    }