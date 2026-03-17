from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .graph.graph_builder import KnowledgeGraph
from .graph.query_engine import GraphQueryEngine
from .memory.anchors import AnchorManager
from .memory.extractor import TripletExtractor
from .memory.ingest_consolidator import IngestConsolidator
from .memory.retriever import MemoryRetriever
from .memory.sleep_consolidator import SleepConsolidator
from .memory.sqlite_storage import SQLiteMemoryStorage
from .memory.storage import MemoryStorage
from .prompt.builder import PromptBuilder


class SynapseMemory:
    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        pinned_facts: Optional[List[str]] = None,
        decay_rate: float = 0.05,
        storage_backend: str = "memory",
        sqlite_db_path: str = "synapsemem.db",
        user_id: str = "default_user",
        agent_id: str = "default_agent",
        session_id: str = "default_session",
    ) -> None:
        self.llm = llm
        self.decay_rate = decay_rate

        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id

        self.extractor = TripletExtractor()
        self.consolidator = IngestConsolidator()
        self.sleep_consolidator = SleepConsolidator()

        self.storage = self._build_storage(
            storage_backend=storage_backend,
            sqlite_db_path=sqlite_db_path,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        self.anchors = AnchorManager(initial_anchors=pinned_facts or [])
        self.prompt_builder = PromptBuilder()

        self.graph = KnowledgeGraph()
        self.graph_query = GraphQueryEngine(self.graph)

        self.retriever = MemoryRetriever(
            self.storage,
            anchor_manager=self.anchors,
            graph_query=self.graph_query,
        )

        self._rebuild_graph_from_storage()

    def _build_storage(
        self,
        storage_backend: str,
        sqlite_db_path: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ):
        backend = storage_backend.strip().lower()

        if backend == "memory":
            return MemoryStorage()

        if backend == "sqlite":
            return SQLiteMemoryStorage(
                db_path=sqlite_db_path,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )

        raise ValueError(
            f"Unsupported storage_backend='{storage_backend}'. "
            f"Use 'memory' or 'sqlite'."
        )

    def _rebuild_graph_from_storage(self) -> None:
        records = self.storage.all()
        triplets = [
            {
                "subject": record["subject"],
                "predicate": record["predicate"],
                "object": record["object"],
            }
            for record in records
        ]
        self.graph.clear()
        self.graph.add_triplets(triplets)

    def ingest(self, text: str) -> List[Dict[str, Any]]:
        triplets = self.extractor.extract(text)
        if not triplets:
            return []

        decisions = self.consolidator.decide_actions(triplets, self.storage)
        applied: List[Dict[str, Any]] = []

        for decision in decisions:
            action = decision["action"]
            triplet = decision["triplet"]

            if action == "ADD":
                self.storage.add_triplets([triplet])
                applied.append({"action": "ADD", "triplet": triplet})

            elif action == "UPDATE":
                existing = decision["existing"]
                updated = self.storage.update_fact(existing["id"], triplet)
                if updated:
                    applied.append({
                        "action": "UPDATE",
                        "triplet": triplet,
                        "replaced_record_id": existing["id"],
                    })

            elif action == "DELETE":
                deleted_count = self.storage.delete_fact(
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

        self._rebuild_graph_from_storage()
        return applied

    def sleep_consolidate(self, dry_run: bool = True) -> Dict[str, Any]:
        report = self.sleep_consolidator.run(
            storage=self.storage,
            dry_run=dry_run,
        )
        if not dry_run:
            self._rebuild_graph_from_storage()
        return report

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(query, top_k=top_k)

    def build_prompt(self, query: str, memories: List[Dict[str, Any]]) -> str:
        anchor_text = self.anchors.get_anchors()
        return self.prompt_builder.build(anchor_text, memories, query)

    def chat(self, user_input: str, top_k: int = 5) -> str:
        self.ingest(user_input)
        memory_pack = self.retrieve(user_input, top_k=top_k)
        prompt = self.build_prompt(user_input, memory_pack)

        if not self.llm:
            raise ValueError(
                "No LLM function provided. Pass a callable like llm(prompt) -> str."
            )

        return self.llm(prompt)

    def add_anchor(self, text: str) -> None:
        self.anchors.add_anchor(text)

    def remove_anchor(self, text: str) -> None:
        self.anchors.remove_anchor(text)

    def get_anchors(self) -> List[str]:
        return self.anchors.get_anchors()

    def delete_topic(self, topic: str) -> int:
        deleted = self.storage.delete_topic(topic)
        self._rebuild_graph_from_storage()
        return deleted

    def delete_fact(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> int:
        deleted = self.storage.delete_fact(subject=subject, predicate=predicate, obj=obj)
        self._rebuild_graph_from_storage()
        return deleted

    def graph_facts_about(self, entity: str):
        return self.graph_query.facts_about(entity)

    def graph_related_entities(self, entity: str, max_depth: int = 2):
        return self.graph_query.related_entities(entity, max_depth=max_depth)

    def graph_find_path(self, start: str, target: str, max_hops: int = 3):
        return self.graph_query.find_path(start, target, max_hops=max_hops)

    def reset(self, clear_anchors: bool = False) -> None:
        self.storage.reset()
        self.graph.clear()
        if clear_anchors:
            self.anchors.clear()