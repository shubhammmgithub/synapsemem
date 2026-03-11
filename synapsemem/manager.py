from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .graph.graph_builder import KnowledgeGraph
from .graph.query_engine import GraphQueryEngine
from .memory.anchors import AnchorManager
from .memory.consolidator import MemoryConsolidator
from .memory.extractor import TripletExtractor
from .memory.retriever import MemoryRetriever
from .memory.storage import MemoryStorage
from .prompt.builder import PromptBuilder


class SynapseMemory:
    """
    High-level orchestration layer for SynapseMem.

    Responsibilities:
    - ingest raw text
    - extract memory triplets
    - consolidate / deduplicate them
    - store them in memory storage
    - mirror them into a symbolic graph
    - retrieve relevant memories
    - merge anchors + memories into a final prompt
    - optionally call a user-supplied LLM function
    """

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        pinned_facts: Optional[List[str]] = None,
        decay_rate: float = 0.05,
    ) -> None:
        self.llm = llm
        self.decay_rate = decay_rate

        self.extractor = TripletExtractor()
        self.consolidator = MemoryConsolidator()
        self.storage = MemoryStorage()
        self.retriever = MemoryRetriever(self.storage)
        self.anchors = AnchorManager(initial_anchors=pinned_facts or [])
        self.prompt_builder = PromptBuilder()

        # Symbolic graph layer
        self.graph = KnowledgeGraph()
        self.graph_query = GraphQueryEngine(self.graph)

    # ---------------------------
    # MEMORY INGESTION PIPELINE
    # ---------------------------

    def ingest(self, text: str) -> None:
        """
        Ingest raw text into the memory system.

        Flow:
        raw text -> triplet extraction -> consolidation -> storage -> graph
        """
        triplets = self.extractor.extract(text)
        clean_triplets = self.consolidator.process(triplets)

        if not clean_triplets:
            return

        self.storage.add_triplets(clean_triplets)
        self.graph.add_triplets(clean_triplets)

    # ---------------------------
    # MEMORY RETRIEVAL PIPELINE
    # ---------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from storage.
        """
        return self.retriever.retrieve(query, top_k=top_k)

    # ---------------------------
    # PROMPT BUILDING
    # ---------------------------

    def build_prompt(self, query: str, memories: List[Dict[str, Any]]) -> str:
        """
        Build the final prompt using:
        - pinned anchors
        - retrieved memories
        - current user query
        """
        anchor_text = self.anchors.get_anchors()
        return self.prompt_builder.build(anchor_text, memories, query)

    # ---------------------------
    # HIGH-LEVEL CHAT METHOD
    # ---------------------------

    def chat(self, user_input: str, top_k: int = 5) -> str:
        """
        Full end-to-end flow:
        ingest -> retrieve -> build prompt -> call LLM
        """
        self.ingest(user_input)
        memory_pack = self.retrieve(user_input, top_k=top_k)
        prompt = self.build_prompt(user_input, memory_pack)

        if not self.llm:
            raise ValueError(
                "No LLM function provided. SynapseMemory is provider-agnostic, "
                "so pass a callable like llm(prompt) -> str."
            )

        return self.llm(prompt)

    # ---------------------------
    # ANCHOR MANAGEMENT
    # ---------------------------

    def add_anchor(self, text: str) -> None:
        self.anchors.add_anchor(text)

    def remove_anchor(self, text: str) -> None:
        self.anchors.remove_anchor(text)

    def get_anchors(self) -> List[str]:
        return self.anchors.get_anchors()

    # ---------------------------
    # MEMORY DELETION HELPERS
    # ---------------------------

    def delete_topic(self, topic: str) -> int:
        """
        Delete all memories belonging to a topic from storage.
        Note: the MVP storage and graph are separate in-memory systems.
        If you want full graph-topic deletion, we can add it in Version B.
        """
        return self.storage.delete_topic(topic)

    def delete_fact(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> int:
        """
        Delete matching facts from storage.
        """
        return self.storage.delete_fact(subject=subject, predicate=predicate, obj=obj)

    # ---------------------------
    # GRAPH HELPERS
    # ---------------------------

    def graph_facts_about(self, entity: str):
        return self.graph_query.facts_about(entity)

    def graph_related_entities(self, entity: str, max_depth: int = 2):
        return self.graph_query.related_entities(entity, max_depth=max_depth)

    def graph_find_path(self, start: str, target: str, max_hops: int = 3):
        return self.graph_query.find_path(start, target, max_hops=max_hops)

    # ---------------------------
    # RESET
    # ---------------------------

    def reset(self) -> None:
        """
        Clear all memory state.
        """
        self.storage.reset()
        self.anchors.clear()
        self.graph.clear()