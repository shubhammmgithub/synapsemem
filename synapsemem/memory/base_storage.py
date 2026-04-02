"""Abstract base class for all SynapseMem storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseMemoryStorage(ABC):
    """
    Every storage backend must implement this interface exactly.
    SQLiteMemoryStorage, QdrantMemoryStorage, ChromaMemoryStorage all
    inherit from this class — manager.py stays untouched.
    """

    @abstractmethod
    def add_triplets(self, triplets: List[Dict]) -> None: ...

    @abstractmethod
    def all(self) -> List[Dict]: ...

    @abstractmethod
    def all_records(self) -> List[Dict]: ...

    @abstractmethod
    def find_exact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]: ...

    @abstractmethod
    def find_by_subject_predicate(self, subject: str, predicate: str) -> List[Dict]: ...

    @abstractmethod
    def update_fact(self, old_record_id: str, new_triplet: Dict) -> bool: ...

    @abstractmethod
    def reinforce(self, record_id: str) -> None: ...

    def update_last_accessed(self, record_id: str) -> None:
        # Default: alias to reinforce (matches existing behaviour)
        self.reinforce(record_id)

    @abstractmethod
    def delete_topic(self, topic: str) -> int: ...

    @abstractmethod
    def delete_fact(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int: ...

    @abstractmethod
    def merge_duplicates(self, merge_actions: List[Dict]) -> int: ...

    @abstractmethod
    def prune_memories(self, prune_actions: List[Dict]) -> int: ...

    @abstractmethod
    def find_semantic_memory(self, subject: str, predicate: str, obj: str) -> Optional[Dict]: ...

    @abstractmethod
    def promote_to_semantic(self, source_records: List[Dict]) -> Optional[Dict]: ...

    @abstractmethod
    def reset(self) -> None: ...

    # ------------------------------------------------------------------ #
    # Shared helpers — available to all subclasses                         #
    # ------------------------------------------------------------------ #

    def _triplet_to_text(self, triplet: Dict) -> str:
        return f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"