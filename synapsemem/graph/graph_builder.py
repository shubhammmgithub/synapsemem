"""Knowledge graph building and management"""


from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple


class KnowledgeGraph:
    """
    Minimal symbolic knowledge graph for SynapseMem MVP.

    Stores triplets like:
        user --likes--> hiking

    Internal representation:
        adjacency[subject] = [(predicate, object), ...]
    """

    def __init__(self) -> None:
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.nodes: Set[str] = set()

    def add_triplet(self, subject: str, predicate: str, obj: str) -> None:
        subject = self._normalize(subject)
        predicate = self._normalize(predicate)
        obj = self._normalize(obj)

        edge = (predicate, obj)
        reverse_edge = (predicate, subject)

        if edge not in self.adjacency[subject]:
            self.adjacency[subject].append(edge)

        if reverse_edge not in self.reverse_adjacency[obj]:
            self.reverse_adjacency[obj].append(reverse_edge)

        self.nodes.add(subject)
        self.nodes.add(obj)

    def add_triplets(self, triplets: List[Dict]) -> None:
        for triplet in triplets:
            self.add_triplet(
                subject=triplet["subject"],
                predicate=triplet["predicate"],
                obj=triplet["object"],
            )

    def get_outgoing(self, subject: str) -> List[Tuple[str, str]]:
        return list(self.adjacency.get(self._normalize(subject), []))

    def get_incoming(self, obj: str) -> List[Tuple[str, str]]:
        return list(self.reverse_adjacency.get(self._normalize(obj), []))

    def neighbors(self, node: str) -> List[str]:
        node = self._normalize(node)
        outgoing = [obj for _, obj in self.adjacency.get(node, [])]
        incoming = [subj for _, subj in self.reverse_adjacency.get(node, [])]
        return list(dict.fromkeys(outgoing + incoming))

    def has_node(self, node: str) -> bool:
        return self._normalize(node) in self.nodes

    def all_triplets(self) -> List[Dict]:
        triplets: List[Dict] = []
        for subject, edges in self.adjacency.items():
            for predicate, obj in edges:
                triplets.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                    }
                )
        return triplets

    def clear(self) -> None:
        self.adjacency.clear()
        self.reverse_adjacency.clear()
        self.nodes.clear()

    def _normalize(self, text: str) -> str:
        return str(text).strip().lower()