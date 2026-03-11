"""Query engine for graph traversal and search"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from .graph_builder import KnowledgeGraph


class GraphQueryEngine:
    """
    Query interface for the symbolic graph.

    MVP features:
    - find direct facts for an entity
    - find reverse facts
    - breadth-first search for small multi-hop paths
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def facts_about(self, entity: str) -> List[Dict]:
        """
        Returns direct outgoing facts for the entity.
        Example:
            user -> likes -> hiking
        """
        entity = entity.strip().lower()
        return [
            {"subject": entity, "predicate": predicate, "object": obj}
            for predicate, obj in self.graph.get_outgoing(entity)
        ]

    def facts_pointing_to(self, entity: str) -> List[Dict]:
        """
        Returns reverse incoming facts for the entity.
        Example:
            user -> likes -> hiking
        queried by 'hiking' returns:
            subject=user, predicate=likes, object=hiking
        """
        entity = entity.strip().lower()
        return [
            {"subject": subject, "predicate": predicate, "object": entity}
            for predicate, subject in self.graph.get_incoming(entity)
        ]

    def find_path(self, start: str, target: str, max_hops: int = 3) -> Optional[List[str]]:
        """
        Find a simple BFS path between two nodes.
        Returns a node path like:
            ["user", "synapsemem", "project"]
        if reachable within max_hops, else None.
        """
        start = start.strip().lower()
        target = target.strip().lower()

        if not self.graph.has_node(start) or not self.graph.has_node(target):
            return None

        if start == target:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()

            if len(path) - 1 >= max_hops:
                continue

            for neighbor in self.graph.neighbors(node):
                if neighbor in visited:
                    continue

                new_path = path + [neighbor]
                if neighbor == target:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def related_entities(self, entity: str, max_depth: int = 2) -> List[str]:
        """
        Return nearby entities up to max_depth hops away.
        Useful for future graph-based retrieval.
        """
        entity = entity.strip().lower()

        if not self.graph.has_node(entity):
            return []

        queue = deque([(entity, 0)])
        visited = {entity}
        related: List[str] = []

        while queue:
            node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.append(neighbor)
                    queue.append((neighbor, depth + 1))

        return related