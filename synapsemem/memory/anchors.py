"""Pinned truths - permanent memory anchors"""


from __future__ import annotations

from typing import Iterable, List


class AnchorManager:
    """
    Stores fixed high-priority memory strings.
    """

    def __init__(self, initial_anchors: Iterable[str] | None = None) -> None:
        self._anchors: List[str] = []
        for anchor in initial_anchors or []:
            self.add_anchor(anchor)

    def add_anchor(self, text: str) -> None:
        text = str(text).strip()
        if text and text not in self._anchors:
            self._anchors.append(text)

    def remove_anchor(self, text: str) -> None:
        text = str(text).strip()
        self._anchors = [anchor for anchor in self._anchors if anchor != text]

    def get_anchors(self) -> List[str]:
        return list(self._anchors)

    def clear(self) -> None:
        self._anchors.clear()