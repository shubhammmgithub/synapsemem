"""Prompt construction and building"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .templates import (
    ANCHOR_SECTION_TEMPLATE,
    FULL_PROMPT_TEMPLATE,
    MEMORY_SECTION_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    USER_SECTION_TEMPLATE,
)


class PromptBuilder:
    """
    Builds the final prompt from:
    - pinned anchors
    - retrieved memories
    - current user query
    """

    def build(
        self,
        anchors: Sequence[str],
        memories: List[Dict],
        query: str,
    ) -> str:
        anchor_text = self._format_anchors(anchors)
        memory_text = self._format_memories(memories)
        user_text = self._format_user_query(query)

        return FULL_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT_TEMPLATE.strip(),
            anchor_section=anchor_text,
            memory_section=memory_text,
            user_section=user_text,
        ).strip()

    def _format_anchors(self, anchors: Sequence[str]) -> str:
        if not anchors:
            return ANCHOR_SECTION_TEMPLATE.format(anchors="- None")

        lines = [f"- {anchor.strip()}" for anchor in anchors if str(anchor).strip()]
        if not lines:
            lines = ["- None"]

        return ANCHOR_SECTION_TEMPLATE.format(anchors="\n".join(lines))

    def _format_memories(self, memories: List[Dict]) -> str:
        if not memories:
            return MEMORY_SECTION_TEMPLATE.format(memories="- No relevant memory found")

        lines: List[str] = []
        for idx, memory in enumerate(memories, start=1):
            subject = memory.get("subject", "unknown")
            predicate = memory.get("predicate", "related_to")
            obj = memory.get("object", "")
            topic = memory.get("topic", "general")
            score = memory.get("score", None)

            fact = f"{subject} {predicate} {obj}".strip()
            if score is not None:
                line = f"{idx}. {fact}  [topic={topic}, score={score}]"
            else:
                line = f"{idx}. {fact}  [topic={topic}]"

            lines.append(line)

        return MEMORY_SECTION_TEMPLATE.format(memories="\n".join(lines))

    def _format_user_query(self, query: str) -> str:
        query = str(query).strip()
        if not query:
            query = "No user query provided."
        return USER_SECTION_TEMPLATE.format(query=query)