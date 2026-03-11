"""Triplet extraction from raw text data"""


from __future__ import annotations

import re
from typing import Dict, List


class TripletExtractor:
    """
    Lightweight rule-based triplet extractor.

    Output format:
    {
        "subject": "user",
        "predicate": "likes",
        "object": "hiking",
        "topic": "preference",
        "priority": 5,
        "source_text": "I like hiking"
    }
    """

    def __init__(self) -> None:
        self._patterns = [
            (r"\bI like ([\w\s\-]+)", "user", "likes", "preference", 5),
            (r"\bI love ([\w\s\-]+)", "user", "loves", "preference", 6),
            (r"\bI prefer ([\w\s\-]+)", "user", "prefers", "preference", 6),
            (r"\bI hate ([\w\s\-]+)", "user", "hates", "preference", 6),
            (r"\bI live in ([\w\s\-]+)", "user", "lives_in", "profile", 7),
            (r"\bI am from ([\w\s\-]+)", "user", "from", "profile", 7),
            (r"\bI work on ([\w\s\-]+)", "user", "works_on", "project", 7),
            (r"\bI am working on ([\w\s\-]+)", "user", "working_on", "project", 7),
            (r"\bMy name is ([\w\s\-]+)", "user", "name_is", "identity", 8),
            (r"\bI am preparing for ([\w\s\-]+)", "user", "preparing_for", "goal", 8),
            (r"\bI want to build ([\w\s\-]+)", "user", "wants_to_build", "goal", 7),
            (r"\bI use ([\w\s,\-]+)", "user", "uses", "tooling", 5),
        ]

    def extract(self, text: str) -> List[Dict]:
        """
        Extract triplets from raw user text.
        Falls back to a generic memory if no pattern matches.
        """
        text = self._normalize_text(text)
        triplets: List[Dict] = []

        for sentence in self._split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue

            matched = False
            for pattern, subject, predicate, topic, priority in self._patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    obj = self._clean_object(match.group(1))
                    if obj:
                        triplets.append(
                            self._build_triplet(
                                subject=subject,
                                predicate=predicate,
                                obj=obj,
                                topic=topic,
                                priority=priority,
                                source_text=sentence,
                            )
                        )
                        matched = True

            if not matched and self._looks_memory_worthy(sentence):
                triplets.append(
                    self._build_triplet(
                        subject="user",
                        predicate="said",
                        obj=sentence,
                        topic="general",
                        priority=3,
                        source_text=sentence,
                    )
                )

        return triplets

    def _split_sentences(self, text: str) -> List[str]:
        return re.split(r"[.!?\n]+", text)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _clean_object(self, obj: str) -> str:
        obj = obj.strip(" .,!?:;")
        obj = re.sub(r"\s+", " ", obj)
        return obj

    def _looks_memory_worthy(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        signals = [
            "i am",
            "i'm",
            "i like",
            "i love",
            "i prefer",
            "i hate",
            "my",
            "always",
            "never",
            "important",
            "remember",
            "working on",
            "building",
        ]
        return any(signal in sentence_lower for signal in signals) and len(sentence) > 8

    def _build_triplet(
        self,
        subject: str,
        predicate: str,
        obj: str,
        topic: str,
        priority: int,
        source_text: str,
    ) -> Dict:
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "topic": topic,
            "priority": priority,
            "source_text": source_text,
        }