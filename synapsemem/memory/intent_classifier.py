"""Intent classifier — runs before ingest to tag incoming text."""

from __future__ import annotations

import re
from typing import Dict, List


_SIGNALS: Dict[str, List[str]] = {
    "delete": [
        "forget that", "remove that", "delete that",
        "forget ", "remove ", "delete ",
    ],
    "task": [
        "i need to", "i have to", "i should", "i must", "i want to",
        "i'm going to", "i am going to", "remind me", "todo", "to-do",
        "working on", "i will",
    ],
    "tool_result": [
        "search returned", "api returned", "result:", "output:", "response:",
        "tool output", "function returned",
    ],
    "preference": [
        "i prefer", "i like", "i love", "i hate", "i dislike",
        "i always", "i never", "i usually", "i tend to",
    ],
    "fact": [
        "i am", "i'm", "my name is", "i live", "i work",
        "i use", "i am from", "i am in",
    ],
    # chitchat: only exact short words — never substrings
    "chitchat": [
        "hey", "hi", "hello", "thanks", "thank you", "ok", "okay",
        "sure", "got it", "sounds good", "nice", "cool",
    ],
}

_PRIORITY_OVERRIDE: Dict[str, int] = {
    "delete": 0,
    "chitchat": 1,
    "tool_result": 4,
    "task": 6,
    "preference": 5,
    "fact": 5,
}

# Pre-compile word-boundary patterns for chitchat to avoid
# substring false positives ("hiking" matching "hi", etc.)
_CHITCHAT_PATTERNS = [
    re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    for word in _SIGNALS["chitchat"]
]


class IntentClassifier:

    def classify(self, text: str) -> str:
        lower = text.strip().lower()

        # Check non-chitchat intents first using simple substring (safe — no short words)
        for intent in ("delete", "tool_result", "task", "preference", "fact"):
            for signal in _SIGNALS[intent]:
                if signal in lower:
                    return intent

        # Chitchat: word-boundary match only
        for pattern in _CHITCHAT_PATTERNS:
            if pattern.search(lower):
                # Only classify as chitchat if the text is short
                # — avoids "I said hello to my colleague and now I work on..."
                if len(lower.split()) <= 6:
                    return "chitchat"

        if len(lower.split()) <= 3:
            return "chitchat"

        return "fact"

    def should_skip(self, intent: str) -> bool:
        return intent == "chitchat"

    def priority_boost(self, intent: str, base_priority: int) -> int:
        override = _PRIORITY_OVERRIDE.get(intent)
        if override is not None:
            return max(override, base_priority)
        return base_priority

    def enrich_triplets(self, triplets: List[Dict], intent: str) -> List[Dict]:
        enriched = []
        for triplet in triplets:
            t = dict(triplet)
            t["intent"] = intent
            t["priority"] = self.priority_boost(intent, int(t.get("priority", 3)))
            enriched.append(t)
        return enriched