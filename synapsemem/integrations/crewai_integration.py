"""CrewAI integration for SynapseMem.

Provides:
    SynapseMemCrewAITool  — a CrewAI-compatible tool agents can use for memory
    SynapseMemCrewStorage — a storage backend shim for CrewAI's memory system

Install:
    pip install crewai crewai-tools

Usage (as a tool on a CrewAI agent):
    from synapsemem.integrations.crewai_integration import SynapseMemCrewAITool

    memory_tool = SynapseMemCrewAITool(
        synapse=SynapseMemory(storage_backend="sqlite", user_id="crew_agent_1")
    )

    agent = Agent(
        role="Research Analyst",
        goal="...",
        tools=[memory_tool],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# pydantic is always available (it's a core dep via fastapi)
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool as CrewBaseTool
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


def _require_crewai():
    if not _CREWAI_AVAILABLE:
        raise ImportError(
            "crewai is required. Install with: pip install crewai crewai-tools"
        )


# ── Pydantic input schema ────────────────────────────────────────────────
# Always defined — pydantic is always present.
# Used by CrewAI's tool system when crewai is installed.

class _MemoryToolInput(BaseModel):
    command: str = Field(
        ...,
        description=(
            "Memory command. Examples: "
            "'store: I prefer Python over JavaScript' or "
            "'retrieve: programming preferences' or "
            "'facts about: user'"
        ),
    )


# ── CrewAI Tool ──────────────────────────────────────────────────────────

class SynapseMemCrewAITool:
    """
    CrewAI-compatible memory tool.

    Supports three command formats:
        store: <text>         — ingest text into private memory
        retrieve: <query>     — semantic search over memory
        facts about: <entity> — graph lookup for a specific entity

    Agents call this tool just like any other CrewAI tool.
    The tool is stateless — it creates no new state, just proxies to SynapseMemory.
    """

    name: str = "SynapseMem Memory Tool"
    description: str = (
        "Persistent memory for AI agents. "
        "Commands: 'store: <fact>', 'retrieve: <query>', 'facts about: <entity>'. "
        "Use this to remember important information across tasks."
    )

    def __init__(self, synapse, top_k: int = 5) -> None:
        _require_crewai()
        self.synapse = synapse
        self.top_k = top_k

    def _run(self, command: str) -> str:
        command = command.strip()
        lower = command.lower()

        # store
        if lower.startswith("store:"):
            text = command[6:].strip()
            if not text:
                return "Error: no text provided to store."
            actions = self.synapse.ingest(text)
            added   = sum(1 for a in actions if a["action"] == "ADD")
            updated = sum(1 for a in actions if a["action"] == "UPDATE")
            return (
                f"Memory stored. "
                f"Added {added} new fact(s), updated {updated} existing fact(s)."
            )

        # retrieve
        if lower.startswith("retrieve:"):
            query = command[9:].strip()
            if not query:
                return "Error: no query provided."
            return self._format_retrieve(query)

        # facts about
        if lower.startswith("facts about:"):
            entity = command[12:].strip().lower()
            facts = self.synapse.graph_facts_about(entity)
            if not facts:
                return f"No known facts about '{entity}'."
            lines = [
                f"- {f['subject']} {f['predicate']} {f['object']}"
                for f in facts
            ]
            return f"Facts about '{entity}':\n" + "\n".join(lines)

        # default: treat entire command as retrieve query
        return self._format_retrieve(command)

    def _format_retrieve(self, query: str) -> str:
        memories = self.synapse.retrieve(query, top_k=self.top_k)
        if not memories:
            return f"No relevant memories found for: '{query}'"
        lines = [
            f"- [{m.get('memory_type', 'episodic')}] "
            f"{m['subject']} {m['predicate']} {m['object']} "
            f"(relevance={m.get('score', 0):.3f})"
            for m in memories
        ]
        return f"Memories relevant to '{query}':\n" + "\n".join(lines)

    def run(self, command: str) -> str:
        return self._run(command)


# ── CrewAI Storage shim ─────────────────────────────────────────────────

class SynapseMemCrewStorage:
    """
    Shim that maps CrewAI's storage interface onto SynapseMemory.
    Use this if you want SynapseMem to power CrewAI's built-in memory system.

    CrewAI storage interface:
        save(value, metadata)
        search(query) → list
        reset()
    """

    def __init__(self, synapse) -> None:
        _require_crewai()
        self.synapse = synapse

    def save(self, value: str, metadata: Optional[Dict] = None) -> None:
        """Ingest a value into SynapseMem."""
        self.synapse.ingest(value)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories as a list of dicts."""
        memories = self.synapse.retrieve(query, top_k=top_k)
        return [
            {
                "id": m["id"],
                "text": f"{m['subject']} {m['predicate']} {m['object']}",
                "score": m.get("score", 0.0),
                "metadata": {
                    "topic": m.get("topic"),
                    "memory_type": m.get("memory_type"),
                    "priority": m.get("priority"),
                },
            }
            for m in memories
        ]

    def reset(self) -> None:
        self.synapse.reset()