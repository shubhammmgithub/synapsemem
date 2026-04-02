"""LangChain integration for SynapseMem.

Provides two integration points:

1. SynapseMemLangChainMemory
   Drop-in replacement for LangChain's ConversationBufferMemory.
   Pass it to any LangChain chain or agent.

2. SynapseMemTool
   A LangChain Tool that agents can call to store/retrieve memories.

Install:
    pip install langchain langchain-core

Usage:
    from synapsemem.integrations.langchain_integration import (
        SynapseMemLangChainMemory,
        SynapseMemTool,
    )

    # As a chain memory
    memory = SynapseMemLangChainMemory(
        synapse=SynapseMemory(storage_backend="sqlite"),
        memory_key="chat_history",
    )
    chain = ConversationChain(llm=llm, memory=memory)

    # As an agent tool
    tools = [SynapseMemTool(synapse=SynapseMemory(...))]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMemory
    from langchain.tools import BaseTool
    from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def _require_langchain():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain is required. Install with: pip install langchain langchain-core"
        )


class SynapseMemLangChainMemory:
    """
    LangChain-compatible memory wrapper around SynapseMemory.

    Implements the BaseMemory interface:
        - load_memory_variables(inputs) → dict
        - save_context(inputs, outputs) → None
        - clear() → None

    The memory_key returned matches LangChain's expected variable name.
    """

    def __init__(
        self,
        synapse,
        memory_key: str = "chat_history",
        input_key: str = "input",
        output_key: str = "output",
        top_k: int = 5,
        return_as_text: bool = True,
    ) -> None:
        _require_langchain()
        self.synapse = synapse
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.top_k = top_k
        self.return_as_text = return_as_text

    # LangChain BaseMemory interface

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant memories for the current input."""
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: "" if self.return_as_text else []}

        memories = self.synapse.retrieve(query, top_k=self.top_k)

        if self.return_as_text:
            lines = [
                f"[{m.get('memory_type', 'episodic')}] "
                f"{m['subject']} {m['predicate']} {m['object']}"
                for m in memories
            ]
            return {self.memory_key: "\n".join(lines)}
        else:
            return {self.memory_key: memories}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Ingest both the human input and AI response into memory."""
        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if human_input:
            self.synapse.ingest(human_input)
        if ai_output:
            self.synapse.ingest(ai_output)

    def clear(self) -> None:
        """Reset memory for the current scope."""
        self.synapse.reset()


class SynapseMemTool:
    """
    LangChain Tool wrapper for SynapseMem.
    Agents can call this tool to explicitly store or retrieve memories.

    The tool accepts commands:
        store: <text>     — ingest text into memory
        retrieve: <query> — retrieve top-k memories for a query

    Returns a human-readable string the agent can reason over.
    """

    name: str = "synapsemem"
    description: str = (
        "Persistent memory tool. "
        "Use 'store: <fact>' to remember something, "
        "or 'retrieve: <query>' to recall relevant memories."
    )

    def __init__(self, synapse, top_k: int = 5) -> None:
        _require_langchain()
        self.synapse = synapse
        self.top_k = top_k

    def _run(self, query: str) -> str:
        query = query.strip()

        if query.lower().startswith("store:"):
            text = query[6:].strip()
            actions = self.synapse.ingest(text)
            added = sum(1 for a in actions if a["action"] == "ADD")
            return f"Stored {added} new memory triplet(s)."

        if query.lower().startswith("retrieve:"):
            search = query[9:].strip()
            memories = self.synapse.retrieve(search, top_k=self.top_k)
            if not memories:
                return "No relevant memories found."
            lines = [
                f"- {m['subject']} {m['predicate']} {m['object']} "
                f"(score={m.get('score', 0):.3f})"
                for m in memories
            ]
            return "Relevant memories:\n" + "\n".join(lines)

        # Default: treat as retrieve
        memories = self.synapse.retrieve(query, top_k=self.top_k)
        if not memories:
            return "No relevant memories found."
        lines = [
            f"- {m['subject']} {m['predicate']} {m['object']}"
            for m in memories
        ]
        return "Relevant memories:\n" + "\n".join(lines)

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported — use ingest_text_async task instead.")

    def run(self, query: str) -> str:
        return self._run(query)