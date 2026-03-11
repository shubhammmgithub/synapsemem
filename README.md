# SynapseMem

SynapseMem is a lightweight neuro-symbolic memory engine for LLM agents.

It helps AI systems remember useful user information across interactions by combining:

- atomic fact extraction
- memory consolidation
- semantic retrieval
- pinned anchors
- symbolic graph memory
- prompt construction

## Features

- Rule-based triplet extraction for memory facts
- In-memory storage for MVP usage
- Relevance-based retrieval
- Pinned anchor support
- Symbolic graph layer for basic fact traversal
- Provider-agnostic LLM integration
- CLI support
- FastAPI dashboard stub

## Installation

### Basic install

```bash
pip install .


Install with dashboard support
pip install ".[dashboard]"
Install with development tools
pip install ".[dev]"


Quick Start
from synapsemem import SynapseMemory

def fake_llm(prompt: str) -> str:
    return f"LLM got this prompt:\\n{prompt}"

memory = SynapseMemory(
    llm=fake_llm,
    pinned_facts=[
        "You are a helpful assistant.",
        "Prefer concise and accurate responses.",
    ]
)

memory.ingest("I love hiking.")
memory.ingest("I work on SynapseMem.")

results = memory.retrieve("What am I working on?")
print(results)

prompt = memory.build_prompt("Suggest something for me", results)
print(prompt)

response = memory.chat("I am preparing for freelancing internship.")
print(response)




Project Structure
synapsemem/
├── __init__.py
├── manager.py
├── cli/
├── dashboards/
├── graph/
├── memory/
├── prompt/
└── utils/


*CLI Usage

After installation:

synapsemem --help

Example:

synapsemem ingest "I love hiking."
synapsemem retrieve "What do I like?"

Note: the MVP CLI uses in-memory state for the current process only.

*Dashboard API

Run the local API:

uvicorn synapsemem.dashboards.api:app --reload
Running Tests
python -m unittest discover tests

Or with pytest:

pytest