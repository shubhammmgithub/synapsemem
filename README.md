# SynapseMem

**A bio-inspired, production-ready memory engine for AI agents.**

SynapseMem gives AI agents persistent, evolving memory that mimics how the human brain stores and retrieves information — with structured extraction, time-aware decay, graph-based reasoning, and multi-agent shared memory.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen)](#testing)
[![PyPI](https://img.shields.io/badge/pypi-synapsemem-orange)](https://pypi.org/project/synapsemem)

---

## Why SynapseMem?

Most LLM applications are stateless. When memory is added, it is usually flat vector storage with no structure, no forgetting, and no learning over time.

SynapseMem takes a different approach — modelling memory the way the brain does:

| Human Brain | SynapseMem |
|---|---|
| Episodic memory | Raw ingested triplets |
| Semantic memory | Consolidated stable facts |
| Reinforcement | Access-frequency boosting |
| Forgetting | Time-based decay + pruning |
| Sleep consolidation | Offline compression pass |
| Synaptic strength | Hybrid relevance score |

---

## Feature Overview

**Core memory pipeline**
- Converts raw text into structured `(subject, predicate, object)` triplets
- ADD / UPDATE / DELETE / NOOP consolidation on every ingest
- Episodic → semantic memory promotion via sleep consolidation
- Time-aware decay with synaptic strength scoring

**Retrieval**
- Hybrid scoring: semantic similarity + decay + priority + graph context + anchor bias
- Knowledge graph with multi-hop reasoning and path finding
- Pinned anchors that always influence context

**Storage backends**
- In-memory (testing / prototyping)
- SQLite (local persistence, default)
- Qdrant (production vector DB)
- Chroma (production vector DB)

**Phase 3 — agent-native features**
- Async ingest pipeline via Celery + Redis
- Intent classification before ingest (fact / preference / task / tool result / delete / chitchat)
- Multi-agent shared memory with conflict resolution (last-write-wins, no-overwrite, anchor-weighted)
- Memory compression: LLM-powered semantic clustering of similar episodic memories
- LangChain and CrewAI integration shims

**Operations**
- FastAPI dashboard with 20+ endpoints
- CLI interface
- Built-in benchmark suite (ingest, retrieval, prompt size, quality, sleep)

---

## Installation

```bash
# Minimal — core memory engine only
pip install synapsemem

# With FastAPI dashboard
pip install "synapsemem[dashboard]"

# With vector DB support
pip install "synapsemem[vector]"

# With async pipeline (Celery + Redis)
pip install "synapsemem[async]"

# With LangChain integration
pip install "synapsemem[langchain]"

# With CrewAI integration
pip install "synapsemem[crewai]"

# Everything
pip install "synapsemem[all]"

# Development
pip install "synapsemem[dev]"
```

---

## Quick Start

### Basic usage

```python
from synapsemem import SynapseMemory

def my_llm(prompt: str) -> str:
    # Replace with your actual LLM call
    # e.g. openai.chat.completions.create(...)
    return "LLM response here"

memory = SynapseMemory(
    llm=my_llm,
    pinned_facts=["You are a helpful assistant."],
    storage_backend="sqlite",         # persists to synapsemem.db
    user_id="alice",
    agent_id="assistant",
    session_id="session_001",
)

# Ingest text — extracts triplets, deduplicates, stores
memory.ingest("I love hiking and outdoor activities.")
memory.ingest("I am preparing for a machine learning internship.")
memory.ingest("I prefer Python over JavaScript.")

# Retrieve relevant memories for a query
results = memory.retrieve("What are my interests?", top_k=5)
for r in results:
    print(f"{r['subject']} {r['predicate']} {r['object']}  (score={r['score']:.3f})")

# Full chat turn: ingest + retrieve + build prompt + call LLM
response = memory.chat("Suggest a project for me.")
print(response)
```

### Switching storage backends

```python
# SQLite — persistent local storage (default for production)
memory = SynapseMemory(storage_backend="sqlite", sqlite_db_path="./my_agent.db")

# Qdrant — requires: pip install qdrant-client
# docker run -p 6333:6333 qdrant/qdrant
memory = SynapseMemory(
    storage_backend="qdrant",
    qdrant_url="http://localhost:6333",
    user_id="alice",
)

# Chroma — requires: pip install chromadb
memory = SynapseMemory(
    storage_backend="chroma",
    chroma_persist_directory="./chroma_db",
    user_id="alice",
)
```

### Sleep consolidation

Run periodically (nightly) to merge duplicates, promote stable facts, and prune weak memories:

```python
# Dry run — see what would happen without writing
report = memory.sleep_consolidate(dry_run=True)
print(f"Would promote {report['promoted']}, merge {report['merged']}, prune {report['pruned']}")

# Live run — actually consolidates
report = memory.sleep_consolidate(dry_run=False)
```

### Memory compression

Cluster similar episodic memories and summarise them using your LLM:

```python
from synapsemem.memory.memory_compressor import MemoryCompressor

compressor = MemoryCompressor(
    llm=my_llm,
    similarity_threshold=0.85,
    min_cluster_size=3,
)
report = compressor.run(storage=memory.storage, dry_run=False)
print(f"Compressed {report['compressed']} clusters into semantic memories")
```

### Async ingest (Celery + Redis)

```bash
# Start Redis
docker run -p 6379:6379 redis:7

# Start Celery worker
celery -A synapsemem.async_pipeline.celery_app worker --loglevel=info
```

```python
from synapsemem.async_pipeline import ingest_text_async

# Fire and forget — returns immediately
result = ingest_text_async.delay(
    text="I prefer dark mode in all my editors.",
    user_id="alice",
    storage_backend="sqlite",
)

# Poll for result
print(result.get(timeout=10))
```

### Multi-agent shared memory

```python
from synapsemem.memory.shared_memory import SharedMemoryStore

# All agents in the same workspace share this store
shared = SharedMemoryStore(
    workspace_id="team_alpha",
    db_path="synapsemem.db",
    conflict_strategy="anchor_weighted",  # or "last_write_wins" / "no_overwrite"
)

# Agent A writes a fact
shared.write_fact(
    {"subject": "user", "predicate": "prefers", "object": "python", "priority": 7},
    agent_id="agent_a",
)

# Agent B reads all shared facts
facts = shared.read_facts()

# Workspace stats
print(shared.workspace_stats())
```

### LangChain integration

```python
from synapsemem.integrations.langchain_integration import SynapseMemLangChainMemory
from langchain.chains import ConversationChain

lc_memory = SynapseMemLangChainMemory(
    synapse=memory,
    memory_key="chat_history",
)

chain = ConversationChain(llm=llm, memory=lc_memory)
chain.predict(input="What do I like to do?")
```

### CrewAI integration

```python
from synapsemem.integrations.crewai_integration import SynapseMemCrewAITool
from crewai import Agent

memory_tool = SynapseMemCrewAITool(synapse=memory)

agent = Agent(
    role="Research Analyst",
    goal="Answer questions using long-term memory",
    tools=[memory_tool],
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Agent / User input                   │
└───────────────────────────┬─────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    Intent Classifier       │  (chitchat → skip)
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    Triplet Extractor       │  text → (s, p, o)
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Ingest Consolidator      │  ADD / UPDATE / DELETE / NOOP
              └─────────────┬─────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │            Storage Layer             │
         │  SQLite │ Qdrant │ Chroma │ Memory   │
         └──────────────────┬──────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     Knowledge Graph        │  auto-built from triplets
              └─────────────┬─────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │         Hybrid Retriever             │
         │  semantic sim + decay + priority     │
         │  + anchor bonus + graph bonus        │
         └──────────────────┬──────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │      Prompt Builder        │  anchors + memories + query
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │           LLM             │
              └───────────────────────────┘

Offline (scheduled):
  Sleep Consolidator → merge + promote + prune
  Memory Compressor  → cluster + summarise → semantic facts
```

### Memory lifecycle

| Stage | What happens |
|---|---|
| **Ingest** | Text → triplets via pattern extraction |
| **Consolidate** | ADD new / UPDATE changed / DELETE removed / NOOP duplicate |
| **Store** | Saved as episodic memory with embedding + metadata |
| **Sleep** | Duplicates merged, repeated facts promoted to semantic, weak facts pruned |
| **Compress** | Similar episodic clusters summarised by LLM into single semantic facts |
| **Retrieve** | Hybrid scoring over all active memories, top-k returned |
| **Reinforce** | Accessed memories get reinforcement_count++ and decay reset |

### Retrieval scoring formula

```
final_score =
    semantic_similarity  × 0.38
  + priority_score       × 0.18
  + synaptic_strength    × 0.18
  + decay_score          × 0.12
  + anchor_bonus         × 0.07
  + graph_bonus          × 0.07
  + semantic_type_bonus  + source_count_bonus
```

---

## API Reference

Start the dashboard API:

```bash
uvicorn synapsemem.dashboards.api:app --reload
# Docs at http://localhost:8000/docs
```

### Memory endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/memory/ingest` | Synchronous ingest |
| POST | `/memory/ingest/async` | Async ingest via Celery (returns task_id) |
| POST | `/memory/ingest/batch/async` | Batch async ingest (up to 500 texts) |
| POST | `/memory/retrieve` | Retrieve top-k memories for a query |
| GET | `/memory/all` | List all active memories |
| GET | `/memory/all-records` | List all records including merged/pruned |
| GET | `/memory/stats` | Memory counts by type and status |
| POST | `/memory/sleep` | Run sleep consolidation |
| POST | `/memory/compress` | Run memory compression pass |
| POST | `/memory/reset` | Reset all memory for current scope |
| DELETE | `/memory/topic/{topic}` | Delete all memories for a topic |

### Async task endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/tasks/{task_id}` | Poll Celery task status |

### Anchor endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/anchors/add` | Add a pinned fact |
| GET | `/anchors` | List all anchors |

### Graph endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/graph/facts/{entity}` | All facts about an entity |
| GET | `/graph/related/{entity}` | Related entities (max_depth hops) |
| GET | `/graph/path` | Shortest path between two entities |

### Shared memory endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/shared/{workspace_id}/write` | Write a fact to shared workspace |
| GET | `/shared/{workspace_id}/facts` | Read all shared facts |
| GET | `/shared/{workspace_id}/stats` | Workspace statistics |
| GET | `/shared/{workspace_id}/agent/{agent_id}` | Facts by a specific agent |
| DELETE | `/shared/{workspace_id}/fact` | Soft-delete a shared fact |

---

## SynapseMem vs Mem0

| Feature | SynapseMem | Mem0 |
|---|---|---|
| **Core philosophy** | Bio-inspired cognitive architecture | Scalable memory-centric personalization |
| **Memory structure** | Knowledge graph + triplets | Vector-based + optional graph |
| **Memory types** | Episodic → Semantic lifecycle | Flat facts with categories |
| **Decay & reinforcement** | Time-aware synaptic strength scoring | Not natively supported |
| **Sleep consolidation** | Merge + promote + prune offline pass | Not natively supported |
| **Memory compression** | LLM-powered semantic clustering | Not natively supported |
| **Storage backends** | Memory, SQLite, Qdrant, Chroma | 24+ DB integrations |
| **Multi-agent memory** | Shared workspace + conflict resolution | Via scoped user/agent IDs |
| **Async pipeline** | Celery + Redis + intent classification | Managed cloud platform |
| **Framework integrations** | LangChain, CrewAI | LangChain, LlamaIndex, and more |
| **Self-hosted** | Yes — fully local, no cloud required | Yes + managed cloud option |
| **Best for** | Agents needing bio-inspired, evolving memory with graph reasoning | Production apps needing broad DB support and managed hosting |

---

## Project Structure

```
synapsemem/
├── manager.py                  # Main SynapseMemory class
├── config.py
├── memory/
│   ├── base_storage.py         # Storage interface contract
│   ├── storage.py              # In-memory backend
│   ├── sqlite_storage.py       # SQLite backend
│   ├── qdrant_storage.py       # Qdrant backend
│   ├── chroma_storage.py       # Chroma backend
│   ├── shared_memory.py        # Multi-agent shared workspace
│   ├── extractor.py            # Text → triplets
│   ├── ingest_consolidator.py  # ADD/UPDATE/DELETE/NOOP logic
│   ├── sleep_consolidator.py   # Offline consolidation
│   ├── memory_compressor.py    # LLM-powered compression
│   ├── intent_classifier.py    # Pre-ingest intent detection
│   ├── retriever.py            # Hybrid retrieval engine
│   ├── decay.py                # Synaptic decay + strength
│   └── anchors.py              # Pinned facts
├── graph/
│   ├── graph_builder.py        # Knowledge graph
│   ├── query_engine.py         # Graph queries
│   └── relationship_rules.py
├── prompt/
│   ├── builder.py              # Prompt assembly
│   └── templates.py
├── async_pipeline/             # Celery tasks
│   ├── celery_app.py
│   ├── tasks.py
│   └── beat_schedule.py
├── integrations/               # Framework shims
│   ├── langchain_integration.py
│   └── crewai_integration.py
├── dashboards/
│   └── api.py                  # FastAPI endpoints
├── cli/
│   └── synapsemem_cli.py
└── utils/
    ├── embeddings.py
    ├── scorer.py
    ├── tokenizer.py
    └── logging.py

benchmarks/                     # Performance benchmarks
tests/                          # 82 tests, 0 failures
examples/
    chatbot_demo.py
```

---

## Benchmarks

Run the full benchmark suite:

```bash
python -m benchmarks.run_all
```

Individual benchmarks:

```bash
python benchmarks/benchmark_ingest.py
python benchmarks/benchmark_retrieval.py
python benchmarks/benchmark_prompt.py
python benchmarks/benchmark_quality.py
python benchmarks/benchmark_sleep.py
```

What each measures:

| Benchmark | Metric |
|---|---|
| Ingest | Latency per ingestion (ms) |
| Retrieval | Query latency (ms) |
| Prompt | Token count and prompt size |
| Quality | Retrieval accuracy on test queries |
| Sleep | Promoted / merged / pruned counts |

---

## Testing

```bash
# Install dev dependencies
pip install "synapsemem[dev]"

# Run all tests
python -m pytest tests/ -v

# Run only Phase 3 tests
python -m pytest tests/ -v -k "phase3"

# Run with optional dep tests (install first)
pip install qdrant-client chromadb langchain langchain-core
python -m pytest tests/ -v
```

Current status: **82 passed, 44 skipped** (skipped = optional deps not installed).

---

## Running the Dashboard

```bash
pip install "synapsemem[dashboard]"
uvicorn synapsemem.dashboards.api:app --reload --port 8000
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYNAPSEMEM_BROKER_URL` | `redis://localhost:6379/0` | Celery broker URL |
| `SYNAPSEMEM_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery result backend |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up the dev environment, run tests, and submit pull requests.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Shubham Raj** — AI/ML engineer focused on LLM systems, RAG architectures, and agentic workflows.