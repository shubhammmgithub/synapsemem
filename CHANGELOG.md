# Changelog

All notable changes to SynapseMem are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.0] — Phase 3: Agent-Native + Scalable

### Added
- **Vector DB adapters** — Qdrant and Chroma backends, both implementing `BaseMemoryStorage` interface
- **`BaseMemoryStorage`** — abstract base class enforcing the storage interface contract across all backends
- **Async ingest pipeline** — Celery + Redis task queue with `ingest_text_async`, `batch_ingest_async`, `sleep_consolidate_async`
- **Intent classifier** — pre-ingest classification (fact / preference / task / tool_result / delete / chitchat) with priority boosting
- **Multi-agent shared memory** — `SharedMemoryStore` with workspace scoping and three conflict resolution strategies: `last_write_wins`, `no_overwrite`, `anchor_weighted`
- **Memory compression** — `MemoryCompressor` clusters similar episodic memories by embedding similarity and summarises them using an LLM
- **LangChain integration** — `SynapseMemLangChainMemory` (drop-in `BaseMemory`) and `SynapseMemTool`
- **CrewAI integration** — `SynapseMemCrewAITool` and `SynapseMemCrewStorage`
- **Phase 3 API endpoints** — async ingest, shared memory CRUD, compression trigger, task polling
- **Celery Beat schedule** — nightly sleep consolidation via `beat_schedule.py`
- **82 tests** passing across all phases (44 skipped for optional deps)
- `pyproject.toml` optional extras: `vector`, `async`, `langchain`, `crewai`, `all`

### Changed
- `manager.py` updated with new constructor params: `qdrant_url`, `qdrant_api_key`, `chroma_persist_directory`, `chroma_host`, `chroma_port`
- `_build_storage()` now routes to Qdrant and Chroma in addition to memory/sqlite
- Version bumped to `0.3.0`, dev status to `4 - Beta`

---

## [0.2.0] — Phase 2: Memory Evolution

### Added
- **Sleep consolidator** — offline pass that merges duplicates, promotes repeated episodic facts to semantic memory, and prunes weak/stale memories
- **SQLite storage** — persistent backend with full scope support (`user_id`, `agent_id`, `session_id`) and migration-safe schema
- **FastAPI dashboard** — REST API with endpoints for ingest, retrieval, sleep consolidation, anchors, graph queries, and memory stats
- **Promotion pipeline** — episodic → semantic memory promotion based on support count and priority thresholds
- **Dry-run mode** for sleep consolidation — preview changes before committing
- **Sleep benchmark** — measures promoted / merged / pruned counts

### Changed
- Retrieval engine updated with `semantic_memory_bonus` and `source_count_bonus` weightings
- `SynapseMemory` now accepts `storage_backend`, `sqlite_db_path`, `user_id`, `agent_id`, `session_id` constructor params

---

## [0.1.0] — Phase 1: Core Memory Engine

### Added
- **Triplet extractor** — rule-based extraction of `(subject, predicate, object)` facts from raw text
- **In-memory storage** — fast in-process storage for testing and prototyping
- **Ingest consolidator** — ADD / UPDATE / DELETE / NOOP decision pipeline
- **Hybrid retrieval engine** — scoring via semantic similarity, priority, decay, synaptic strength, anchor bonus, graph bonus
- **Decay system** — `compute_decay_score` and `compute_synaptic_strength` with reinforcement-based slow decay
- **Knowledge graph** — auto-built from stored triplets with `facts_about`, `related_entities`, `find_path`
- **Anchor manager** — pinned facts that always bias retrieval and prompt building
- **Prompt builder** — assembles LLM context from anchors + retrieved memories + query
- **`SynapseMemory` manager** — unified API: `ingest`, `retrieve`, `build_prompt`, `chat`, `sleep_consolidate`
- **CLI** — `synapsemem` command-line interface
- **Benchmark suite** — ingest, retrieval, prompt size, quality benchmarks
- **Initial test suite** — extractor, retrieval, decay, prompt builder, end-to-end tests