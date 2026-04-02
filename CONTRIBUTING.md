# Contributing to SynapseMem

Thank you for your interest in contributing. This document covers everything you need to get started.

---

## Development setup

**Requirements:** Python 3.10+, Git

```bash
# 1. Fork and clone
git clone https://github.com/your-username/synapsemem.git
cd synapsemem

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify setup
python -m pytest tests/ -v
# Expected: 82 passed, 44 skipped
```

---

## Running tests

```bash
# All tests
python -m pytest tests/ -v

# Specific phase
python -m pytest tests/ -v -k "phase3"

# Single file
python -m pytest tests/test_retrieval.py -v

# With optional deps (install first)
pip install qdrant-client chromadb langchain langchain-core crewai
python -m pytest tests/ -v
```

All tests must pass before submitting a PR. Skipped tests (optional deps not installed) are fine.

---

## Project structure

The key files to understand before contributing:

- `synapsemem/manager.py` — the main `SynapseMemory` class, entry point for all features
- `synapsemem/memory/base_storage.py` — the storage interface every backend must implement
- `synapsemem/memory/ingest_consolidator.py` — ADD/UPDATE/DELETE/NOOP pipeline
- `synapsemem/memory/retriever.py` — hybrid retrieval scoring
- `synapsemem/dashboards/api.py` — FastAPI endpoints

---

## Adding a new storage backend

1. Create `synapsemem/memory/yourdb_storage.py`
2. Inherit from `BaseMemoryStorage` in `base_storage.py`
3. Implement all abstract methods (the interface is enforced at test time by `TestBaseStorageInterface`)
4. Add `"yourdb"` as a case in `manager.py` `_build_storage()`
5. Add to `pyproject.toml` optional deps
6. Write tests in `tests/test_phase3_step1_vector_adapters.py` following the existing Qdrant/Chroma pattern

---

## Submitting a pull request

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Write your code and tests
3. Run the full test suite: `python -m pytest tests/ -v`
4. Commit with a clear message: `git commit -m "feat: add Redis storage adapter"`
5. Push and open a PR against `main`

**PR checklist:**
- [ ] Tests pass
- [ ] New code has tests
- [ ] Docstrings added for public methods
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

---

## Reporting bugs

Open a GitHub issue with:
- Python version
- SynapseMem version (`pip show synapsemem`)
- Minimal reproduction script
- Full traceback

---

## Commit message convention

```
feat: add new feature
fix: fix a bug
test: add or fix tests
docs: documentation changes
refactor: code change with no feature/fix
chore: dependency updates, tooling
```