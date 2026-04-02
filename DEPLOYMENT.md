# Deployment Guide

This guide covers two things: publishing SynapseMem to PyPI and deploying the dashboard API.

---

## Publishing to PyPI

### 1. Prerequisites

```bash
pip install build twine
```

You'll need a PyPI account at [pypi.org](https://pypi.org) and a TestPyPI account at [test.pypi.org](https://test.pypi.org).

Create an API token on both (Account Settings → API tokens → Add API token). Store it securely.

---

### 2. Version bump

Before every release, update the version in `pyproject.toml`:

```toml
[project]
version = "0.3.1"   # bump this
```

And add a section to `CHANGELOG.md` under `[Unreleased]`.

---

### 3. Build the package

From the project root:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build both sdist and wheel
python -m build
```

This creates two files in `dist/`:
- `synapsemem-0.3.0.tar.gz` — source distribution
- `synapsemem-0.3.0-py3-none-any.whl` — wheel

---

### 4. Test on TestPyPI first

Always publish to TestPyPI before the real PyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted, use `__token__` as username and your TestPyPI API token as password.

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ synapsemem
```

Run a quick sanity check:

```python
from synapsemem import SynapseMemory
m = SynapseMemory()
m.ingest("I like coffee.")
print(m.retrieve("What do I like?"))
```

---

### 5. Publish to PyPI

Once TestPyPI looks good:

```bash
python -m twine upload dist/*
```

Use `__token__` as username and your PyPI API token as password.

Verify the release:

```bash
pip install synapsemem==0.3.0
```

---

### 6. Tag the release on GitHub

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v0.3.0"
git tag v0.3.0
git push origin main --tags
```

Then create a GitHub Release from the tag — paste the CHANGELOG section as the release notes.

---

### Using a `.pypirc` file (optional — avoids typing credentials every time)

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

Set permissions: `chmod 600 ~/.pypirc` (Linux/macOS).

---

## Deploying the Dashboard API

### Local development

```bash
pip install "synapsemem[dashboard]"
uvicorn synapsemem.dashboards.api:app --reload --port 8000
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### With async pipeline (Celery + Redis)

```bash
# Terminal 1 — Redis
docker run -p 6379:6379 redis:7

# Terminal 2 — Celery worker
celery -A synapsemem.async_pipeline.celery_app worker --loglevel=info --queues=ingest,maintenance

# Terminal 3 — Celery Beat (scheduled jobs)
celery -A synapsemem.async_pipeline.celery_app beat --loglevel=info

# Terminal 4 — FastAPI
uvicorn synapsemem.dashboards.api:app --reload --port 8000
```

---

### Production deployment (Docker Compose)

Create `docker-compose.yml` in your project root:

```yaml
version: "3.9"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    command: uvicorn synapsemem.dashboards.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - SYNAPSEMEM_BROKER_URL=redis://redis:6379/0
      - SYNAPSEMEM_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    volumes:
      - ./data:/app/data

  worker:
    build: .
    command: celery -A synapsemem.async_pipeline.celery_app worker --loglevel=info
    environment:
      - SYNAPSEMEM_BROKER_URL=redis://redis:6379/0
      - SYNAPSEMEM_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis

  beat:
    build: .
    command: celery -A synapsemem.async_pipeline.celery_app beat --loglevel=info
    environment:
      - SYNAPSEMEM_BROKER_URL=redis://redis:6379/0
      - SYNAPSEMEM_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
```

Create a `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY synapsemem/ ./synapsemem/

RUN pip install ".[dashboard,async]"

EXPOSE 8000
```

Run:

```bash
docker compose up --build
```

---

### Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `SYNAPSEMEM_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `SYNAPSEMEM_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery result store |

---

## Release checklist

Before every release:

- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] Built with `python -m build`
- [ ] Tested on TestPyPI
- [ ] Published to PyPI
- [ ] Git tag created and pushed
- [ ] GitHub Release created with changelog notes