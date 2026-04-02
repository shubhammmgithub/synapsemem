"""Celery application factory for SynapseMem async pipeline.

Usage:
    # Start a worker (from project root):
    celery -A synapsemem.async_pipeline.celery_app worker --loglevel=info

    # Start Redis (Docker):
    docker run -p 6379:6379 redis:7

Requirements:
    pip install celery redis
"""

from __future__ import annotations

from celery import Celery

# ── broker / backend ────────────────────────────────────────────────────
# Override via environment variables in production:
#   SYNAPSEMEM_BROKER_URL=redis://user:pass@host:6379/0
#   SYNAPSEMEM_RESULT_BACKEND=redis://user:pass@host:6379/1

import os

BROKER_URL = os.environ.get("SYNAPSEMEM_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("SYNAPSEMEM_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "synapsemem",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_acks_late=True,           # ack only after the task succeeds
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker process
    # Result TTL
    result_expires=3600,           # keep results for 1 hour
    # Routing
    task_routes={
        "synapsemem.async_pipeline.tasks.ingest_text_async": {"queue": "ingest"},
        "synapsemem.async_pipeline.tasks.batch_ingest_async": {"queue": "ingest"},
        "synapsemem.async_pipeline.tasks.sleep_consolidate_async": {"queue": "maintenance"},
    },
    # Timezone
    timezone="UTC",
    enable_utc=True,
)