"""Celery Beat periodic schedule for SynapseMem maintenance tasks.

Start beat scheduler (from project root):
    celery -A synapsemem.async_pipeline.celery_app beat --loglevel=info

Or start worker + beat together (dev only):
    celery -A synapsemem.async_pipeline.celery_app worker --beat --loglevel=info
"""

from __future__ import annotations

from celery.schedules import crontab

from .celery_app import celery_app

celery_app.conf.beat_schedule = {
    # Run sleep consolidation every night at 2 AM UTC for default scope
    "nightly-sleep-consolidation": {
        "task": "synapsemem.async_pipeline.tasks.sleep_consolidate_async",
        "schedule": crontab(hour=2, minute=0),
        "kwargs": {
            "user_id": "default_user",
            "agent_id": "default_agent",
            "session_id": "default_session",
            "storage_backend": "sqlite",
            "dry_run": False,
        },
    },
}