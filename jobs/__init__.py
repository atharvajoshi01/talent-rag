"""Async job queue for long-running index-build operations.

The talent_rag.jobs module decouples the HTTP-facing API from the heavy
work of embedding generation, FAISS index construction, and on-disk
persistence. The API enqueues a Job, returns immediately with a job_id,
and a background worker drains the queue and writes results back into the
store. Callers poll the job endpoint for completion.

The pattern mirrors the Atlassian Open Service Broker architecture
(FastAPI -> queue -> worker -> state store) at a single-process scale
suitable for an open source demo. The store is SQLite for zero external
dependencies; the worker is an asyncio task started in the FastAPI
lifespan handler.
"""

from .models import Job, JobStatus
from .store import JobStore
from .worker import IndexBuildWorker

__all__ = ["Job", "JobStatus", "JobStore", "IndexBuildWorker"]
