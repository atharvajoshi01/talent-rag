"""Background async worker that drains the job store.

The worker is a single asyncio task scheduled by the FastAPI lifespan
handler. It polls the store at a fixed interval, claims the next pending
job, and dispatches by job_type to a handler. Handlers run the heavy
(CPU/IO bound) embedding and index-build work inside a default executor
so the asyncio event loop stays responsive to incoming HTTP traffic.

This is the worker stage of the API -> queue -> worker -> state store
pattern. The boundary between API and worker is the SQLite job store; no
in-process shared state is required between them.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .models import Job
from .store import JobStore

logger = logging.getLogger(__name__)

JobHandler = Callable[[Job, JobStore], Awaitable[dict]]


class IndexBuildWorker:
    """Asyncio worker that processes index_build jobs from a JobStore."""

    def __init__(
        self,
        store: JobStore,
        *,
        poll_interval_seconds: float = 0.5,
        job_type: str = "index_build",
    ):
        self._store = store
        self._poll = poll_interval_seconds
        self._job_type = job_type
        self._handlers: dict[str, JobHandler] = {
            "index_build": self._handle_index_build,
        }
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Add a new handler. Useful for tests and future job types."""
        self._handlers[job_type] = handler

    async def start(self) -> None:
        """Schedule the worker loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="talent-rag-worker")
        logger.info("Started IndexBuildWorker (poll=%ss)", self._poll)

    async def stop(self) -> None:
        """Ask the worker to stop and wait for the current iteration to finish."""
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
        logger.info("Stopped IndexBuildWorker")

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                loop = asyncio.get_event_loop()
                claimed = await loop.run_in_executor(
                    None,
                    lambda: self._store.claim_next(job_type=self._job_type),
                )
                if claimed is None:
                    await asyncio.sleep(self._poll)
                    continue
                await self._dispatch(claimed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker loop iteration crashed")
                await asyncio.sleep(self._poll)

    async def _dispatch(self, job: Job) -> None:
        handler = self._handlers.get(job.job_type)
        if handler is None:
            self._store.mark_failed(
                job.job_id, f"no handler registered for job_type={job.job_type!r}"
            )
            return
        try:
            result = await handler(job, self._store)
            self._store.mark_succeeded(job.job_id, result)
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            logger.error("Job %s failed: %s\n%s", job.job_id, exc, tb)
            self._store.mark_failed(job.job_id, f"{type(exc).__name__}: {exc}")

    async def _handle_index_build(self, job: Job, store: JobStore) -> dict:
        """Default handler: run IndexBuilder against payload paths.

        Payload schema (all paths optional, at least one of candidates_path /
        roles_path required):
          {
            "candidates_path": "data/candidates.json",
            "roles_path":      "data/roles.json",
            "index_dir":       "data/indices",
            "use_openai_embeddings": false
          }
        """
        from talent_rag.retrieval import IndexBuilder

        payload = job.payload or {}
        candidates_path = payload.get("candidates_path")
        roles_path = payload.get("roles_path")
        index_dir = payload.get("index_dir")
        use_openai_embeddings = bool(payload.get("use_openai_embeddings", False))

        if not candidates_path and not roles_path:
            raise ValueError(
                "payload must include candidates_path and/or roles_path"
            )

        store.set_progress(job.job_id, "initializing IndexBuilder")
        builder = IndexBuilder(use_openai_embeddings=use_openai_embeddings)

        store.set_progress(job.job_id, "building index")
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None,
            lambda: builder.build_index(
                candidates_path=candidates_path,
                roles_path=roles_path,
            ),
        )

        if index_dir:
            store.set_progress(job.job_id, "persisting index to disk")
            await loop.run_in_executor(
                None, builder.save_index, Path(index_dir)
            )

        return {
            "total_documents_indexed": len(vector_store),
            "index_dir": str(index_dir) if index_dir else None,
            "use_openai_embeddings": use_openai_embeddings,
        }
