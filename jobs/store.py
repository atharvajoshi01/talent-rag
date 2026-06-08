"""SQLite-backed job store.

A deliberately small persistent queue. The schema fits the lifecycle we
care about (pending -> running -> succeeded/failed) and exposes a single
atomic claim primitive so concurrent workers cannot pick up the same job.

SQLite was chosen because it is durable across process restarts, requires
no external service, and runs anywhere the rest of the project runs. For
a production-grade deployment swap this out for SQS + DynamoDB; the
interface is intentionally narrow to make that switch painless.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .models import Job, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    job_type   TEXT NOT NULL,
    status     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    result     TEXT,
    error      TEXT,
    progress   TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_jobs_status_created
    ON jobs (status, created_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        job_type=row["job_type"],
        status=JobStatus(row["status"]),
        payload=json.loads(row["payload"]) if row["payload"] else {},
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        progress=row["progress"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class JobStore:
    """Tiny SQLite job store with an atomic claim_next primitive.

    Instances are safe to share across threads thanks to a per-instance
    lock around all write paths. A separate sqlite3 connection per call
    keeps things simple and avoids cross-thread cursor surprises.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def create(self, *, job_type: str, payload: dict) -> Job:
        """Insert a new job in the pending state and return it."""
        job_id = uuid.uuid4().hex
        now = _utcnow_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs ("
                " job_id, job_type, status, payload,"
                " result, error, progress, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?)",
                (
                    job_id,
                    job_type,
                    JobStatus.PENDING.value,
                    json.dumps(payload),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row)

    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_pending(self, *, limit: int = 100) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs"
                " WHERE status = ?"
                " ORDER BY created_at ASC"
                " LIMIT ?",
                (JobStatus.PENDING.value, limit),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def claim_next(self, *, job_type: Optional[str] = None) -> Optional[Job]:
        """Atomically move the oldest pending job to running and return it.

        Returns None when no pending job matches. Two concurrent callers
        will see distinct jobs (or one of them will see None); they will
        never both receive the same job.
        """
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = (
                "SELECT * FROM jobs"
                " WHERE status = ?"
                + (" AND job_type = ?" if job_type else "")
                + " ORDER BY created_at ASC LIMIT 1"
            )
            params: tuple = (
                (JobStatus.PENDING.value, job_type)
                if job_type
                else (JobStatus.PENDING.value,)
            )
            row = conn.execute(query, params).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            now = _utcnow_iso()
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (JobStatus.RUNNING.value, now, row["job_id"]),
            )
            conn.execute("COMMIT")
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
        return _row_to_job(row)

    def set_progress(self, job_id: str, progress: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, updated_at = ? WHERE job_id = ?",
                (progress, _utcnow_iso(), job_id),
            )

    def mark_succeeded(self, job_id: str, result: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs"
                " SET status = ?, result = ?, error = NULL, updated_at = ?"
                " WHERE job_id = ?",
                (
                    JobStatus.SUCCEEDED.value,
                    json.dumps(result),
                    _utcnow_iso(),
                    job_id,
                ),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs"
                " SET status = ?, error = ?, result = NULL, updated_at = ?"
                " WHERE job_id = ?",
                (JobStatus.FAILED.value, error, _utcnow_iso(), job_id),
            )

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
