"""Job and JobStatus types for the async index-build queue."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for a queued job.

    pending   -> created, not yet claimed by a worker
    running   -> claimed by a worker, in progress
    succeeded -> worker finished without error, result is populated
    failed    -> worker finished with error, error message is populated
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(BaseModel):
    """An async unit of work.

    Jobs carry a freeform payload that the worker interprets according to
    `job_type`. For the index-build path, payload contains the candidates
    and roles paths plus index destination. Status moves forward only;
    terminal states (succeeded, failed) are never reopened.
    """

    job_id: str = Field(description="Stable identifier, UUID4 hex string")
    job_type: str = Field(
        default="index_build",
        description="Discriminator that tells the worker how to handle this job",
    )
    status: JobStatus = Field(default=JobStatus.PENDING)
    payload: dict = Field(
        default_factory=dict,
        description="Inputs for the worker, e.g. candidates_path",
    )
    result: Optional[dict] = Field(
        default=None,
        description="Populated when status == succeeded",
    )
    error: Optional[str] = Field(
        default=None,
        description="Populated when status == failed",
    )
    progress: Optional[str] = Field(
        default=None,
        description="Optional human-readable progress note set by the worker",
    )
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)
