"""Tests for the async job queue (talent_rag.jobs).

Covers the SQLite-backed JobStore (create, claim_next, status transitions,
list_pending, count_by_status) and the IndexBuildWorker (happy path,
handler dispatch, failure capture).

These tests exercise the queue independently of FAISS or any embedding
backend by registering a fake handler. That keeps the test suite fast
(under 1s) and avoids pulling sentence-transformers at import time.
"""

from __future__ import annotations

import asyncio

import pytest

from talent_rag.jobs import IndexBuildWorker, Job, JobStatus, JobStore


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite")


class TestJobStoreCRUD:
    def test_create_returns_pending_job(self, store):
        job = store.create(job_type="index_build", payload={"x": 1})
        assert job.status == JobStatus.PENDING
        assert job.payload == {"x": 1}
        assert job.job_type == "index_build"
        assert job.job_id  # non-empty

    def test_get_round_trips_job(self, store):
        created = store.create(job_type="index_build", payload={"k": "v"})
        fetched = store.get(created.job_id)
        assert fetched is not None
        assert fetched.job_id == created.job_id
        assert fetched.payload == {"k": "v"}

    def test_get_unknown_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_list_pending_returns_oldest_first(self, store):
        a = store.create(job_type="index_build", payload={})
        b = store.create(job_type="index_build", payload={})
        pending = store.list_pending()
        assert [j.job_id for j in pending] == [a.job_id, b.job_id]

    def test_count_by_status_groups(self, store):
        a = store.create(job_type="index_build", payload={})
        b = store.create(job_type="index_build", payload={})
        store.mark_succeeded(a.job_id, {"ok": True})
        counts = store.count_by_status()
        assert counts.get("pending") == 1
        assert counts.get("succeeded") == 1
        assert b.status == JobStatus.PENDING


class TestClaimNext:
    def test_claim_moves_to_running(self, store):
        created = store.create(job_type="index_build", payload={})
        claimed = store.claim_next(job_type="index_build")
        assert claimed is not None
        assert claimed.job_id == created.job_id
        assert claimed.status == JobStatus.RUNNING

    def test_claim_returns_none_when_empty(self, store):
        assert store.claim_next(job_type="index_build") is None

    def test_claim_skips_non_matching_job_type(self, store):
        store.create(job_type="other_type", payload={})
        assert store.claim_next(job_type="index_build") is None

    def test_two_claims_get_distinct_jobs(self, store):
        a = store.create(job_type="index_build", payload={})
        b = store.create(job_type="index_build", payload={})
        first = store.claim_next(job_type="index_build")
        second = store.claim_next(job_type="index_build")
        assert first is not None and second is not None
        assert {first.job_id, second.job_id} == {a.job_id, b.job_id}


class TestStatusTransitions:
    def test_mark_succeeded_writes_result(self, store):
        job = store.create(job_type="index_build", payload={})
        store.mark_succeeded(job.job_id, {"docs": 42})
        fetched = store.get(job.job_id)
        assert fetched.status == JobStatus.SUCCEEDED
        assert fetched.result == {"docs": 42}
        assert fetched.error is None

    def test_mark_failed_writes_error(self, store):
        job = store.create(job_type="index_build", payload={})
        store.mark_failed(job.job_id, "boom")
        fetched = store.get(job.job_id)
        assert fetched.status == JobStatus.FAILED
        assert fetched.error == "boom"
        assert fetched.result is None

    def test_progress_updates_visible(self, store):
        job = store.create(job_type="index_build", payload={})
        store.set_progress(job.job_id, "step 1 of 3")
        fetched = store.get(job.job_id)
        assert fetched.progress == "step 1 of 3"

    def test_is_terminal_only_for_terminal_states(self):
        assert not Job(job_id="x", status=JobStatus.PENDING).is_terminal()
        assert not Job(job_id="x", status=JobStatus.RUNNING).is_terminal()
        assert Job(job_id="x", status=JobStatus.SUCCEEDED).is_terminal()
        assert Job(job_id="x", status=JobStatus.FAILED).is_terminal()


@pytest.mark.asyncio
class TestWorkerLifecycle:
    async def test_worker_processes_pending_job_via_custom_handler(self, store):
        worker = IndexBuildWorker(store, poll_interval_seconds=0.05,
                                  job_type="echo")
        seen = {}

        async def handler(job, s):
            seen["payload"] = job.payload
            s.set_progress(job.job_id, "almost done")
            return {"echoed": job.payload}

        worker.register_handler("echo", handler)
        await worker.start()
        try:
            job = store.create(job_type="echo", payload={"hello": "world"})
            for _ in range(40):
                await asyncio.sleep(0.05)
                current = store.get(job.job_id)
                if current.status == JobStatus.SUCCEEDED:
                    break
            else:
                pytest.fail("job did not reach SUCCEEDED in time")
        finally:
            await worker.stop()

        assert seen["payload"] == {"hello": "world"}
        final = store.get(job.job_id)
        assert final.status == JobStatus.SUCCEEDED
        assert final.result == {"echoed": {"hello": "world"}}
        assert final.progress == "almost done"

    async def test_worker_records_handler_exception_as_failed(self, store):
        worker = IndexBuildWorker(store, poll_interval_seconds=0.05,
                                  job_type="broken")

        async def handler(job, s):
            raise RuntimeError("kaboom")

        worker.register_handler("broken", handler)
        await worker.start()
        try:
            job = store.create(job_type="broken", payload={})
            for _ in range(40):
                await asyncio.sleep(0.05)
                current = store.get(job.job_id)
                if current.status == JobStatus.FAILED:
                    break
            else:
                pytest.fail("job did not reach FAILED in time")
        finally:
            await worker.stop()

        final = store.get(job.job_id)
        assert final.status == JobStatus.FAILED
        assert "kaboom" in (final.error or "")
        assert "RuntimeError" in (final.error or "")

    async def test_unknown_job_type_marked_failed(self, store):
        worker = IndexBuildWorker(store, poll_interval_seconds=0.05,
                                  job_type="mystery")
        await worker.start()
        try:
            job = store.create(job_type="mystery", payload={})
            for _ in range(40):
                await asyncio.sleep(0.05)
                current = store.get(job.job_id)
                if current.status == JobStatus.FAILED:
                    break
            else:
                pytest.fail("unknown-type job was not marked FAILED")
        finally:
            await worker.stop()

        final = store.get(job.job_id)
        assert final.status == JobStatus.FAILED
        assert "no handler" in (final.error or "")
