"""Deterministic multi-agent Stage B queue and lifecycle soak benchmark.

Set ``DAITA_RUN_STAGE_B_CONCURRENCY_SOAK=1`` to run this intentionally heavier
SQLite lifecycle benchmark. It makes no model or external network calls.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from daita.jobs.models import (
    JobAttemptStatus,
    JobDesiredState,
    JobExecutionMode,
    JobResourceBinding,
    JobRun,
    JobSpecification,
    JobStatus,
    MAX_QUEUED_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_AGENT,
    MAX_RUNNING_JOBS_PER_SOURCE,
)
from daita.llm.models import ModelSensitivity
from daita.storage.sqlite import SQLiteStateStore

_AUTHORIZATION = "DAITA_RUN_STAGE_B_CONCURRENCY_SOAK"
_AGENT_COUNT = 4
_SOURCE_COUNT = 4

pytestmark = [
    pytest.mark.integration,
    pytest.mark.acceptance,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=f"set {_AUTHORIZATION}=1 to run the deterministic lifecycle soak",
    ),
]


def _stored_job(
    agent_index: int,
    job_index: int,
    *,
    now: datetime,
) -> JobRun:
    agent_id = f"benchmark-agent-{agent_index}"
    source_index = job_index % _SOURCE_COUNT
    source_id = f"benchmark-source-{source_index}"
    resource_id = f"benchmark-resource-{source_index}-{job_index:03d}"
    resource_revision = "sha256:" + sha256(resource_id.encode("utf-8")).hexdigest()
    contract_digest = "sha256:" + sha256(b"benchmark-capability-v1").hexdigest()
    created_at = now + timedelta(microseconds=job_index)
    specification = JobSpecification(
        job_kind="data_profile",
        arguments={"resource_ids": (resource_id,), "sample_rows": 5},
        resource_bindings=(
            JobResourceBinding(
                source_id=source_id,
                source_revision="benchmark-source-revision-v1",
                resource_id=resource_id,
                resource_revision=resource_revision,
                adapter_id="sqlite",
                sensitivity=ModelSensitivity.INTERNAL,
            ),
        ),
        execution_capability_id="jobs.data_profile.execute",
        execution_contract_digest=contract_digest,
        execution_mode=JobExecutionMode.DAITA,
        sensitivity=ModelSensitivity.INTERNAL,
        deadline_at=now + timedelta(minutes=30),
        max_wall_time_seconds=60,
    )
    return JobRun(
        job_id=f"job-agent-{agent_index}-{job_index:03d}",
        agent_id=agent_id,
        conversation_id=f"conversation-{agent_index}-{job_index:03d}",
        origin_run_id=f"run-origin-{agent_index}-{job_index:03d}",
        origin_call_id=f"call-origin-{agent_index}-{job_index:03d}",
        specification=specification,
        specification_digest=specification.digest,
        status=JobStatus.QUEUED,
        desired_state=JobDesiredState.RUN,
        created_at=created_at,
        updated_at=created_at,
    )


async def test_concurrent_queue_admission_is_atomic_at_the_exact_bound(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    store = await SQLiteStateStore.open(tmp_path / "queue-admission-soak.sqlite")
    try:
        candidates = tuple(
            _stored_job(0, index, now=now)
            for index in range(MAX_QUEUED_JOBS_PER_AGENT + 12)
        )
        outcomes = await asyncio.gather(
            *(store.admit_job(job) for job in candidates),
            return_exceptions=True,
        )
        admitted = tuple(item for item in outcomes if isinstance(item, JobRun))
        rejected = tuple(item for item in outcomes if isinstance(item, Exception))
        assert len(admitted) == MAX_QUEUED_JOBS_PER_AGENT
        assert len(rejected) == 12
        assert all(
            isinstance(item, ValueError) and str(item) == "job_queue_limit_exceeded"
            for item in rejected
        )
        persisted = await store.list_jobs(
            "benchmark-agent-0",
            limit=MAX_QUEUED_JOBS_PER_AGENT,
        )
        assert len(persisted) == MAX_QUEUED_JOBS_PER_AGENT
        assert len({item.job_id for item in persisted}) == len(persisted)
    finally:
        await store.close()


async def test_multi_agent_claim_cancel_recovery_and_fencing_soak(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    path = tmp_path / "multi-agent-lifecycle-soak.sqlite"
    store = await SQLiteStateStore.open(path)
    agent_ids = tuple(f"benchmark-agent-{index}" for index in range(_AGENT_COUNT))
    jobs = tuple(
        _stored_job(agent_index, job_index, now=now)
        for agent_index in range(_AGENT_COUNT)
        for job_index in range(MAX_QUEUED_JOBS_PER_AGENT)
    )
    for offset in range(0, len(jobs), 16):
        admitted = await asyncio.gather(
            *(store.admit_job(job) for job in jobs[offset : offset + 16])
        )
        assert len(admitted) == len(jobs[offset : offset + 16])

    wrong_owner = await store.request_job_cancel(
        "benchmark-agent-1",
        "job-agent-0-000",
        requested_at=now + timedelta(milliseconds=1),
    )
    assert wrong_owner is None

    claim_serial = 0
    stale_rejections = 0
    cancelled_jobs = 0
    for cycle in range(64):
        claimed_at = now + timedelta(seconds=cycle * 4 + 1)
        claim_coroutines = []
        for agent_id in agent_ids:
            for _ in range(MAX_RUNNING_JOBS_PER_AGENT * 2):
                claim_serial += 1
                suffix = f"{claim_serial:032x}"
                claim_coroutines.append(
                    store.claim_next_job(
                        agent_id,
                        claim_token=f"claim-{claim_serial}",
                        execution_run_id=f"run-{suffix}",
                        reserved_artifact_id=f"artifact-{suffix}",
                        claimed_at=claimed_at,
                        lease_seconds=60,
                    )
                )
        claim_results = await asyncio.gather(*claim_coroutines)
        claimed = tuple(item for item in claim_results if item is not None)
        if not claimed:
            break

        for agent_id in agent_ids:
            running = await store.list_jobs(
                agent_id,
                statuses=frozenset({JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED}),
                limit=MAX_RUNNING_JOBS_PER_AGENT,
            )
            assert len(running) <= MAX_RUNNING_JOBS_PER_AGENT
            per_source = Counter(
                source_id for job in running for source_id in job.source_ids
            )
            assert all(
                count <= MAX_RUNNING_JOBS_PER_SOURCE for count in per_source.values()
            )

        async def settle(job: JobRun) -> tuple[int, int]:
            attempt = job.current_attempt
            assert attempt is not None
            job_index = int(job.job_id.rsplit("-", 1)[1])
            transition_at = claimed_at + timedelta(seconds=1)
            completed_at = claimed_at + timedelta(seconds=2)
            if job_index % 17 == 0 and attempt.number == 1:
                recovered = await store.recover_stale_job(
                    job.agent_id,
                    job.job_id,
                    recovered_at=transition_at,
                    restart_safe=True,
                )
                assert recovered is not None
                assert recovered.status is JobStatus.QUEUED
                assert recovered.attempts[-1].status is JobAttemptStatus.FENCED
                stale = await store.finalize_job_attempt(
                    job.agent_id,
                    job.job_id,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    attempt_status=JobAttemptStatus.FAILED,
                    completed_at=completed_at,
                    failure_code="stale_worker_must_lose",
                )
                assert stale is None
                return 1, 0
            if job_index % 11 == 0:
                requested = await store.request_job_cancel(
                    job.agent_id,
                    job.job_id,
                    requested_at=transition_at,
                )
                assert requested is not None
                assert requested.status is JobStatus.CANCEL_REQUESTED
                settled = await store.finalize_job_attempt(
                    job.agent_id,
                    job.job_id,
                    claim_token=attempt.claim_token,
                    fencing_epoch=attempt.fencing_epoch,
                    attempt_status=JobAttemptStatus.CANCELLED,
                    completed_at=completed_at,
                )
                assert settled is not None
                assert settled.status is JobStatus.CANCELLED
                return 0, 1
            settled = await store.finalize_job_attempt(
                job.agent_id,
                job.job_id,
                claim_token=attempt.claim_token,
                fencing_epoch=attempt.fencing_epoch,
                attempt_status=JobAttemptStatus.FAILED,
                completed_at=completed_at,
                failure_code="benchmark_settled",
            )
            assert settled is not None
            assert settled.status is JobStatus.FAILED
            return 0, 0

        settlements = await asyncio.gather(*(settle(job) for job in claimed))
        stale_rejections += sum(item[0] for item in settlements)
        cancelled_jobs += sum(item[1] for item in settlements)
    else:
        raise AssertionError("lifecycle soak did not drain within its cycle bound")

    assert stale_rejections == _AGENT_COUNT * 3
    assert cancelled_jobs == _AGENT_COUNT * 5
    for agent_id in agent_ids:
        persisted = await store.list_jobs(
            agent_id,
            limit=MAX_QUEUED_JOBS_PER_AGENT,
        )
        assert len(persisted) == MAX_QUEUED_JOBS_PER_AGENT
        assert all(item.terminal for item in persisted)
        assert all(len(item.attempts) <= 2 for item in persisted)
        statuses = Counter(item.status for item in persisted)
        assert statuses[JobStatus.CANCELLED] == 5
        assert statuses[JobStatus.FAILED] == MAX_QUEUED_JOBS_PER_AGENT - 5
    await store.close()

    reopened = await SQLiteStateStore.open(path)
    try:
        for agent_id in agent_ids:
            persisted = await reopened.list_jobs(
                agent_id,
                limit=MAX_QUEUED_JOBS_PER_AGENT,
            )
            assert len(persisted) == MAX_QUEUED_JOBS_PER_AGENT
            assert all(item.terminal for item in persisted)
            assert all(
                item.specification_digest == item.specification.digest
                for item in persisted
            )
    finally:
        await reopened.close()
