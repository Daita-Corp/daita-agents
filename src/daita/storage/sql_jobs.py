"""Checklist-job persistence for the SQL state store.

Every function runs inside a caller-owned transaction or read connection;
``SQLStateStore`` owns the transaction boundary.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from ..distribution.models import Delivery
from ..jobs.models import (
    UNCERTAIN_WRITE_QUESTION,
    Job,
    JobState,
)
from .errors import JobStoreConflictError
from .sql_connection import SQLConnection
from .sql_routines import _datetime_us, _insert_delivery
from .sqlite_codecs.jobs import decode_job, encode_job


def load_job(connection: SQLConnection, agent_id: str, job_id: str) -> Job | None:
    row = connection.execute(
        "SELECT data FROM jobs WHERE agent_id = ? AND job_id = ?",
        (agent_id, job_id),
    ).fetchone()
    if row is None:
        return None
    job = decode_job(row[0])
    if job.agent_id != agent_id or job.job_id != job_id:
        raise ValueError("stored job identity differs from its row")
    return job


def list_jobs(
    connection: SQLConnection,
    agent_id: str,
    *,
    states: frozenset[JobState],
    limit: int,
) -> tuple[Job, ...]:
    values = sorted(item.value for item in states)
    clause = "" if not values else f" AND state IN ({', '.join('?' for _ in values)})"
    rows = connection.execute(
        "SELECT data FROM jobs WHERE agent_id = ?"
        + clause
        + " ORDER BY updated_at_us DESC, job_id LIMIT ?",
        (agent_id, *values, limit),
    ).fetchall()
    return tuple(decode_job(row[0]) for row in rows)


def insert_job(connection: SQLConnection, job: Job) -> Job:
    if job.revision != 1:
        raise ValueError("a new job starts at revision one")
    connection.execute(
        "INSERT INTO jobs(agent_id, job_id, conversation_id, state, revision, "
        "created_at_us, updated_at_us, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job.agent_id,
            job.job_id,
            job.conversation_id,
            job.state.value,
            job.revision,
            _datetime_us(job.created_at),
            _datetime_us(job.updated_at),
            encode_job(job),
        ),
    )
    return job


def replace_job(connection: SQLConnection, job: Job) -> Job:
    """Write ``job`` over the revision before it, or raise a conflict."""

    cursor = connection.execute(
        "UPDATE jobs SET state = ?, revision = ?, updated_at_us = ?, data = ? "
        "WHERE agent_id = ? AND job_id = ? AND revision = ? AND conversation_id = ?",
        (
            job.state.value,
            job.revision,
            _datetime_us(job.updated_at),
            encode_job(job),
            job.agent_id,
            job.job_id,
            job.revision - 1,
            job.conversation_id,
        ),
    )
    if cursor.rowcount != 1:
        raise JobStoreConflictError("the job changed since it was read")
    return job


def finish_job(connection: SQLConnection, job: Job, delivery: Delivery | None) -> Job:
    replace_job(connection, job)
    if delivery is not None:
        if delivery.subject_id != job.job_id or delivery.agent_id != job.agent_id:
            raise ValueError("job delivery belongs to another job")
        _insert_delivery(connection, delivery)
    return job


def wait_for_uncertain_write(
    connection: SQLConnection, agent_id: str, job_id: str, at: datetime
) -> None:
    """An uncertain receipt holds its job for a person, atomically with it."""

    job = load_job(connection, agent_id, job_id)
    if job is None or job.state is not JobState.RUNNING:
        return
    replace_job(
        connection,
        replace(
            job,
            state=JobState.WAITING_FOR_YOU,
            question=UNCERTAIN_WRITE_QUESTION,
            revision=job.revision + 1,
            updated_at=max(at, job.updated_at),
        ),
    )


def release_uncertain_write(
    connection: SQLConnection, agent_id: str, job_id: str, at: datetime
) -> None:
    """Resolving the job's last uncertain write lets it continue."""

    job = load_job(connection, agent_id, job_id)
    if (
        job is None
        or job.state is not JobState.WAITING_FOR_YOU
        or job.question != UNCERTAIN_WRITE_QUESTION
        or connection.execute(
            "SELECT 1 FROM effect_receipts "
            "WHERE agent_id = ? AND job_id = ? AND unresolved = 1",
            (agent_id, job_id),
        ).fetchone()
        is not None
    ):
        return
    replace_job(
        connection,
        replace(
            job,
            state=JobState.RUNNING,
            question=None,
            revision=job.revision + 1,
            updated_at=max(at, job.updated_at),
        ),
    )


__all__ = [
    "finish_job",
    "insert_job",
    "list_jobs",
    "load_job",
    "release_uncertain_write",
    "replace_job",
    "wait_for_uncertain_write",
]
