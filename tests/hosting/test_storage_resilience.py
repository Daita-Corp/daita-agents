"""Real composition remains observable and safe during storage failures."""

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from daita import Agent
from daita.storage.errors import (
    StorageCommitUnknownError,
    StorageOwnershipLostError,
    StorageUnavailableError,
)
from daita.storage.sqlite import SQLiteStateStore
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name,operation",
    [("jobs", "list_stale_graph_attempts"), ("routines", "next_routine_deadline")],
)
async def test_supervisor_recovers_after_known_noncommit_failure(
    tmp_path, monkeypatch, name, operation
):
    agent = await Agent.create(
        "retry-storage", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls = 0
    retried = asyncio.Event()
    original = getattr(SQLiteStateStore, operation)

    async def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StorageUnavailableError("private driver detail must not escape")
        value = await original(self, *args, **kwargs)
        retried.set()
        return value

    try:
        monkeypatch.setattr(SQLiteStateStore, operation, fail_once)
        supervisor = (
            agent._embedded._job_supervisor
            if name == "jobs"
            else agent._embedded._routine_supervisor
        )
        supervisor.wake()
        await asyncio.wait_for(retried.wait(), 3)
        # Allow the successful poll to finish and publish its healthy status.
        for _ in range(100):
            status = next(
                item for item in agent.background_status() if item.name == name
            )
            if status.state == "running":
                break
            await asyncio.sleep(0.01)
        assert status.state == "running"
        assert status.failure_code is None
        assert calls >= 2
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "error_type", [StorageCommitUnknownError, StorageOwnershipLostError]
)
@pytest.mark.parametrize(
    "name,operation",
    [("jobs", "expire_due_graphs"), ("routines", "next_routine_deadline")],
)
async def test_supervisor_stops_on_unknown_commit_or_lost_ownership(
    tmp_path, monkeypatch, caplog, error_type, name, operation
):
    agent = await Agent.create(
        "stop-storage", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    calls = 0

    async def fail(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        raise error_type("secret connection string")

    try:
        monkeypatch.setattr(SQLiteStateStore, operation, fail)
        supervisor = (
            agent._embedded._job_supervisor
            if name == "jobs"
            else agent._embedded._routine_supervisor
        )
        supervisor.wake()
        assert supervisor._driver is not None
        await asyncio.wait_for(asyncio.shield(supervisor._driver), 2)
        assert supervisor.status.state == "failed"
        assert supervisor.status.failure_code == error_type.code
        supervisor.wake()
        await asyncio.sleep(0.05)
        assert calls == 1
        assert "secret connection string" not in caplog.text
    finally:
        await agent.close()


async def test_idle_graph_polling_is_bounded_and_wake_is_prompt(
    tmp_path: Path, monkeypatch
):
    agent = await Agent.create(
        "idle-storage", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    counts: Counter[str] = Counter()
    woke = asyncio.Event()
    original = SQLiteStateStore.list_stale_graph_attempts

    async def observe(self, *args, **kwargs):
        counts["poll"] += 1
        woke.set()
        return await original(self, *args, **kwargs)

    try:
        monkeypatch.setattr(SQLiteStateStore, "list_stale_graph_attempts", observe)
        await asyncio.sleep(1.1)
        assert 1 <= counts["poll"] <= 6
        woke.clear()
        agent._embedded._job_supervisor.wake()
        await asyncio.wait_for(woke.wait(), 0.3)
    finally:
        await agent.close()
