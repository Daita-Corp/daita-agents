"""Real composition remains observable and safe during storage failures."""

import asyncio
import logging
import threading
from collections import Counter
from pathlib import Path

import pytest

from daita import Agent
from daita.hosting.execution_governor import AdmissionClosedError
from daita.jobs.supervisor import JobSupervisor
from daita.routines.supervisor import RoutineSupervisor
from daita.storage import sqlite as sqlite_module
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


@pytest.mark.parametrize("name", ["jobs", "routines"])
async def test_host_admission_close_stops_supervisors_without_failure(
    tmp_path, monkeypatch, caplog, name
):
    caplog.set_level(logging.INFO)
    agent = await Agent.create(
        "orderly-close", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    embedded = agent._embedded
    supervisor: JobSupervisor | RoutineSupervisor
    try:
        if name == "jobs":
            supervisor = embedded._job_supervisor
            # The next graph poll is refused its storage permit by admission.
            await embedded._admission_coordinator.begin_draining()
        else:
            supervisor = embedded._routine_supervisor

            async def closed(self, *args, **kwargs):
                raise AdmissionClosedError("host admission is not open")

            # A routine poll that meets closed host admission.
            monkeypatch.setattr(SQLiteStateStore, "next_routine_deadline", closed)
        supervisor.wake()
        assert supervisor._driver is not None
        await asyncio.wait_for(asyncio.shield(supervisor._driver), 2)
        assert supervisor.status.state == "stopped"
        assert supervisor.status.failure_code is None
    finally:
        await agent.close()
    assert f"{name} supervisor stopped (host_closing)" in caplog.text
    assert "supervisor_failed" not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


async def test_cancelled_supervisor_read_releases_its_connection_before_close(
    tmp_path, monkeypatch
):
    """Reproduce the shutdown race: a supervisor is cancelled mid-read."""

    agent = await Agent.create(
        "release-on-close", root=tmp_path, workspace=workspace_for(tmp_path)
    )
    open_connections: set[int] = set()
    entered = threading.Event()
    release = threading.Event()
    armed = threading.Event()
    checkpoints: list[int] = []
    original_connect = sqlite_module._SQLiteDatabase.connect
    original_checkpoint = sqlite_module._checkpoint_wal_for_close

    class TrackedConnection(sqlite_module._SQLiteConnection):
        def execute(self, query, parameters=()):
            if armed.is_set():
                armed.clear()
                entered.set()
                release.wait(5)
            return super().execute(query, parameters)

        def close(self):
            open_connections.discard(id(self))
            super().close()

        def __exit__(self, exc_type, exc, traceback):
            open_connections.discard(id(self))
            return super().__exit__(exc_type, exc, traceback)

    def connect(self, *, read_only=False):
        opened = original_connect(self, read_only=read_only)
        assert isinstance(opened, sqlite_module._SQLiteConnection)
        connection = TrackedConnection(opened.connection)
        open_connections.add(id(connection))
        return connection

    def checkpoint(path):
        checkpoints.append(len(open_connections))
        release.set()  # Never leave a reader blocking the real checkpoint.
        original_checkpoint(path)

    monkeypatch.setattr(sqlite_module._SQLiteDatabase, "connect", connect)
    monkeypatch.setattr(sqlite_module, "_checkpoint_wal_for_close", checkpoint)
    closing = None
    try:
        armed.set()
        agent._embedded._job_supervisor.wake()
        assert await asyncio.to_thread(entered.wait, 5)
        closing = asyncio.create_task(agent.close())
        await asyncio.sleep(0.2)
        # The cancelled supervisor still owns a connection, so state stays open.
        assert not closing.done() and checkpoints == []
    finally:
        release.set()
        if closing is not None:
            await closing
        else:
            await agent.close()
    assert checkpoints == [0]
