"""SQLite proof for process contention, known rollback and read cost bounds."""

import asyncio
import sqlite3
import sys

import pytest

import daita.storage.sqlite as sqlite_module
from daita.loop.models import RunInput
from daita.storage.errors import StorageUnavailableError
from daita.storage.sqlite import SQLiteStateStore
from tests.support.graph import GRAPH_NOW
from tests.support.jobs import stored_job

pytestmark = pytest.mark.integration


async def test_conversation_page_and_access_have_constant_query_counts(
    tmp_path, monkeypatch
):
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        for index in range(20):
            run = RunInput(
                id=f"run-{index}",
                agent_id="agent-1",
                conversation_id="conversation-1",
                message="test",
                created_at=GRAPH_NOW,
            )
            await store.start(run)
            await store.append_at(run.id, 0, run.start_message())
        statements: list[str] = []
        original = sqlite_module._connect_read_only

        def observed(path):
            connection = original(path)
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr(sqlite_module, "_connect_read_only", observed)
        page = await store.conversation_run_page("agent-1", "conversation-1", limit=20)
        assert len(page) == 20 and all(
            len(run.transcript.messages) == 1 for run in page
        )
        assert (
            sum(
                sql.lstrip().upper().startswith(("SELECT", "WITH"))
                for sql in statements
            )
            == 2
        )
        statements.clear()
        assert await store.conversation_access(
            "agent-1", "conversation-1", caller_principal_id="agent-1"
        ) == (True, True)
        assert sum(sql.lstrip().upper().startswith("SELECT") for sql in statements) == 1
    finally:
        await store.close()


async def test_sqlite_busy_is_retryable_only_after_known_rollback(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    original = sqlite_module._connect

    def no_wait(path):
        connection = original(path)
        connection.execute("PRAGMA busy_timeout = 0")
        return connection

    try:
        monkeypatch.setattr(sqlite_module, "_connect", no_wait)
        blocker = sqlite3.connect(path)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            with pytest.raises(StorageUnavailableError):
                await store.admit_job(stored_job())
        finally:
            blocker.rollback()
            blocker.close()
        assert await store.load_job("agent-1", "job-1") is None
        await store.admit_job(stored_job())
        assert await store.load_job("agent-1", "job-1") is not None
    finally:
        await store.close()


async def test_a_job_revision_has_one_winner_across_processes(tmp_path):
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    processes = []
    code = """
import asyncio, sys
from dataclasses import replace
from daita.storage.errors import JobStoreConflictError
from daita.storage.sqlite import SQLiteStateStore

async def main():
    path, contender = sys.argv[1:]
    store = await SQLiteStateStore.open(path, current_home_validated=True)
    try:
        job = await store.load_job('agent-1', 'job-1')
        # Every contender read revision 1 before any of them writes.
        await asyncio.to_thread(sys.stdin.readline)
        try:
            await store.replace_job(
                replace(job, objective='Contender ' + contender, revision=2)
            )
            print('won', flush=True)
        except JobStoreConflictError:
            print('lost', flush=True)
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        await store.close()
asyncio.run(main())
"""
    try:
        await store.admit_job(stored_job())
        for contender in range(4):
            processes.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    code,
                    str(path),
                    str(contender),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
        for process in processes:
            assert process.stdin is not None
            process.stdin.write(b"\n")
            await process.stdin.drain()
        claims = []
        for process in processes:
            assert process.stdout is not None
            claims.append(await asyncio.wait_for(process.stdout.readline(), 20))
        results = [
            await asyncio.wait_for(process.communicate(b"\n"), 20)
            for process in processes
        ]
        assert all(process.returncode == 0 for process in processes), results
        assert sorted(claim.strip() for claim in claims) == [
            b"lost",
            b"lost",
            b"lost",
            b"won",
        ]
        current = await store.load_job("agent-1", "job-1")
        assert current is not None and current.revision == 2
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
        await asyncio.gather(*(process.wait() for process in processes))
        await store.close()
