"""SQLite proof for process contention, known rollback and read cost bounds."""

import asyncio
import sqlite3
import sys

import pytest

import daita.storage.sqlite as sqlite_module
from daita.loop.models import RunInput
from daita.storage.errors import StorageUnavailableError
from daita.storage.sqlite import SQLiteStateStore
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = pytest.mark.integration


async def test_conversation_page_and_attempt_guard_have_constant_query_counts(
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
        await store.admit_graph(graph_admission())
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
        statements.clear()
        snapshot = await store.read_graph_attempt_authority(
            "agent-1", "job-1", "worker", "missing"
        )
        assert snapshot is not None and snapshot.task is not None
        assert sum(sql.lstrip().upper().startswith("SELECT") for sql in statements) == 3
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
                await store.admit_graph(graph_admission())
        finally:
            blocker.rollback()
            blocker.close()
        assert await store.inspect_graph("agent-1", "job-1") is None
        await store.admit_graph(graph_admission())
        assert await store.inspect_graph("agent-1", "job-1") is not None
    finally:
        await store.close()


async def test_graph_claim_has_one_winner_across_processes(tmp_path):
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    processes = []
    code = """
import asyncio, sys
from datetime import datetime, timedelta
from daita.storage.sqlite import SQLiteStateStore
from daita.jobs.graph.models import BudgetAmount

async def main():
    path, contender, instant = sys.argv[1:]
    now = datetime.fromisoformat(instant)
    store = await SQLiteStateStore.open(path, current_home_validated=True)
    try:
        attempt = await store.claim_graph_task(
            'agent-1', 'job-1', 'worker', attempt_id='attempt-' + contender,
            claim_token='claim-' + contender, run_id='run-' + contender,
            executor_id='executor-1', claimed_at=now, lease_seconds=30,
            absolute_deadline_at=now + timedelta(minutes=2),
            budget_reservations=(BudgetAmount('work_units', 1),),
        )
        print('won' if attempt is not None else 'lost', flush=True)
        # Keep handles alive until all competing claims have settled. SQLite's
        # local-home close performs an exclusive WAL checkpoint after drain.
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        await store.close()
asyncio.run(main())
"""
    try:
        await store.admit_graph(graph_admission())
        for contender in range(4):
            processes.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    code,
                    str(path),
                    str(contender),
                    GRAPH_NOW.isoformat(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            )
        readers = []
        for process in processes:
            assert process.stdout is not None
            readers.append(process.stdout)
        claims = await asyncio.wait_for(
            asyncio.gather(*(reader.readline() for reader in readers)), 20
        )
        # Close only after drain, serializing the exclusive local-home
        # checkpoints without serializing the competing claim transactions.
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
        inspection = await store.inspect_graph("agent-1", "job-1")
        assert inspection is not None and len(inspection.attempts) == 1
        assert inspection.graph.active_attempt_count == 1
    finally:
        for process in processes:
            if process.returncode is None:
                process.kill()
        await asyncio.gather(*(process.wait() for process in processes))
        await store.close()
