"""Real PostgreSQL transactions under fencing, cancellation and transport loss."""

import asyncio
import threading

import psycopg
import pytest

from daita.identity import AgentIdentity
from daita.storage.errors import (
    StorageCommitUnknownError,
    StorageOwnershipLostError,
    StorageUnavailableError,
)
from daita.storage.postgres import PostgresStateStore
from daita.storage.sql import _run_cancellation_safe_transaction
from tests.support.graph import GRAPH_NOW, graph_admission

pytestmark = pytest.mark.integration


async def test_new_generation_fences_every_old_mutation(postgres_config):
    old, new = await PostgresStateStore.open(
        postgres_config
    ), await PostgresStateStore.open(postgres_config)
    try:
        assert await new.advance_fence() == old.fencing_epoch + 1
        with pytest.raises(StorageOwnershipLostError):
            await old.admit_graph(graph_admission())
        with pytest.raises(StorageOwnershipLostError):
            await old.initialize_identity(
                AgentIdentity(id="old", display_name="Old", created_at=GRAPH_NOW)
            )
        with pytest.raises(StorageOwnershipLostError):
            await old.advance_fence()
        assert await new.inspect_graph("agent-1", "job-1") is None
        await new.admit_graph(graph_admission())
        assert await new.inspect_graph("agent-1", "job-1") is not None
    finally:
        await old.close()
        await new.close()


async def test_acknowledgement_loss_is_unknown_and_never_replayed(
    postgres_config, monkeypatch
):
    store = await PostgresStateStore.open(postgres_config)
    real_execute = psycopg.Connection.execute
    calls = []

    def lose_ack(self, query, *args, **kwargs):
        result = real_execute(self, query, *args, **kwargs)
        if query == "COMMIT":
            calls.append(1)
            raise psycopg.OperationalError(
                "injected lost acknowledgement after real commit"
            )
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(psycopg.Connection, "execute", lose_ack)
            with pytest.raises(StorageCommitUnknownError):
                await store.admit_graph(graph_admission())
        with pytest.raises(StorageCommitUnknownError):
            await store.admit_graph(graph_admission())
        assert calls == [1]
    finally:
        await store.close()
    reopened = await PostgresStateStore.open(postgres_config)
    try:
        inspection = await reopened.inspect_graph("agent-1", "job-1")
        assert inspection is not None and len(inspection.tasks) == 2
    finally:
        await reopened.close()


async def test_backend_termination_rolls_back_without_partial_graph(
    postgres_database, postgres_config
):
    dsn, _ = postgres_database
    store = await PostgresStateStore.open(postgres_config)
    try:

        def interrupted(connection):
            connection.execute(
                "INSERT INTO metadata(key, data) VALUES ('uncommitted', '{}')"
            )
            pid = connection.execute("SELECT pg_backend_pid()").fetchone()[0]
            with psycopg.connect(dsn(), autocommit=True) as admin:
                admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
            connection.execute("SELECT 1")

        from daita.storage.errors import StorageError

        with pytest.raises(StorageError):
            await _run_cancellation_safe_transaction(store._postgres, interrupted)
        with psycopg.connect(dsn(), autocommit=True) as admin:
            assert admin.execute(
                "SELECT count(*) FROM daita_state.metadata WHERE namespace_id = %s",
                (postgres_config.namespace_id,),
            ).fetchone() == (0,)
        await store.admit_graph(graph_admission())
        assert await store.inspect_graph("agent-1", "job-1") is not None
    finally:
        await store.close()


async def test_cancellation_after_mutation_start_drains_exactly_one_commit(
    postgres_config,
):
    store = await PostgresStateStore.open(postgres_config)
    started, release = threading.Event(), threading.Event()
    try:

        def write(connection):
            connection.execute(
                "INSERT INTO metadata(key, data) VALUES ('committed', '{}')"
            )
            started.set()
            assert release.wait(5)
            return "committed"

        task = asyncio.create_task(
            _run_cancellation_safe_transaction(store._postgres, write)
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert await task == "committed"
        with store._postgres.connect(read_only=True) as connection:
            assert connection.execute(
                "SELECT count(*) FROM metadata WHERE key = 'committed'"
            ).fetchone() == (1,)
    finally:
        release.set()
        await store.close()


async def test_queued_write_cannot_overtake_unknown_commit(
    postgres_config, monkeypatch
):
    store = await PostgresStateStore.open(postgres_config)
    committed, release = threading.Event(), threading.Event()
    real_execute = psycopg.Connection.execute
    calls = []

    def lose_ack(self, query, *args, **kwargs):
        result = real_execute(self, query, *args, **kwargs)
        if query == "COMMIT":
            calls.append(1)
            committed.set()
            assert release.wait(5)
            raise psycopg.OperationalError("lost commit acknowledgement")
        return result

    def write(connection):
        connection.execute("INSERT INTO metadata(key, data) VALUES ('once', '{}')")

    tasks = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(psycopg.Connection, "execute", lose_ack)
            tasks.append(
                asyncio.create_task(
                    _run_cancellation_safe_transaction(store._postgres, write)
                )
            )
            assert await asyncio.to_thread(committed.wait, 5)
            queued = store._postgres.connect()
            tasks.append(
                asyncio.create_task(asyncio.to_thread(queued.begin, write=True))
            )
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            queued.close()
            assert all(
                isinstance(result, StorageCommitUnknownError) for result in results
            )
            assert calls == [1]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await store.close()


async def test_server_confirmed_commit_rejection_is_retryable(
    postgres_config, monkeypatch
):
    store = await PostgresStateStore.open(postgres_config)
    real_execute = psycopg.Connection.execute

    def reject_commit(self, query, *args, **kwargs):
        if query == "COMMIT":
            real_execute(self, "ROLLBACK")
            raise psycopg.errors.SerializationFailure("server rejected transaction")
        return real_execute(self, query, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(psycopg.Connection, "execute", reject_commit)
            with pytest.raises(StorageUnavailableError):
                await store.admit_graph(graph_admission())
        assert await store.inspect_graph("agent-1", "job-1") is None
        await store.admit_graph(graph_admission())
        assert await store.inspect_graph("agent-1", "job-1") is not None
    finally:
        await store.close()
