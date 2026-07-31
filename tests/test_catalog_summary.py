from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading

import pytest

from daita import Agent, CatalogSummary, SQLiteSource
from daita.adapters.models import SourceRegistration
from daita.catalog import (
    CatalogSearchRequest,
    CatalogSync,
    CatalogSyncStatus,
    CatalogTraversalRequest,
    ResourceKind,
)
from daita.catalog.models import CatalogSnapshotRef
from daita.llm.models import FinishReason, ModelProfile, ModelResponse
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore
import daita.storage.sqlite as sqlite_store


def _database(path: Path, *, with_tables: bool = True) -> None:
    with sqlite3.connect(path) as connection:
        if not with_tables:
            return
        connection.executescript("""
            CREATE TABLE parent (id INTEGER PRIMARY KEY);
            CREATE TABLE child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER REFERENCES parent(id)
            );
            """)


def _snapshot_decode_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], int]:
    original_loads = sqlite_store._loads
    counter_lock = threading.Lock()
    count = 0

    def counting_loads(value: str) -> object:
        nonlocal count
        if '"__record__":"SourceCatalogSnapshot"' in value:
            with counter_lock:
                count += 1
        return original_loads(value)

    monkeypatch.setattr(sqlite_store, "_loads", counting_loads)
    return lambda: count


async def test_catalog_summary_aggregates_current_active_snapshots_and_latest_sync(
    tmp_path: Path,
):
    first_time = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    current_time = first_time

    def clock() -> datetime:
        return current_time

    first_database = tmp_path / "first.sqlite"
    second_database = tmp_path / "second.sqlite"
    _database(first_database)
    _database(second_database)
    agent = await Agent.create("summary", root=tmp_path, clock=clock)
    try:
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=0,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )

        first = await agent.attach(SQLiteSource(first_database, name="First"))
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=2,
            relationship_count=1,
            latest_successful_sync_completed_at=first_time,
            is_empty=False,
        )

        current_time = first_time + timedelta(minutes=5)
        second = await agent.attach(SQLiteSource(second_database, name="Second"))
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=2,
            resource_count=4,
            relationship_count=2,
            latest_successful_sync_completed_at=current_time,
            is_empty=False,
        )

        await agent.detach(first.id)
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=2,
            relationship_count=1,
            latest_successful_sync_completed_at=current_time,
            is_empty=False,
        )
        await agent.detach(second.id)
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=0,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )
    finally:
        await agent.close()


async def test_empty_successful_snapshot_is_not_ready_but_retains_sync_time(
    tmp_path: Path,
):
    completed_at = datetime(2026, 7, 22, 13, 0, tzinfo=timezone.utc)
    database = tmp_path / "empty.sqlite"
    _database(database, with_tables=False)
    agent = await Agent.create("empty", root=tmp_path, clock=lambda: completed_at)
    try:
        await agent.attach(SQLiteSource(database))
        summary = await agent.catalog_summary()
        assert summary.active_source_count == 1
        assert summary.resource_count == 0
        assert summary.relationship_count == 0
        assert summary.latest_successful_sync_completed_at == completed_at
        assert summary.is_empty is True
    finally:
        await agent.close()


async def test_catalog_preview_contains_only_active_current_snapshot_truth(
    tmp_path: Path,
):
    first_database = tmp_path / "preview-first.sqlite"
    second_database = tmp_path / "preview-second.sqlite"
    with sqlite3.connect(first_database) as connection:
        connection.execute("CREATE TABLE first_only (id INTEGER PRIMARY KEY)")
    with sqlite3.connect(second_database) as connection:
        connection.execute("CREATE TABLE second_only (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("preview-active", root=tmp_path)
    try:
        first = await agent.attach(SQLiteSource(first_database, name="First"))
        second = await agent.attach(SQLiteSource(second_database, name="Second"))
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == (
            "first_only",
            "second_only",
        )

        await agent.detach(first.id)
        assert all(
            key[:2] != (agent.id, first.id)
            for key in agent._embedded._store._decoded_catalog_snapshots
        )
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == ("second_only",)
        assert {resource.source_id for resource in preview} == {second.id}

        with sqlite3.connect(second_database) as connection:
            connection.execute("DROP TABLE second_only")
            connection.execute("CREATE TABLE refreshed_only (id INTEGER PRIMARY KEY)")
        await agent.refresh_source(second.id)
        preview = await agent.catalog_preview(limit=50)
        assert tuple(resource.name for resource in preview) == ("refreshed_only",)

        await agent.detach(second.id)
        assert all(
            key[:2] != (agent.id, second.id)
            for key in agent._embedded._store._decoded_catalog_snapshots
        )
        assert await agent.catalog_preview(limit=50) == ()
        assert (await agent.catalog_summary()).is_empty is True
    finally:
        await agent.close()


async def test_broad_catalog_discovery_matches_any_term_and_resource_kind(
    tmp_path: Path,
):
    database = tmp_path / "search.sqlite"
    _database(database)
    agent = await Agent.create("search", root=tmp_path)
    try:
        await agent.attach(SQLiteSource(database))
        result = await agent.search_catalog(
            CatalogSearchRequest(
                agent_id=agent.id,
                query="tables views datasets schemas relationships",
                resource_kinds=(ResourceKind.TABLE, ResourceKind.VIEW),
                limit=50,
            )
        )

        assert tuple(hit.name for hit in result.hits) == ("child", "parent")
        assert result.total_matches == 2
        assert all("kind" in hit.matched_fields for hit in result.hits)
    finally:
        await agent.close()


async def test_current_snapshot_refs_are_agent_and_source_isolated(tmp_path: Path):
    first_database = tmp_path / "refs-first.sqlite"
    second_database = tmp_path / "refs-second.sqlite"
    _database(first_database)
    _database(second_database)
    agent = await Agent.create("snapshot-refs", root=tmp_path)
    try:
        first = await agent.attach(SQLiteSource(first_database))
        second = await agent.attach(SQLiteSource(second_database))
        store = agent._embedded._store

        refs = await store.list_current_snapshot_refs(agent.id, ())
        assert refs == tuple(
            sorted(refs, key=lambda item: (item.source_id, item.sync_id))
        )
        assert {item.source_id for item in refs} == {first.id, second.id}
        assert all(item.agent_id == agent.id for item in refs)
        assert (
            await store.list_current_snapshot_refs(
                agent.id,
                (second.id, first.id),
            )
            == refs
        )
        assert await store.list_current_snapshot_refs(agent.id, (first.id,)) == tuple(
            item for item in refs if item.source_id == first.id
        )
        assert await store.list_current_snapshot_refs("another-agent", ()) == ()
        assert (
            await store.list_current_snapshot_refs(agent.id, ("missing-source",)) == ()
        )

        with pytest.raises(ValueError, match="duplicates"):
            await store.list_current_snapshot_refs(agent.id, (first.id, first.id))
        with pytest.raises(ValueError, match="non-empty"):
            await store.list_current_snapshot_refs(agent.id, ("",))
    finally:
        await agent.close()


async def test_current_snapshot_load_requires_the_exact_current_reference(
    tmp_path: Path,
):
    database = tmp_path / "exact-ref.sqlite"
    _database(database)
    agent = await Agent.create("snapshot-exact-ref", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        ref = (await store.list_current_snapshot_refs(agent.id, (source.id,)))[0]

        snapshot = await store.load_current_snapshot(ref)
        assert snapshot is not None
        assert snapshot.sync.agent_id == ref.agent_id
        assert snapshot.sync.source_id == ref.source_id
        assert snapshot.sync.id == ref.sync_id
        assert (
            await store.load_current_snapshot(
                CatalogSnapshotRef(
                    agent_id=ref.agent_id,
                    source_id=ref.source_id,
                    sync_id="not-current",
                )
            )
            is None
        )
        assert (
            await store.load_current_snapshot(
                CatalogSnapshotRef(
                    agent_id="another-agent",
                    source_id=ref.source_id,
                    sync_id=ref.sync_id,
                )
            )
            is None
        )
    finally:
        await agent.close()


def test_snapshot_ref_rejects_unbounded_or_blank_identity_fields():
    with pytest.raises(ValueError, match="non-empty"):
        CatalogSnapshotRef(agent_id="", source_id="source", sync_id="sync")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        CatalogSnapshotRef(agent_id="agent", source_id=" source", sync_id="sync")
    with pytest.raises(ValueError, match="512"):
        CatalogSnapshotRef(agent_id="agent", source_id="source", sync_id="x" * 513)


async def test_catalog_reads_decode_one_generation_once_and_share_concurrent_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "decode-once.sqlite"
    _database(database)
    agent = await Agent.create("snapshot-decode-once", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        decode_count = _snapshot_decode_counter(monkeypatch)

        loaded = await asyncio.gather(
            *(reader.load_current_snapshot(ref) for _ in range(16))
        )
        assert loaded[0] is not None
        assert all(snapshot is loaded[0] for snapshot in loaded)
        snapshot = loaded[0]
        assert snapshot is not None

        resources = await reader.list_resources(agent.id, source.id)
        await reader.load_resource(agent.id, resources[0].id)
        await reader.load_revision(
            agent.id,
            resources[0].id,
            resources[0].current_revision,
        )
        await reader.load_facets(
            agent.id,
            resources[0].id,
            resources[0].current_revision,
        )
        await reader.load_incident_relationships(agent.id, resources[0].id)
        await reader.load_relationships(
            agent.id,
            tuple(item.id for item in snapshot.relationships),
        )
        await reader.search(
            CatalogSearchRequest(agent_id=agent.id, query="parent", limit=50)
        )
        await reader.traverse(
            CatalogTraversalRequest(
                agent_id=agent.id,
                from_resource_ids=(snapshot.relationships[0].from_resource_id,),
                to_resource_ids=(snapshot.relationships[0].to_resource_id,),
            )
        )
        assert decode_count() == 1
    finally:
        await agent.close()


async def test_failed_snapshot_decode_does_not_publish_a_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "failed-decode.sqlite"
    _database(database)
    agent = await Agent.create("failed-snapshot-decode", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        original_loads = sqlite_store._loads
        fail_next_snapshot = True

        def failing_loads(value: str) -> object:
            nonlocal fail_next_snapshot
            if fail_next_snapshot and '"__record__":"SourceCatalogSnapshot"' in value:
                fail_next_snapshot = False
                raise ValueError("forced snapshot decode failure")
            return original_loads(value)

        monkeypatch.setattr(sqlite_store, "_loads", failing_loads)
        with pytest.raises(ValueError, match="forced snapshot decode failure"):
            await reader.load_current_snapshot(ref)
        assert reader._decoded_catalog_snapshots == {}

        snapshot = await reader.load_current_snapshot(ref)
        assert snapshot is not None
        assert set(reader._decoded_catalog_snapshots) == {
            (ref.agent_id, ref.source_id, ref.sync_id)
        }
    finally:
        await agent.close()


async def test_refresh_invalidates_stale_ref_and_decodes_the_new_generation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "generation-refresh.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE old_table (id INTEGER PRIMARY KEY)")
    agent = await Agent.create("snapshot-generation-refresh", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        reader = SQLiteStateStore(agent.home / "state.db")
        decode_count = _snapshot_decode_counter(monkeypatch)
        old_ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        old_resources = await reader.list_resources(agent.id, source.id)
        assert tuple(item.name for item in old_resources) == ("old_table",)
        assert decode_count() == 1

        with sqlite3.connect(database) as connection:
            connection.execute("DROP TABLE old_table")
            connection.execute("CREATE TABLE new_table (id INTEGER PRIMARY KEY)")
        await agent.refresh_source(source.id)

        new_ref = (await reader.list_current_snapshot_refs(agent.id, (source.id,)))[0]
        assert new_ref != old_ref
        assert await reader.load_current_snapshot(old_ref) is None
        new_resources = await reader.list_resources(agent.id, source.id)
        assert tuple(item.name for item in new_resources) == ("new_table",)
        assert await reader.load_resource(agent.id, old_resources[0].id) is None
        assert decode_count() == 2
        assert set(reader._decoded_catalog_snapshots) == {
            (new_ref.agent_id, new_ref.source_id, new_ref.sync_id)
        }
    finally:
        await agent.close()


async def test_failed_snapshot_commit_does_not_publish_candidate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "failed-cache-publication.sqlite"
    _database(database)
    agent = await Agent.create("failed-cache-publication", root=tmp_path)
    try:
        source = await agent.attach(SQLiteSource(database))
        store = agent._embedded._store
        cache_before = dict(store._decoded_catalog_snapshots)
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE uncommitted (id INTEGER PRIMARY KEY)")

        def fail_commit(connection):
            raise RuntimeError("forced catalog commit failure")

        monkeypatch.setattr(sqlite_store, "_commit_catalog_transaction", fail_commit)
        with pytest.raises(RuntimeError, match="forced catalog commit failure"):
            await agent.refresh_source(source.id)

        assert store._decoded_catalog_snapshots == cache_before
        assert tuple(
            item.name
            for item in await agent.list_catalog_resources(source_id=source.id)
        ) == ("child", "parent")
    finally:
        await agent.close()


async def test_running_partial_and_failed_syncs_never_contribute_or_replace_truth(
    tmp_path: Path,
):
    started_at = datetime(2026, 7, 22, 14, 0, tzinfo=timezone.utc)
    agent = await Agent.create("sync-state", root=tmp_path, clock=lambda: started_at)
    store = SQLiteStateStore(agent.home / "state.db")
    try:
        registration = SourceRegistration.build(
            agent_id=agent.id,
            adapter_id="test.adapter",
            native_identity="partial-only",
            display_name="Partial only",
            configuration={},
            attached_at=started_at,
        )
        await store.register_source(registration)
        await store.record_sync(
            CatalogSync(
                id="running-sync",
                agent_id=agent.id,
                source_id=registration.id,
                adapter_id=registration.adapter_id,
                status=CatalogSyncStatus.RUNNING,
                started_at=started_at,
            )
        )
        await store.record_sync(
            CatalogSync(
                id="partial-sync",
                agent_id=agent.id,
                source_id=registration.id,
                adapter_id=registration.adapter_id,
                status=CatalogSyncStatus.PARTIAL,
                started_at=started_at,
                completed_at=started_at + timedelta(minutes=1),
                resource_count=99,
                relationship_count=88,
                error_code="bounded_partial",
            )
        )
        assert await agent.catalog_summary() == CatalogSummary(
            active_source_count=1,
            resource_count=0,
            relationship_count=0,
            latest_successful_sync_completed_at=None,
            is_empty=True,
        )

        database = tmp_path / "successful.sqlite"
        _database(database)
        successful = await agent.attach(SQLiteSource(database))
        committed = await agent.catalog_summary()
        await store.record_sync(
            CatalogSync(
                id="failed-after-success",
                agent_id=agent.id,
                source_id=successful.id,
                adapter_id=successful.adapter_id,
                status=CatalogSyncStatus.FAILED,
                started_at=started_at + timedelta(minutes=2),
                completed_at=started_at + timedelta(minutes=3),
                error_code="source_attach_failed",
            )
        )
        assert await agent.catalog_summary() == committed
        assert committed.active_source_count == 2
        assert committed.resource_count == 2
        assert committed.relationship_count == 1
    finally:
        await store.close()
        await agent.close()


async def test_refresh_cancellation_before_transaction_keeps_old_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "pre-commit.sqlite"
    _database(database)
    agent = await Agent.create("pre-commit", root=tmp_path)
    source = await agent.attach(SQLiteSource(database))
    old_resources = await agent.list_catalog_resources(source_id=source.id)
    cache_before = dict(agent._embedded._store._decoded_catalog_snapshots)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE later (id INTEGER PRIMARY KEY)")

    before_start = threading.Event()
    release_start = threading.Event()
    original_start = sqlite_store._CatalogCommitGate.start

    def paused_start(self, connection):
        before_start.set()
        if not release_start.wait(5):
            raise TimeoutError("catalog transaction start was not released")
        return original_start(self, connection)

    monkeypatch.setattr(sqlite_store._CatalogCommitGate, "start", paused_start)
    refresh = asyncio.create_task(agent.refresh_source(source.id))
    try:
        assert await asyncio.to_thread(before_start.wait, 5)
        refresh.cancel()
        await asyncio.sleep(0)
        release_start.set()
        with pytest.raises(asyncio.CancelledError):
            await refresh

        current_resources = await agent.list_catalog_resources(source_id=source.id)
        assert current_resources == old_resources
        assert agent._embedded._store._decoded_catalog_snapshots == cache_before
        with sqlite3.connect(agent.home / "state.db") as connection:
            sync_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT id FROM syncs WHERE source_id = ? ORDER BY id",
                    (source.id,),
                )
            )
        store = SQLiteStateStore(agent.home / "state.db")
        syncs = tuple(
            [await store.load_sync(agent.id, sync_id) for sync_id in sync_ids]
        )
        assert any(
            sync is not None and sync.status is CatalogSyncStatus.FAILED
            for sync in syncs
        )
        current_sync_ids = {resource.current_sync_id for resource in current_resources}
        assert all(
            sync is None
            or sync.status is not CatalogSyncStatus.FAILED
            or sync.id not in current_sync_ids
            for sync in syncs
        )
    finally:
        release_start.set()
        await agent.close()


async def test_refresh_cancellation_after_transaction_start_commits_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "commit-wins.sqlite"
    _database(database)
    agent = await Agent.create("commit-wins", root=tmp_path)
    source = await agent.attach(SQLiteSource(database))
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE later (id INTEGER PRIMARY KEY)")

    before_commit = threading.Event()
    release_commit = threading.Event()
    original_commit = sqlite_store._commit_catalog_transaction

    def paused_commit(connection):
        before_commit.set()
        if not release_commit.wait(5):
            raise TimeoutError("catalog transaction commit was not released")
        original_commit(connection)

    monkeypatch.setattr(sqlite_store, "_commit_catalog_transaction", paused_commit)
    refresh = asyncio.create_task(agent.refresh_source(source.id))
    try:
        assert await asyncio.to_thread(before_commit.wait, 5)
        refresh.cancel()
        release_commit.set()
        assert await refresh == source

        resources = await agent.list_catalog_resources(source_id=source.id)
        assert tuple(resource.name for resource in resources) == (
            "child",
            "later",
            "parent",
        )
        sync_ids = {resource.current_sync_id for resource in resources}
        assert len(sync_ids) == 1
        store = SQLiteStateStore(agent.home / "state.db")
        sync = await store.load_sync(agent.id, sync_ids.pop())
        assert sync is not None
        assert sync.status is CatalogSyncStatus.SUCCEEDED
    finally:
        release_commit.set()
        await agent.close()


async def test_first_attach_cancellation_before_commit_publishes_no_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    database = tmp_path / "cancel-first.sqlite"
    _database(database)
    agent = await Agent.create("cancel-first", root=tmp_path)
    before_start = threading.Event()
    release_start = threading.Event()
    original_start = sqlite_store._CatalogCommitGate.start

    def paused_start(self, connection):
        before_start.set()
        if not release_start.wait(5):
            raise TimeoutError("catalog transaction start was not released")
        return original_start(self, connection)

    monkeypatch.setattr(sqlite_store._CatalogCommitGate, "start", paused_start)
    attach = asyncio.create_task(agent.attach(SQLiteSource(database)))
    try:
        assert await asyncio.to_thread(before_start.wait, 5)
        attach.cancel()
        await asyncio.sleep(0)
        release_start.set()
        with pytest.raises(asyncio.CancelledError):
            await attach

        assert await agent.list_sources() == ()
        assert await agent.list_catalog_resources() == ()
    finally:
        release_start.set()
        await agent.close()


async def test_catalog_summary_is_not_persisted_or_added_to_model_state(tmp_path: Path):
    database = tmp_path / "modeled.sqlite"
    _database(database)
    provider = MockModelProvider(
        (ModelResponse(finish_reason=FinishReason.STOP, text="done"),)
    )
    profile = ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=256,
        supports_tools=True,
    )
    agent = await Agent.create(
        "projection-only",
        root=tmp_path,
        model=provider,
        model_profile=profile,
    )
    state_path = agent.home / "state.db"
    try:
        await agent.attach(SQLiteSource(database))
        summary = await agent.catalog_summary()
        assert summary.is_empty is False
        result = await agent.run("Answer normally")
        transcript = await agent.transcript(result.run_id)
        request_text = repr(provider.requests[0])
        transcript_text = repr(transcript)
        for field_name in (
            "active_source_count",
            "latest_successful_sync_completed_at",
            "relationship_count",
            "readiness",
        ):
            assert field_name not in request_text
            assert field_name not in transcript_text
    finally:
        await agent.close()

    with sqlite3.connect(state_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "catalog_summaries" not in tables
    assert "readiness" not in tables
