from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.adapters import DiscoveryRequest, SQLiteSource
from daita.catalog import CatalogSyncConflictError, CatalogSyncStatus
from daita.identity import AgentIdentity
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _create_source(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE alpha (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()


def _add_resource(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE beta (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()


def _running(sync):
    return replace(
        sync,
        status=CatalogSyncStatus.RUNNING,
        completed_at=None,
        source_revision=None,
        resource_count=0,
        relationship_count=0,
    )


async def _discover(
    path: Path,
    *,
    attached_at: datetime,
    requested_at: datetime,
    sync_id: str,
):
    adapter = await SQLiteSource(path).open(
        agent_id="agent-1",
        attached_at=attached_at,
        clock=lambda: requested_at + timedelta(seconds=1),
    )
    try:
        result = await adapter.discover(
            DiscoveryRequest(
                agent_id="agent-1",
                source_id=adapter.registration.id,
                sync_id=sync_id,
                requested_at=requested_at,
            )
        )
        return adapter.registration, result.snapshot
    finally:
        await adapter.close()


async def test_refresh_preserves_first_observation_and_dates_new_resources(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite3"
    store_path = tmp_path / "agent.sqlite3"
    _create_source(source_path)
    store = await SQLiteOperationStore.open(store_path)
    await store.initialize_identity(
        AgentIdentity(
            id="agent-1",
            display_name="Catalog refresh agent",
            created_at=NOW,
        )
    )

    first_requested = NOW + timedelta(minutes=1)
    first_registration, first_discovery = await _discover(
        source_path,
        attached_at=NOW,
        requested_at=first_requested,
        sync_id="sync-1",
    )
    assert {resource.first_observed_at for resource in first_discovery.resources} == {
        first_requested
    }

    try:
        await store.register_source(first_registration)
        await store.record_sync(_running(first_discovery.sync))
        untrusted_history = replace(
            first_discovery,
            resources=tuple(
                replace(resource, first_observed_at=NOW)
                for resource in first_discovery.resources
            ),
        )
        first_committed = await store.commit_snapshot(untrusted_history)
        assert first_committed.resources[0].first_observed_at == first_requested

        _add_resource(source_path)
        second_requested = NOW + timedelta(minutes=2)
        second_registration, second_discovery = await _discover(
            source_path,
            attached_at=NOW + timedelta(minutes=2),
            requested_at=second_requested,
            sync_id="sync-2",
        )
        assert second_registration.id == first_registration.id
        assert {
            resource.first_observed_at for resource in second_discovery.resources
        } == {second_requested}

        await store.record_sync(_running(second_discovery.sync))
        second_committed = await store.commit_snapshot(second_discovery)
        committed_by_name = {
            resource.name: resource for resource in second_committed.resources
        }
        assert committed_by_name["alpha"].first_observed_at == first_requested
        assert committed_by_name["alpha"].last_observed_at == second_requested
        assert committed_by_name["beta"].first_observed_at == second_requested
        assert committed_by_name["beta"].last_observed_at == second_requested
        assert await store.commit_snapshot(second_discovery) == second_committed
        assert set(await store.list_resources("agent-1", first_registration.id)) == set(
            second_committed.resources
        )
    finally:
        await store.close()


async def test_refresh_rejects_freshness_regression_and_sync_identity_change(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.sqlite3"
    _create_source(source_path)
    store = await SQLiteOperationStore.open(tmp_path / "agent.sqlite3")
    await store.initialize_identity(
        AgentIdentity(
            id="agent-1",
            display_name="Catalog refresh agent",
            created_at=NOW,
        )
    )
    registration, first = await _discover(
        source_path,
        attached_at=NOW,
        requested_at=NOW + timedelta(minutes=1),
        sync_id="sync-1",
    )
    try:
        await store.register_source(registration)
        committed = await store.commit_snapshot(first)

        _, regression = await _discover(
            source_path,
            attached_at=NOW,
            requested_at=NOW + timedelta(minutes=2),
            sync_id="sync-regression",
        )
        regression = replace(
            regression,
            resources=tuple(
                replace(
                    resource,
                    first_observed_at=NOW,
                    last_observed_at=NOW,
                )
                for resource in regression.resources
            ),
        )
        with pytest.raises(
            CatalogSyncConflictError,
            match="resource_freshness_regressed",
        ):
            await store.commit_snapshot(regression)
        assert await store.load_sync("agent-1", "sync-regression") is None

        _, identity_candidate = await _discover(
            source_path,
            attached_at=NOW,
            requested_at=NOW + timedelta(minutes=3),
            sync_id="sync-identity",
        )
        await store.record_sync(_running(identity_candidate.sync))
        identity_candidate = replace(
            identity_candidate,
            sync=replace(identity_candidate.sync, adapter_id="not-sqlite"),
        )
        with pytest.raises(CatalogSyncConflictError, match="identity_changed"):
            await store.commit_snapshot(identity_candidate)
        assert await store.list_resources("agent-1", registration.id) == tuple(
            sorted(
                committed.resources,
                key=lambda item: (item.kind.value, item.name, item.id),
            )
        )
    finally:
        await store.close()
