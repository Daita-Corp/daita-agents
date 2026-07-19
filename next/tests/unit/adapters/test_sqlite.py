from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import (
    DiscoveryLimitError,
    DiscoveryRequest,
    ResourceAdapter,
    ResourceNotFoundError,
    ResourceRef,
    SQLiteResourceAdapter,
    SQLiteSource,
    SQLiteSourceError,
    SourceClosedError,
    StaleResourceError,
)
from daita.catalog import (
    FacetKind,
    RelationshipKind,
    ResourceKind,
    catalog_resource_id,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def _create_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE customers (
                tenant_id INTEGER NOT NULL,
                id INTEGER NOT NULL,
                name TEXT DEFAULT 'unknown',
                PRIMARY KEY (tenant_id, id)
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                tenant_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                status TEXT,
                FOREIGN KEY (tenant_id, customer_id)
                    REFERENCES customers (tenant_id, id)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            );
            CREATE UNIQUE INDEX idx_orders_status
                ON orders (status) WHERE status IS NOT NULL;
            CREATE VIEW active_orders AS
                SELECT id, customer_id FROM orders WHERE status = 'active';
            """)
        connection.commit()
    finally:
        connection.close()


async def _open(path: Path) -> SQLiteResourceAdapter:
    source = SQLiteSource(path=path, name="Fixture")
    return await source.open(
        agent_id="agent-1",
        attached_at=NOW,
        clock=lambda: NOW + timedelta(seconds=5),
    )


def _request(adapter, sync_id: str = "sync-1", **limits: int) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id="agent-1",
        source_id=adapter.registration.id,
        sync_id=sync_id,
        requested_at=NOW,
        **limits,
    )


async def test_sqlite_discovery_builds_complete_catalog_snapshot(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.db"
    _create_database(database)
    adapter = await _open(database)

    try:
        result = await adapter.discover(_request(adapter))
    finally:
        await adapter.close()

    snapshot = result.snapshot
    assert snapshot.sync.adapter_id == "sqlite"
    assert snapshot.sync.resource_count == 3
    assert snapshot.sync.relationship_count == 1
    assert {resource.name: resource.kind for resource in snapshot.resources} == {
        "active_orders": ResourceKind.VIEW,
        "customers": ResourceKind.TABLE,
        "orders": ResourceKind.TABLE,
    }

    facets = {facet.resource_id: facet for facet in snapshot.facets}
    orders = next(
        resource for resource in snapshot.resources if resource.name == "orders"
    )
    orders_facet = facets[orders.id]
    assert orders_facet.kind is FacetKind.TABULAR
    column_payloads = orders_facet.payload["columns"]
    assert isinstance(column_payloads, tuple)
    column_names: list[object] = []
    for column_payload in column_payloads:
        assert isinstance(column_payload, FrozenJsonObject)
        column_names.append(column_payload["name"])
    assert column_names == [
        "id",
        "tenant_id",
        "customer_id",
        "status",
    ]
    index_payloads = orders_facet.payload["indexes"]
    assert isinstance(index_payloads, tuple)
    index_payload = index_payloads[0]
    assert isinstance(index_payload, FrozenJsonObject)
    assert index_payload["name"] == "idx_orders_status"
    assert index_payload["predicate"] == "status IS NOT NULL"

    relationship = snapshot.relationships[0]
    assert relationship.kind is RelationshipKind.REFERENCES
    assert [
        (pair.source_field, pair.target_field) for pair in relationship.field_pairs
    ] == [
        ("tenant_id", "tenant_id"),
        ("customer_id", "id"),
    ]
    assert isinstance(relationship.attributes, FrozenJsonObject)
    assert relationship.attributes.to_dict() == {
        "match": "NONE",
        "on_delete": "RESTRICT",
        "on_update": "CASCADE",
    }


async def test_discovery_excludes_rtree_virtual_and_shadow_tables(
    tmp_path: Path,
) -> None:
    database = tmp_path / "rtree.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE places (id INTEGER PRIMARY KEY, name TEXT);
            CREATE VIRTUAL TABLE place_bounds USING rtree(
                id,
                min_x,
                max_x,
                min_y,
                max_y
            );
            """)
    adapter = await _open(database)

    try:
        result = await adapter.discover(_request(adapter))
    finally:
        await adapter.close()

    assert tuple(resource.name for resource in result.snapshot.resources) == ("places",)
    assert result.snapshot.sync.resource_count == 1


async def test_discovery_keeps_ids_and_structural_revisions_stable(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stable.db"
    _create_database(database)
    adapter = await _open(database)

    try:
        first = await adapter.discover(_request(adapter, "sync-1"))
        second = await adapter.discover(_request(adapter, "sync-2"))
    finally:
        await adapter.close()

    first_resources = {
        resource.native_identity: (resource.id, resource.current_revision)
        for resource in first.snapshot.resources
    }
    second_resources = {
        resource.native_identity: (resource.id, resource.current_revision)
        for resource in second.snapshot.resources
    }
    assert first_resources == second_resources
    assert [edge.id for edge in first.snapshot.relationships] == [
        edge.id for edge in second.snapshot.relationships
    ]
    assert [edge.revision for edge in first.snapshot.relationships] == [
        edge.revision for edge in second.snapshot.relationships
    ]


async def test_sqlite_connection_is_uri_read_only_and_query_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "readonly.db"
    _create_database(database)
    adapter = await _open(database)

    try:
        assert isinstance(adapter, ResourceAdapter)
        assert adapter._connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            adapter._connection.execute("CREATE TABLE forbidden (id INTEGER)")
        assert not hasattr(adapter, "query")
        assert not hasattr(adapter, "execute")
    finally:
        await adapter.close()


async def test_inspect_returns_one_linked_snapshot_and_checks_revision(
    tmp_path: Path,
) -> None:
    database = tmp_path / "inspect.db"
    _create_database(database)
    adapter = await _open(database)

    try:
        discovered = await adapter.discover(_request(adapter))
        orders = next(
            resource
            for resource in discovered.snapshot.resources
            if resource.name == "orders"
        )
        reference = ResourceRef.from_resource(orders)
        inspected = await adapter.inspect(reference)

        assert inspected.resource.id == orders.id
        assert inspected.revision.revision == orders.current_revision
        assert len(inspected.facets) == 1
        assert len(inspected.relationships) == 1

        with pytest.raises(StaleResourceError):
            await adapter.inspect(replace(reference, revision="sha256:" + "0" * 64))
        missing = ResourceRef(
            agent_id="agent-1",
            source_id=adapter.registration.id,
            resource_id=catalog_resource_id(
                adapter.registration.id,
                ResourceKind.TABLE,
                "main.missing",
            ),
            native_identity="main.missing",
            kind=ResourceKind.TABLE,
        )
        with pytest.raises(ResourceNotFoundError):
            await adapter.inspect(missing)
    finally:
        await adapter.close()


async def test_discovery_limits_fail_without_partial_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "bounded.db"
    _create_database(database)
    adapter = await _open(database)

    try:
        with pytest.raises(DiscoveryLimitError, match="resource limit"):
            await adapter.discover(_request(adapter, max_resources=2))
        with pytest.raises(DiscoveryLimitError, match="column limit"):
            await adapter.discover(
                _request(adapter, max_columns_per_resource=2),
            )
    finally:
        await adapter.close()


@pytest.mark.parametrize("unsafe_kind", ["directory", "symlink", "parent_symlink"])
async def test_sqlite_source_rejects_non_files_and_symlinks(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    database = real_directory / "source.db"
    _create_database(database)

    if unsafe_kind == "directory":
        candidate = real_directory
    elif unsafe_kind == "symlink":
        candidate = tmp_path / "linked.db"
        candidate.symlink_to(database)
    else:
        linked_directory = tmp_path / "linked-directory"
        linked_directory.symlink_to(real_directory, target_is_directory=True)
        candidate = linked_directory / "source.db"

    with pytest.raises(SQLiteSourceError, match="regular file"):
        await _open(candidate)


async def test_non_sqlite_file_fails_with_normalized_error(tmp_path: Path) -> None:
    database = tmp_path / "not-sqlite.db"
    database.write_bytes(os.urandom(128))

    with pytest.raises(SQLiteSourceError) as caught:
        await _open(database)

    assert caught.value.code == "sqlite_open_failed"
    assert "file is not a database" not in str(caught.value).lower()


async def test_health_and_close_are_bounded_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "health.db"
    _create_database(database)
    adapter = await _open(database)

    healthy = await adapter.health()
    assert healthy.healthy is True
    assert healthy.details["query_only"] is True

    await adapter.close()
    await adapter.close()

    closed = await adapter.health()
    assert closed.healthy is False
    assert closed.error_code == "source_closed"
    with pytest.raises(SourceClosedError):
        await adapter.discover(_request(adapter))
