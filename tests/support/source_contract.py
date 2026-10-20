"""Reusable contract checks that every registered source type must satisfy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters.models import DiscoveryRequest, ResourceRef
from daita.adapters.protocols import (
    ResourceAdapter,
    ResourceSource,
    SourceClosedError,
    StaleResourceError,
)
from daita.adapters.registry import SourceOperation, SourceType
from daita.catalog.models import FacetKind, SourceCatalogSnapshot, TabularFacet
from daita.domains.data import ReadTabularRequest, TabularFilter, TabularReadBackend
from daita.domains.data.results import BoundedResultProjection
from daita.errors import DaitaError
from daita.security import SecretProvider

CONTRACT_AGENT_ID = "agent-source-contract"
CONTRACT_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _request(adapter: ResourceAdapter, sync_id: str) -> DiscoveryRequest:
    return DiscoveryRequest(
        agent_id=CONTRACT_AGENT_ID,
        source_id=adapter.registration.id,
        sync_id=sync_id,
        requested_at=CONTRACT_NOW,
    )


def _assert_within_request(
    snapshot: SourceCatalogSnapshot, request: DiscoveryRequest
) -> None:
    # Reconstruction reruns the snapshot's own structural validation.
    assert replace(snapshot) == snapshot
    assert snapshot.sync.agent_id == request.agent_id
    assert snapshot.sync.source_id == request.source_id
    assert snapshot.sync.id == request.sync_id
    assert len(snapshot.resources) <= request.max_resources
    assert len(snapshot.relationships) <= request.max_relationships
    for facet in snapshot.facets:
        if facet.kind is FacetKind.TABULAR:
            tabular = TabularFacet.from_payload(facet.payload)
            assert len(tabular.columns) <= request.max_columns_per_resource
            assert len(tabular.indexes) <= request.max_indexes_per_resource


async def assert_source_type_contract(
    source_type: SourceType,
    source: ResourceSource,
    *,
    secret_provider: SecretProvider,
) -> None:
    """Open, discover, inspect, health-check, close and rebuild one source."""

    adapter = await source.open(
        agent_id=CONTRACT_AGENT_ID,
        attached_at=CONTRACT_NOW,
        clock=lambda: CONTRACT_NOW,
    )
    registration = adapter.registration
    try:
        assert registration.adapter_id == source_type.adapter_id

        first_request = _request(adapter, "sync-contract-first")
        first = (await adapter.discover(first_request)).snapshot
        _assert_within_request(first, first_request)
        assert first.resources, "the contract needs at least one resource"
        second_request = _request(adapter, "sync-contract-second")
        second = (await adapter.discover(second_request)).snapshot
        _assert_within_request(second, second_request)
        assert {item.id for item in second.resources} == {
            item.id for item in first.resources
        }

        resource = first.resources[0]
        with pytest.raises(StaleResourceError):
            await adapter.inspect(
                ResourceRef(
                    agent_id=CONTRACT_AGENT_ID,
                    source_id=registration.id,
                    resource_id=resource.id,
                    native_identity=resource.native_identity,
                    kind=resource.kind,
                    revision="sha256:" + "0" * 64,
                )
            )

        health = await adapter.health()
        assert (health.source_id, health.adapter_id) == (
            registration.id,
            registration.adapter_id,
        )

        rebuilt = source_type.from_registration(
            registration, secret_provider=secret_provider
        )
        rebuilt_adapter = await rebuilt.open(
            agent_id=CONTRACT_AGENT_ID,
            attached_at=CONTRACT_NOW,
            clock=lambda: CONTRACT_NOW,
        )
        try:
            assert rebuilt_adapter.registration == registration
        finally:
            await rebuilt_adapter.close()
    finally:
        await adapter.close()

    await adapter.close()
    with pytest.raises(SourceClosedError):
        await adapter.discover(_request(adapter, "sync-contract-closed"))
    with pytest.raises(SourceClosedError):
        await adapter.inspect(
            ResourceRef(
                agent_id=CONTRACT_AGENT_ID,
                source_id=registration.id,
                resource_id=resource.id,
                native_identity=resource.native_identity,
                kind=resource.kind,
            )
        )


def _rows(projection: BoundedResultProjection) -> list[str]:
    """Rows as an order-free multiset of canonical JSON text."""

    return sorted(canonical_json(row.to_dict()) for row in projection.rows)


async def assert_tabular_read_contract(
    agent: Agent,
    *,
    source_id: str,
    resource_id: str,
    columns: tuple[str, ...],
    key_column: str,
    range_column: str,
) -> None:
    """Read one attached resource through its type's TABULAR_READ backend.

    The resource needs at least three rows, a ``key_column`` and a numeric
    ``range_column``. Expected filtered rows are computed here from the
    unfiltered read, independently of the backend's implementation.
    """

    (registration,) = [
        item for item in await agent.list_sources() if item.id == source_id
    ]
    backends = agent._embedded._source_backends[SourceOperation.TABULAR_READ]
    backend = cast(TabularReadBackend, backends[registration.adapter_id])
    (resource,) = [
        item
        for item in await agent.list_catalog_resources(source_id=source_id)
        if item.id == resource_id
    ]

    async def read(
        *filters: TabularFilter,
        limit: int = 10_000,
        max_bytes: int = 4 * 1_024 * 1_024,
        revision: str = resource.current_revision,
    ) -> BoundedResultProjection:
        projection = await backend.read_tabular(
            ReadTabularRequest(
                agent_id=agent.id,
                source_id=source_id,
                resource_id=resource_id,
                expected_revision=revision,
                columns=columns,
                filters=filters,
                limit=limit,
                max_bytes=max_bytes,
                timeout_seconds=5.0,
            )
        )
        assert isinstance(projection, BoundedResultProjection)
        assert all(set(row) == set(columns) for row in projection.rows)
        return projection

    full = await read()
    assert not full.truncated
    rows = [cast(Mapping[str, object], row.to_dict()) for row in full.rows]
    assert len(rows) >= 3, "the tabular contract needs at least three rows"

    def expect(predicate: Callable[[Mapping[str, object]], bool]) -> list[str]:
        return sorted(canonical_json(row) for row in rows if predicate(row))

    key = rows[0][key_column]
    assert _rows(await read(TabularFilter(key_column, "eq", key))) == expect(
        lambda row: row[key_column] == key
    )
    keys = (key, next(row[key_column] for row in rows if row[key_column] != key))
    assert _rows(await read(TabularFilter(key_column, "in", keys))) == expect(
        lambda row: row[key_column] in keys
    )
    values = sorted(cast(float, row[range_column]) for row in rows)
    low, high = values[len(values) // 2], values[-1]
    assert _rows(
        await read(
            TabularFilter(range_column, "gte", low),
            TabularFilter(range_column, "lt", high),
        )
    ) == expect(lambda row: low <= cast(float, row[range_column]) < high)
    assert _rows(await read(TabularFilter(range_column, "gt", low))) == expect(
        lambda row: cast(float, row[range_column]) > low
    )
    assert _rows(await read(TabularFilter(range_column, "lte", low))) == expect(
        lambda row: cast(float, row[range_column]) <= low
    )

    by_rows = await read(limit=len(rows) - 1)
    assert by_rows.truncated and "row_limit" in by_rows.truncation_reasons
    by_bytes = await read(max_bytes=2)
    assert by_bytes.truncated and "byte_limit" in by_bytes.truncation_reasons

    with pytest.raises(DaitaError) as stale:
        await read(revision="sha256:" + "0" * 64)
    assert stale.value.error_code == "query_binding_stale"


__all__ = [
    "CONTRACT_AGENT_ID",
    "CONTRACT_NOW",
    "assert_source_type_contract",
    "assert_tabular_read_contract",
]
