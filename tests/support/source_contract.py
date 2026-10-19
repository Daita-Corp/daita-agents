"""Reusable contract checks that every registered source type must satisfy."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from daita.adapters.models import DiscoveryRequest, ResourceRef
from daita.adapters.protocols import (
    ResourceAdapter,
    ResourceSource,
    SourceClosedError,
    StaleResourceError,
)
from daita.adapters.registry import SourceType
from daita.catalog.models import FacetKind, SourceCatalogSnapshot, TabularFacet
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


__all__ = ["CONTRACT_AGENT_ID", "CONTRACT_NOW", "assert_source_type_contract"]
