from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import (
    DiscoveryRequest,
    ResourceAdapter,
    ResourceRef,
    SourceHealth,
    SourceRegistration,
    SourceStore,
)
from daita.catalog import ResourceKind
from daita.capabilities import ExtensionDeclarations

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


def test_source_registration_has_stable_identity_and_frozen_configuration() -> None:
    options = {"query_only": True}
    configuration: dict[str, object] = {
        "path": "/data/example.db",
        "options": options,
    }
    first = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity="/data/example.db",
        display_name="Example",
        configuration=configuration,
        attached_at=NOW,
    )
    renamed = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity="/data/example.db",
        display_name="Renamed",
        configuration={"path": "/data/example.db"},
        attached_at=NOW + timedelta(days=1),
    )
    options["query_only"] = False

    assert first.id == renamed.id
    assert isinstance(first.configuration, FrozenJsonObject)
    stored_options = first.configuration["options"]
    assert isinstance(stored_options, FrozenJsonObject)
    assert stored_options["query_only"] is True
    with pytest.raises(FrozenInstanceError):
        first.display_name = "changed"  # type: ignore[misc]


def test_source_registration_detach_is_chronological_and_irreversible() -> None:
    registration = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id="sqlite",
        native_identity="/data/example.db",
        display_name="Example",
        configuration={},
        attached_at=NOW,
    )

    detached = registration.detach(NOW + timedelta(seconds=1))

    assert detached.detached_at == NOW + timedelta(seconds=1)
    with pytest.raises(ValueError, match="before attachment"):
        registration.detach(NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="already detached"):
        detached.detach(NOW + timedelta(seconds=2))


def test_request_reference_and_health_records_fail_closed() -> None:
    request = DiscoveryRequest(
        agent_id="agent-1",
        source_id="source-1",
        sync_id="sync-1",
        requested_at=NOW,
    )
    assert request.max_resources == 1_000

    with pytest.raises(ValueError, match="max_resources"):
        replace(request, max_resources=0)
    with pytest.raises(ValueError, match="stable identity"):
        ResourceRef(
            agent_id="agent-1",
            source_id="source-1",
            resource_id="wrong",
            native_identity="main.orders",
            kind=ResourceKind.TABLE,
        )
    with pytest.raises(ValueError, match="requires error_code"):
        SourceHealth(
            agent_id="agent-1",
            source_id="source-1",
            adapter_id="sqlite",
            healthy=False,
            checked_at=NOW,
        )


def test_resource_adapter_and_source_store_are_structural_protocols() -> None:
    class AdapterShape:
        registration = object()

        async def discover(self, request: object) -> object:
            return request

        async def inspect(self, resource: object) -> object:
            return resource

        async def health(self) -> object:
            return object()

        def declarations(self) -> ExtensionDeclarations:
            return ExtensionDeclarations()

        async def close(self) -> None:
            return None

    class StoreShape:
        async def register_source(self, registration: object) -> object:
            return registration

        async def load_source(self, agent_id: str, source_id: str) -> None:
            return None

        async def list_sources(self, agent_id: str) -> tuple[()]:
            return ()

        async def detach_source(
            self,
            agent_id: str,
            source_id: str,
            detached_at: datetime,
        ) -> None:
            return None

    assert isinstance(AdapterShape(), ResourceAdapter)
    assert isinstance(StoreShape(), SourceStore)
